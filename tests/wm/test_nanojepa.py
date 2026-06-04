"""Tests for stable_worldmodel.wm.nanojepa.

All unit tests run on CPU with only torch + numpy (no mujoco / gym-robotics /
lance). One guarded end-to-end smoke test exercises the full collect ->
load_dataset -> train -> CEM plan path and is skipped when the env stack is
absent. Shapes are kept tiny (16x16 images, 8px patches -> a 2x2 patch grid)
so the whole module runs in a second or two.
"""

import os
import random

import numpy as np
import pytest
import torch

from stable_worldmodel.wm.nanojepa import (
    MaskCollator,
    NanoJEPA,
    TransitionView,
    compute_collapse_metrics,
    covariance_loss,
    jepa_loss,
    make_loader,
    variance_loss,
)
from stable_worldmodel.wm.nanojepa.module import (
    ActionEncoder,
    Predictor,
    ProprioEncoder,
    VisionEncoder,
    sinusoidal_position_embedding_2d,
)


# ---------------------------------------------------------------------------
# Shared fixtures / fakes
# ---------------------------------------------------------------------------

D = 192  # embed_dim used throughout (divisible by 4 for 2D sincos)


@pytest.fixture(autouse=True)
def _seed():
    """Make mask sampling + weight init deterministic per test."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


class FakeWindowDataset:
    """In-memory stand-in for a swm 2-step dataset (C0 contract).

    ``ds[i]`` returns a dict of 2-step windows: ``pixels`` uint8 [2,3,H,W]
    (CHW), ``proprio`` f32 [2,P], ``action`` f32 [2,A]. Mirrors what
    ``swm.data.load_dataset(num_steps=2)`` yields, without needing lance.
    """

    num_steps = 2

    def __init__(
        self, n: int = 8, image_size: int = 16, p: int = 25, a: int = 4
    ):
        self.n = n
        self.image_size = image_size
        self.p = p
        self.a = a

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        s = self.image_size
        return {
            'pixels': torch.randint(0, 255, (2, 3, s, s), dtype=torch.uint8),
            'proprio': torch.randn(2, self.p),
            'action': torch.randn(2, self.a),
        }


def _small_model() -> NanoJEPA:
    return NanoJEPA(
        image_size=16,
        patch_size=8,
        embed_dim=D,
        encoder_depth=2,
        encoder_heads=3,
        predictor_depth=2,
        predictor_heads=3,
    )


def _make_batch(batch_size: int = 4) -> dict:
    """Build a real C2 batch by running the MaskCollator over a fake set.

    The default ``min_context_patches=16`` is sized for a 64x64 / 8px grid
    (64 patches); on this tiny 16x16 / 8px grid there are only 4 patches, so
    we pass a feasible ``min_context_patches`` to keep at least one context
    patch while still exercising the disjoint context/target split.
    """
    ds = FakeWindowDataset(n=batch_size)
    view = TransitionView(ds)
    collate = MaskCollator(patch_size=8, image_size=16, min_context_patches=1)
    return collate([view[i] for i in range(batch_size)])


# ---------------------------------------------------------------------------
# 1. module shapes
# ---------------------------------------------------------------------------


def test_vision_encoder_full_grid_shape():
    enc = VisionEncoder(image_size=16, patch_size=8, embed_dim=D)
    out = enc(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4, D)


def test_vision_encoder_patch_indices_shape():
    enc = VisionEncoder(image_size=16, patch_size=8, embed_dim=D)
    idx = torch.arange(2)
    out = enc(torch.randn(2, 3, 16, 16), patch_indices=idx)
    assert out.shape == (2, 2, D)


def test_proprio_encoder_shape():
    out = ProprioEncoder(obs_dim=25, embed_dim=D)(torch.randn(2, 25))
    assert out.shape == (2, 1, D)


def test_action_encoder_shape():
    out = ActionEncoder(act_dim=4, embed_dim=D)(torch.randn(2, 4))
    assert out.shape == (2, 1, D)


def test_predictor_output_shape():
    pred = Predictor(embed_dim=D, depth=2, num_heads=3, num_patches=4)
    b, n_ctx, n_tgt = 2, 3, 2
    context_tokens = torch.randn(b, n_ctx, D)
    proprio_token = torch.randn(b, 1, D)
    action_token = torch.randn(b, 1, D)
    target_indices = torch.tensor([0, 1], dtype=torch.long)
    out = pred(context_tokens, proprio_token, action_token, target_indices)
    assert out.shape == (b, n_tgt, D)


def test_sinusoidal_pos_embed_shape():
    pe = sinusoidal_position_embedding_2d(2, D)
    assert pe.shape == (4, D)


# ---------------------------------------------------------------------------
# 2. losses
# ---------------------------------------------------------------------------


def test_variance_loss_collapsed_is_one():
    # All-zeros -> std 0 -> relu(1 - 0) = 1.
    loss = variance_loss(torch.zeros(256, D))
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)


def test_variance_loss_spread_is_zero():
    # std ~ 5 >> 1 -> relu(1 - std) = 0.
    loss = variance_loss(torch.randn(256, D) * 5.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-3)


def test_covariance_loss_finite_nonnegative_scalar():
    loss = covariance_loss(torch.randn(256, D))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_jepa_loss_detaches_target():
    predicted = torch.randn(8, 4, D, requires_grad=True)
    target = torch.randn(8, 4, D, requires_grad=True)
    loss = jepa_loss(predicted, target)
    loss.backward()
    # Gradient flows to predicted but never to the (detached) target.
    assert predicted.grad is not None
    assert target.grad is None


# ---------------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------------


def test_collapse_metrics_high_rank_for_random():
    m = compute_collapse_metrics(torch.randn(512, 64))
    assert 'embedding_std_mean' in m
    assert 'effective_rank' in m
    assert m['effective_rank'] > 20


def test_collapse_metrics_low_rank_for_collapsed():
    # ``compute_collapse_metrics`` mean-centres before the SVD, so a constant
    # offset is removed and only the *variation* between samples sets the rank.
    # Collapse therefore means the variation is (near) rank-1: every embedding
    # lies on a single direction ``v`` (scaled per-sample) plus tiny noise.
    v = torch.randn(64)
    a = torch.randn(512, 1)
    z = a * v.unsqueeze(0) + 1e-6 * torch.randn(512, 64)
    m = compute_collapse_metrics(z)
    assert m['effective_rank'] < 5


# ---------------------------------------------------------------------------
# 4. masking / collator / loader
# ---------------------------------------------------------------------------


def test_transition_view_single_step_shapes_and_range():
    view = TransitionView(FakeWindowDataset(n=4))
    img, pro = view.single_step(0)
    assert img.shape == (3, 16, 16)
    assert img.dtype == torch.float32
    assert pro.shape == (25,)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_transition_view_getitem_tuple():
    view = TransitionView(FakeWindowDataset(n=4))
    img_t, pro_t, act_t, img_tp1, pro_tp1 = view[0]
    assert img_t.shape == (3, 16, 16)
    assert img_t.dtype == torch.float32
    assert img_t.min() >= 0.0 and img_t.max() <= 1.0
    assert pro_t.shape == (25,) and pro_t.dtype == torch.float32
    assert act_t.shape == (4,) and act_t.dtype == torch.float32
    assert img_tp1.shape == (3, 16, 16)
    assert img_tp1.min() >= 0.0 and img_tp1.max() <= 1.0
    assert pro_tp1.shape == (25,)


def test_transition_view_rejects_wrong_num_steps():
    bad = FakeWindowDataset(n=4)
    bad.num_steps = 1
    with pytest.raises(AssertionError):
        TransitionView(bad)


C2_KEYS = {
    'image_t',
    'proprio_t',
    'action_t',
    'image_tp1',
    'proprio_tp1',
    'context_indices',
    'target_indices',
}


def test_collator_emits_exact_c2_keys():
    batch = _make_batch(batch_size=4)
    assert set(batch.keys()) == C2_KEYS


def test_collator_image_t_float_in_unit_range():
    batch = _make_batch(batch_size=4)
    img = batch['image_t']
    assert img.shape == (4, 3, 16, 16)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_collator_context_target_disjoint_and_cover_grid():
    # 2x2 grid -> exactly 4 patches; collator masks are shared per batch.
    batch = _make_batch(batch_size=4)
    ctx = batch['context_indices']
    tgt = batch['target_indices']
    assert ctx.dtype == torch.long and tgt.dtype == torch.long
    ctx_set = set(ctx.tolist())
    tgt_set = set(tgt.tolist())
    assert ctx_set.isdisjoint(tgt_set)
    assert ctx_set | tgt_set == set(range(4))


def test_collator_respects_min_context_when_feasible():
    # min_context_patches feasible on a 4x4 grid (16 patches).
    ds = FakeWindowDataset(n=4, image_size=32)
    view = TransitionView(ds)
    collate = MaskCollator(patch_size=8, image_size=32, min_context_patches=4)
    batch = collate([view[i] for i in range(4)])
    assert len(batch['context_indices']) >= 4


def test_make_loader_yields_one_c2_batch():
    loader = make_loader(
        FakeWindowDataset(n=8),
        batch_size=4,
        masking_cfg={
            'patch_size': 8,
            'image_size': 16,
            'min_context_patches': 1,
        },
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert set(batch.keys()) == C2_KEYS
    assert batch['image_t'].shape == (4, 3, 16, 16)


# ---------------------------------------------------------------------------
# 5. model
# ---------------------------------------------------------------------------


def test_forward_predicted_and_target_shapes_match():
    model = _small_model()
    batch = _make_batch(batch_size=4)
    out = model(batch)
    n_tgt = batch['target_indices'].shape[0]
    assert out['predicted'].shape == (4, n_tgt, D)
    assert out['target'].shape == (4, n_tgt, D)


def test_target_encoder_params_frozen():
    model = _small_model()
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_update_target_encoder_moves_weights():
    model = _small_model()
    # Target starts as a deepcopy of context, so EMA is a no-op until the
    # context encoder diverges. Perturb the context first.
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    model.update_target_encoder(0.9)
    after = list(model.target_encoder.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_from_config_builds_model():
    cfg = {
        'env': {'image_size': 16},
        'model': {
            'patch_size': 8,
            'embed_dim': D,
            'encoder_depth': 2,
            'encoder_heads': 3,
            'predictor_depth': 2,
            'predictor_heads': 3,
        },
    }
    model = NanoJEPA.from_config(cfg)
    out = model(_make_batch(batch_size=4))
    assert out['predicted'].shape[-1] == D


def test_save_and_from_checkpoint_roundtrip(tmp_path):
    model = _small_model()
    path = tmp_path / 'nanojepa.pt'
    model.save_checkpoint(
        str(path),
        epoch=3,
        optimizer=None,
        history={'loss': [1.0]},
        vicreg=True,
    )
    loaded, ckpt = NanoJEPA.from_checkpoint(str(path), device='cpu')
    assert ckpt['epoch'] == 3
    assert ckpt['vicreg'] is True
    src = model.state_dict()
    dst = loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        torch.testing.assert_close(src[k], dst[k])


# ---------------------------------------------------------------------------
# 6. get_cost (planning contract C3)
# ---------------------------------------------------------------------------


def test_get_cost_returns_2d_shape():
    model = _small_model()
    b, s, t, h, a = 2, 3, 1, 5, 4
    info_dict = {
        'pixels': torch.rand(b, s, t, 3, 16, 16),
        'proprio': torch.randn(b, s, t, 25),
        'goal': torch.rand(b, s, t, 3, 16, 16),
    }
    action_candidates = torch.randn(b, s, h, a)
    cost = model.get_cost(info_dict, action_candidates)
    assert cost.shape == (b, s)


# ---------------------------------------------------------------------------
# 7. guarded end-to-end smoke (env stack required)
# ---------------------------------------------------------------------------


def test_end_to_end_smoke(tmp_path):
    # Opt-in: this test renders MuJoCo, which needs a working GL backend.
    # Headless machines without one can *hard-crash* (native segfault) rather
    # than raise a catchable exception, so it is skipped unless explicitly
    # enabled. Run with e.g. ``NANOJEPA_E2E=1 MUJOCO_GL=egl pytest ...``.
    if not os.environ.get('NANOJEPA_E2E'):
        pytest.skip('set NANOJEPA_E2E=1 (and a working MUJOCO_GL) to run')
    pytest.importorskip('gymnasium_robotics')
    pytest.importorskip('lance')

    import stable_worldmodel as swm
    from stable_worldmodel.policy import RandomPolicy
    from stable_worldmodel.wm.nanojepa import (
        build_cem_policy,
        covariance_loss as cov_loss,
        jepa_loss as j_loss,
        variance_loss as var_loss,
    )

    env_name = 'swm/FetchPush-v3'
    image_size = 16
    path = str(tmp_path / 'smoke.lance')

    # --- collect 2 episodes with a random policy --------------------------
    # MuJoCo headless GL rendering is environment-specific (no working GL
    # backend on some CI/dev machines). Skip cleanly on a catchable render
    # error rather than failing the suite.
    try:
        world = swm.World(
            env_name,
            num_envs=2,
            image_shape=(image_size, image_size),
            max_episode_steps=20,
        )
        world.set_policy(RandomPolicy(seed=0))
        world.collect(path=path, episodes=2, seed=0, format='lance')
        world.close()
    except (RuntimeError, ImportError, OSError) as exc:
        pytest.skip(f'env/render stack unavailable: {exc}')

    dataset = swm.data.load_dataset(
        path, num_steps=2, keys_to_load=['pixels', 'proprio', 'action']
    )
    assert len(dataset) > 0

    # --- build + train one tiny epoch (minimal manual loop) ---------------
    model = NanoJEPA(
        image_size=image_size,
        patch_size=8,
        embed_dim=D,
        encoder_depth=2,
        encoder_heads=3,
        predictor_depth=2,
        predictor_heads=3,
    )
    loader = make_loader(
        dataset,
        batch_size=2,
        masking_cfg={
            'patch_size': 8,
            'image_size': image_size,
            'min_context_patches': 1,
        },
        shuffle=True,
        num_workers=0,
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    model.train()
    for batch in loader:
        out = model(batch)
        pred, target = out['predicted'], out['target']
        loss = j_loss(pred, target) + var_loss(pred) + 0.04 * cov_loss(pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target_encoder(0.99)
        assert torch.isfinite(loss)

    # --- plan with CEM + dataset-driven evaluate --------------------------
    model.eval()
    policy = build_cem_policy(
        model,
        horizon=2,
        receding_horizon=1,
        num_samples=8,
        n_steps=2,
        topk=4,
        image_size=image_size,
        device='cpu',
    )
    eval_world = swm.World(
        env_name,
        num_envs=2,
        image_shape=(image_size, image_size),
        max_episode_steps=40,
    )
    eval_world.set_policy(policy)
    results = eval_world.evaluate(
        dataset=dataset,
        episodes_idx=[0, 1],
        start_steps=[0, 0],
        goal_offset=5,
        eval_budget=10,
    )
    eval_world.close()
    assert 'success_rate' in results
