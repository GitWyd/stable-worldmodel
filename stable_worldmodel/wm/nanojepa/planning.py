"""Planning utilities for NanoJEPA.

Two surfaces, mirroring the tutorial's "first principles, then graduate" arc:

* :func:`latent_shooting` — a hand-written random-shooting planner in latent
  space. Read this first: it *is* the idea behind model-predictive control,
  written out in ~20 lines.
* :func:`build_cem_policy` — wires NanoJEPA's ``get_cost`` into swm's
  :class:`~stable_worldmodel.solver.CEMSolver` +
  :class:`~stable_worldmodel.policy.WorldModelPolicy`, so the same latent cost
  is optimised by a real CEM loop and evaluated with ``World.evaluate``.
"""

import numpy as np
import torch
from torchvision.transforms import v2

from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver


def _to_chw01(image) -> torch.Tensor:
    """Accept HWC uint8 / CHW tensor and return CHW float32 in [0, 1]."""
    t = torch.as_tensor(np.asarray(image))
    if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC -> CHW
        t = t.permute(2, 0, 1)
    return t.float().div(255.0) if t.dtype == torch.uint8 else t.float()


@torch.no_grad()
def latent_shooting(
    model,
    image,
    proprio,
    goal_image,
    *,
    action_sampler,
    num_candidates: int = 256,
    horizon: int = 10,
    device: str | torch.device = 'cpu',
) -> np.ndarray:
    """Random-shooting planner in latent space (teaching baseline).

    Sample ``num_candidates`` random action sequences, roll each forward in the
    model's latent space, and return the sequence whose predicted final state
    is closest to the goal embedding — all without touching the simulator.

    Args:
        model: a trained :class:`NanoJEPA`.
        image: current observation image (HWC uint8 or CHW tensor).
        proprio: current proprio vector ``[P]``.
        goal_image: goal observation image.
        action_sampler: ``callable() -> action [A]`` (e.g. ``env.action_space
            .sample``).
        num_candidates: number of random action sequences to score.
        horizon: planning horizon (sequence length).
        device: compute device.

    Returns:
        Best action sequence as ``np.ndarray [horizon, A]``.
    """
    model.eval()
    img = _to_chw01(image).unsqueeze(0).to(device)  # (1, 3, H, W)
    pro = torch.as_tensor(np.asarray(proprio)).float().unsqueeze(0).to(device)
    goal = _to_chw01(goal_image).unsqueeze(0).to(device)

    z_goal = model._encode_goal_z(goal)  # (1, D)

    candidates = torch.as_tensor(
        np.stack(
            [
                np.stack([action_sampler() for _ in range(horizon)])
                for _ in range(num_candidates)
            ]
        ),
        dtype=torch.float32,
        device=device,
    )  # (C, H, A)

    ctx = model.context_encoder(img, patch_indices=None)  # (1, N, D)
    proprio_token = model.proprio_encoder(pro)  # (1, 1, D)
    ctx = ctx.expand(num_candidates, -1, -1)
    proprio_token = proprio_token.expand(num_candidates, -1, -1)

    z = model._latent_rollout(ctx, proprio_token, candidates)  # (C, D)
    dist = (z - z_goal).pow(2).sum(dim=-1)  # (C,)
    best = candidates[dist.argmin()]  # (H, A)
    return best.cpu().numpy()


def build_cem_policy(
    model,
    *,
    horizon: int,
    receding_horizon: int = 1,
    num_samples: int = 300,
    n_steps: int = 10,
    topk: int = 30,
    image_size: int = 64,
    device: str | torch.device = 'cpu',
) -> WorldModelPolicy:
    """Wire NanoJEPA into a CEM-based :class:`WorldModelPolicy`.

    Uses a plain scale+resize image transform (NanoJEPA trains from scratch on
    [0, 1] 64x64 images — *no* ImageNet normalisation, unlike the pretrained
    backbones in ``scripts/plan/eval_wm.py``) and raw proprio (``process={}``).
    """
    tf = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=image_size),
        ]
    )
    solver = CEMSolver(
        model=model,
        num_samples=num_samples,
        n_steps=n_steps,
        topk=topk,
        device=device,
    )
    config = PlanConfig(horizon=horizon, receding_horizon=receding_horizon)
    return WorldModelPolicy(
        solver=solver,
        config=config,
        transform={'pixels': tf, 'goal': tf},
        process={},  # raw proprio: NanoJEPA trains on unnormalised proprio
    )


__all__ = ['latent_shooting', 'build_cem_policy']
