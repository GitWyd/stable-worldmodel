# NanoJEPA — frozen interface contracts (ratified T0, refined by T0-SPIKE)

These are normative. Wave-1/2/3 tickets build against them. Any change after
ratification needs a note here + a ping to dependent tickets. Shapes use
`B`=batch/envs, `S`=planning samples, `T`=history frames, `N`=patches, `D`=embed_dim,
`P`=proprio dim (25), `A`=action dim (4), `H`=planning horizon.

## C0 — swm dataset sample  *(refined R1, verified in `data/formats/lance.py`)*
`swm.data.load_dataset(path, num_steps=2, frameskip=1,
keys_to_load=["pixels","proprio","action"])[i]` returns a dict of 2-step windows:
- `pixels`  → `torch.uint8 [2, 3, H, W]`  **CHW, RGB** (JPEG-decoded via
  `decode_jpeg(...).permute(2,0,1)`). **Not HWC, not numpy.**
- `proprio` → `torch.float32 [2, P]`
- `action`  → `torch.float32 [2, A]`  (reshaped to `[num_steps, -1]`)

Windows never cross an episode reset (verified `Dataset.clip_indices`). `(t,t+1)` =
window index `[0]` / `[1]`. Images are JPEG q95 (near-lossless) — a small,
documented train/collect fidelity note.

## C1 — `NanoJEPA` public API  *(refined R2)*
```python
NanoJEPA(image_size=64, patch_size=8, embed_dim=192,
         encoder_depth=6, encoder_heads=3, predictor_depth=4,
         predictor_heads=3, obs_dim=25, act_dim=4)
forward(batch: dict) -> {"predicted": [B,n_tgt,D], "target": [B,n_tgt,D]}
encode(images: [B,3,H,W], proprios: [B,P]) -> [B, D]          # fused, for probes
predict_next(images: [B,3,H,W], proprios: [B,P], actions: [B,A]) -> [B, D]  # pooled 1-step
update_target_encoder(momentum: float) -> None
rollout(info_dict: dict, action_candidates: [B,S,H,A]) -> info_dict  # adds 'predicted_z'
criterion(info_dict: dict, action_candidates) -> [B, S]
get_cost(info_dict: dict, action_candidates: [B,S,H,A]) -> [B, S]    # Costable
@classmethod from_config(cfg: dict) -> "NanoJEPA"
@classmethod from_checkpoint(path, device="cpu") -> ("NanoJEPA", dict)
save_checkpoint(path, *, epoch, optimizer=None, history=None, vicreg=True) -> None
```
`forward`/`encode`/`predict_next`/`update_target_encoder` are unchanged from upstream
NanoJEPA. `rollout`/`criterion`/`get_cost`/`from_*` are net-new (planning + swm bridge).

## C2 — training batch dict  (output of `MaskCollator.__call__`) — unchanged from upstream
```
image_t      Float[B,3,H,W]∈[0,1]   proprio_t   Float[B,P]   action_t Float[B,A]
image_tp1    Float[B,3,H,W]∈[0,1]   proprio_tp1 Float[B,P]
context_indices Long[n_ctx] (shared) target_indices Long[n_tgt] (shared)
```

## C3 — planning `info_dict`  *(refined R2/R3, verified in `solver/cem.py`,
## `policy.py`, `world/world.py::_extract_init_goal`, `scripts/plan/eval_wm.py`)*
The solver expands each per-env info tensor to a leading `[B, S, ...]`; per-env
tensors carry a history dim `T`. After the policy `transform`, `get_cost` receives:
- `info_dict["pixels"]`  → `Float[B, S, T, 3, H, W]` in [0,1] (CHW; `[:, :, -1]` = current)
- `info_dict["proprio"]` → `Float[B, S, T, P]` (raw; `process={}`)
- `info_dict["goal"]`    → goal **image** `Float[B, S, T, 3, H, W]` (rendered frame at
  `start_step + goal_offset`; injected by dataset-driven `World.evaluate`)
- `info_dict["goal_proprio"]` → `Float[B, S, T, P]` (present but unused by `get_cost`)
- `action_candidates`    → `Float[B, S, H, A]`

`get_cost` returns **2-D** `Tensor[B, S]` (asserted by `solver/cem.py:215-221`).
Pattern mirrors `wm/pldm/pldm.py::get_cost`: assert `goal`; encode goal; `rollout`;
`criterion` → `[B,S]`.

## C4 — losses (`losses.py`) — byte-faithful to upstream NanoJEPA
```python
jepa_loss(predicted, target) -> Tensor          # smooth_l1(predicted, target.detach())
variance_loss(z) -> Tensor                       # relu(1 - std).mean()
covariance_loss(z) -> Tensor                     # off-diagonal cov^2 / D
```
Docstrings cross-reference the library twin `swm.wm.loss.VCReg`.

## C5 — metrics & probes (`metrics.py`)  *(refined R4)*
```python
compute_collapse_metrics(z) -> {"embedding_std_mean": float, "effective_rank": float}
linear_probe(model, dataset, device, target_slice=slice(0,3), epochs=20,
             lr=1e-3, batch_size=256, num_samples=None) -> {"mse": float, "r2": float}
cross_modal_eval(model, dataset, device, num_samples=1000,
                 batch_size=256) -> {"cos_vision": float, "cos_proprio": float}
```
`linear_probe`/`cross_modal_eval` consume a swm `Dataset` via `TransitionView.single_step(i)`
(end-effector target = `proprio[:3]`). Probes use the **fused** `encode()` space; planning
cost (C3) uses the **layer-normed target-encoder patch space** (predictor output space).

## C6 — config schema (flat YAML; Hydra mirrors it)
```yaml
env:    {name: swm/FetchPush-v3, image_size: 64, num_episodes: 10000, max_steps: 50}
model:  {patch_size: 8, embed_dim: 192, encoder_depth: 6, encoder_heads: 3,
         predictor_depth: 4, predictor_heads: 3}
masking:{num_targets: 4, target_scale: [0.15,0.2], target_aspect_ratio: [0.75,1.5],
         min_context_patches: 16}
training:{epochs, batch_size, lr, lr_min, warmup_epochs, weight_decay_start,
          weight_decay_end, ema_momentum_start, ema_momentum_end, grad_clip,
          variance_weight, covariance_weight, vicreg: true, log_every, save_every,
          mps_cache_every}
data:   {path: <name>.lance, num_workers: 0, pin_memory: false}
```
`--no-vicreg` → `training.vicreg=false` (zeros var/cov weights). Env id is
`swm/FetchPush-v3`; `data.path` is a lance dataset name.

## C7 — `TransitionView` (`masking.py`)  *(refined R1)*
```python
class TransitionView(torch.utils.data.Dataset):
    """swm 2-step Dataset -> NanoJEPA tuple. Pixels already CHW uint8; just /255."""
    __getitem__(i) -> (image_t[3,H,W]∈[0,1], proprio_t[P], action_t[A],
                       image_tp1[3,H,W]∈[0,1], proprio_tp1[P])
    single_step(i) -> (image[3,H,W]∈[0,1], proprio[P])   # for probes/cross-modal
```

## C8 — checkpoint (self-contained teaching path + additive Hydra)
`torch.save({"epoch","model_state_dict","optimizer_state_dict","history","vicreg",
"model_cfg"}, path)`; `NanoJEPA.from_checkpoint(path, device) -> (model, ckpt)`.
Graduation path additionally supports `swm.wm.utils.save_pretrained` /
`swm.wm.utils.load_pretrained` (Hydra-config instantiation).

## C9 — data collection (T4)
`World("swm/FetchPush-v3", num_envs, image_shape=(64,64), max_episode_steps=50)` +
`RandomPolicy` → `world.collect(path=..., episodes=..., format="lance")`. Lance columns
include `pixels` (64×64×3), `proprio` (≥25), `action` (4). T4 asserts dims at runtime.

## C10 — `get_cost` ↔ solver wiring (`planning.py`, T8)  *(refined R3)*
```python
build_cem_policy(model, *, horizon, receding_horizon=1, num_samples=300,
                 n_steps=10, topk=30, image_size=64, device="cpu") -> WorldModelPolicy
```
Builds `CEMSolver(model=model, num_samples=..., n_steps=..., topk=...)` +
`WorldModelPolicy(solver, PlanConfig(horizon, receding_horizon),
transform={'pixels': tf, 'goal': tf})` with
`tf = v2.Compose([ToImage(), ToDtype(float32, scale=True), Resize(image_size)])`
(**no ImageNet normalize**), `process={}` (raw proprio). Plus a hand-written
`latent_shooting(model, ...)` teaching baseline.

## C11 — packaging & style
`wm/nanojepa/__init__.py` exports `NanoJEPA, MaskCollator, TransitionView, make_loader,
jepa_loss, variance_loss, covariance_loss, compute_collapse_metrics, linear_probe,
cross_modal_eval, build_cem_policy, latent_shooting`. `wm/__init__.py` gains
`from .nanojepa import *`. Ruff: line-length 79, 4-space, **single quotes**, py310+.
```
