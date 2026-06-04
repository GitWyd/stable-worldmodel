---
title: 1. Environment & Data
summary: Collect FetchPush transitions and turn them into I-JEPA training batches
sidebar_title: 1. Env & Data
---

# NanoJEPA — 1. Environment & Data

NanoJEPA is an educational, from-scratch multimodal JEPA (Joint-Embedding
Predictive Architecture) for robotics, in the spirit of NanoGPT. It lives in
`stable_worldmodel.wm.nanojepa` and is meant to be read top-to-bottom. This
chapter is the first stop in that curriculum:

1. **Env & Data** (this page) — collect FetchPush, load 2-step windows, mask them.
2. [Model](2_model.md) — the from-scratch ViT, modality encoders, EMA target.
3. [Train & Collapse](3_train_and_collapse.md) — the loss and the collapse experiment.
4. [Evaluate](4_evaluate.md) — probes, cross-modal tests, collapse metrics.
5. [Planning](5_planning.md) — latent shooting, then CEM + a real success rate.

The whole point of a JEPA is to predict the *future* of the world in **embedding
space** rather than in pixel space. To learn that, we only need diverse
experience of how the world changes in response to actions — no expert demos, no
rewards. A random policy on `swm/FetchPush-v3` is enough.

## Collect a transition dataset

`stable_worldmodel` handles all of the env wrapping, rendering, and writing. We
just point a `World` at FetchPush, attach a `RandomPolicy`, and call
`World.collect`. The script `scripts/data/collect_nanojepa.py` is exactly this:

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

world = swm.World(
    'swm/FetchPush-v3',
    num_envs=8,
    image_shape=(64, 64),
    max_episode_steps=50,
)
world.set_policy(RandomPolicy(seed=0))
world.collect(
    path='nanojepa_fetchpush_quick.lance',
    episodes=500,
    seed=0,
    format='lance',
)
world.close()
```

From the command line:

```bash
python scripts/data/collect_nanojepa.py \
    --out nanojepa_fetchpush_quick.lance \
    --episodes 500 --num-envs 8
```

The output is a [Lance](https://lancedb.github.io/lance/) dataset with three
columns relevant to us:

| column    | shape        | meaning                                  |
| --------- | ------------ | ---------------------------------------- |
| `pixels`  | `64x64x3`    | RGB camera frame                         |
| `proprio` | `>=25`-d     | robot joint / end-effector state         |
| `action`  | `4`-d        | the action that produced the transition  |

The collection script ends by re-loading the dataset and asserting these dims
(the C0/C9 contract), so a successful run prints something like:

```
  pixels  (2, 3, 64, 64) torch.uint8
  proprio (2, 25) torch.float32
  action  (2, 4) torch.float32
  OK — N transition windows available
```

!!! note "You usually do not have to collect manually"
    The trainer (`scripts/train/nanojepa.py`) calls `ensure_dataset`: if the
    `data.path` lance file does not exist, it collects it for you with the same
    `World` + `RandomPolicy` recipe before training starts.

## Load 2-step windows

A JEPA transition needs both `t` and `t+1`. `swm.data.load_dataset` gives us
exactly that with `num_steps=2`: each sample is a dict of 2-step windows that
**never cross an episode reset**.

```python
import stable_worldmodel as swm

ds = swm.data.load_dataset(
    'nanojepa_fetchpush_quick.lance',
    num_steps=2,
    keys_to_load=['pixels', 'proprio', 'action'],
)

sample = ds[0]
sample['pixels'].shape    # torch.Size([2, 3, 64, 64])  uint8, CHW, RGB
sample['proprio'].shape   # torch.Size([2, 25])         float32
sample['action'].shape    # torch.Size([2, 4])          float32
```

Index `[0]` is `t`, index `[1]` is `t+1`. Note the pixels are already **CHW
uint8** (JPEG-decoded), not HWC numpy — so the adapter only has to divide by 255.

## TransitionView: dataset → model tuple

NanoJEPA's model is written against a plain transition tuple. `TransitionView`
(`wm/nanojepa/masking.py`) is the single bridge from a swm 2-step dataset to that
tuple. It asserts `num_steps == 2` and just normalises pixels:

```python
from stable_worldmodel.wm.nanojepa import TransitionView

view = TransitionView(ds)
image_t, proprio_t, action_t, image_tp1, proprio_tp1 = view[0]
# image_t / image_tp1 : float32 [3, 64, 64] in [0, 1]
# proprio_t / proprio_tp1 : float32 [25]
# action_t : float32 [4]   (the action taken at t)

# A second accessor used later by the probes (chapter 4):
img, pro = view.single_step(0)   # (image [3,H,W] in [0,1], proprio [P])
```

`action[0]` is the action taken *at* `t` — `World.collect` aligns actions to
transitions, so it is the action that drives `image_t -> image_tp1`. That
alignment is what lets the predictor learn dynamics.

## MaskCollator: I-JEPA multiblock masking

The last data ingredient is the masking strategy. NanoJEPA uses
[I-JEPA](https://arxiv.org/abs/2301.08243) multiblock masking: sample a few
rectangular **target** blocks in the patch grid, and use the complement as
**context**. The context encoder only ever sees context patches; the predictor
must fill in the targets. The spatial-block structure forces the model to learn
local coherence instead of memorising individual patches.

`MaskCollator` is a collate function, so the masks are sampled once per batch and
*shared* across the batch (different image content, same patch positions) — that
keeps the context/target tensors a uniform size.

```python
from stable_worldmodel.wm.nanojepa import MaskCollator, make_loader

# 64x64 image, 8x8 patches -> an 8x8 = 64-patch grid.
collate = MaskCollator(
    patch_size=8,
    image_size=64,
    num_targets=4,
    target_scale=(0.15, 0.2),
    target_aspect_ratio=(0.75, 1.5),
    min_context_patches=16,
)
```

The easiest way to get a ready-to-train `DataLoader` is `make_loader`, which wraps
the dataset in a `TransitionView` and attaches a `MaskCollator` for you:

```python
loader = make_loader(
    ds,
    batch_size=128,
    masking_cfg={
        'patch_size': 8,
        'image_size': 64,
        'num_targets': 4,
        'target_scale': (0.15, 0.2),
        'target_aspect_ratio': (0.75, 1.5),
        'min_context_patches': 16,
    },
    num_workers=0,
    pin_memory=False,
)

batch = next(iter(loader))
batch.keys()
# dict_keys(['image_t', 'proprio_t', 'action_t', 'image_tp1',
#            'proprio_tp1', 'context_indices', 'target_indices'])
```

The batch dict (the C2 contract) is what the model's `forward` consumes:

| key               | shape                  | notes                          |
| ----------------- | ---------------------- | ------------------------------ |
| `image_t`         | `[B, 3, H, W]` in [0,1] | masked input (context)         |
| `proprio_t`       | `[B, P]`               | current proprio                |
| `action_t`        | `[B, A]`               | action at `t`                  |
| `image_tp1`       | `[B, 3, H, W]` in [0,1] | full next image (target side)  |
| `proprio_tp1`     | `[B, P]`               | next proprio (unused by loss)  |
| `context_indices` | `[n_ctx]` long         | shared patch positions         |
| `target_indices`  | `[n_tgt]` long         | shared patch positions         |

The masking config matches the `masking:` block in
`scripts/train/config/nanojepa_flat/quick.yaml`, so everything you set up here is
exactly what the trainer uses.

Next: [the model](2_model.md) — what those context/target patches flow through.
