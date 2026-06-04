---
title: 2. The Model
summary: A from-scratch ViT, modality encoders, predictor, and EMA target encoder
sidebar_title: 2. Model
---

# NanoJEPA — 2. The Model

The data from [chapter 1](1_env_and_data.md) flows through four hand-written
pieces in `wm/nanojepa/module.py` and the model that fuses them in
`wm/nanojepa/nanojepa.py`. Everything is deliberately readable: `stable_worldmodel`
ships production ViTs (`wm/pldm/module.py`, the HuggingFace `vit_hf` backbones),
but we re-implement a tiny one so you can read the whole encoder — including the
I-JEPA patch-index masking that the library backbones do not expose.

## The pieces (module.py)

**`VisionEncoder`** — a ViT-Tiny: `PatchEmbed` (a strided `Conv2d`) turns the
image into patch tokens, fixed 2D sin/cos positional embeddings are added, and a
stack of pre-norm `TransformerBlock`s processes them. The key detail is that
`patch_indices` selects a subset of patches *before* adding position — that is
exactly how I-JEPA context masking is realised:

```python
# only the visible context patches are encoded
context_tokens = vision_encoder(images, patch_indices=context_indices)
# patch_indices=None -> encode the full image (used for targets / probing)
full_tokens = vision_encoder(images, patch_indices=None)
```

Attention is a **manual QKV matmul** (no `scaled_dot_product_attention`) with
`.contiguous()` on every operand — `F.scaled_dot_product_attention` has known
bugs on the Apple MPS backend, and NanoJEPA is designed to run there.

**`ProprioEncoder`** and **`ActionEncoder`** — small MLPs that lift the 25-d
proprio vector and the 4-d action vector into the same `embed_dim`, each
returning a single `[B, 1, D]` token. The action token is what bridges the
temporal gap from `image_t` to `image_{t+1}`.

**`Predictor`** — a second (shallower) transformer. Its input sequence is
`[context_vision_tokens, proprio_token, action_token, mask_tokens]`. The mask
tokens are a single learned parameter, broadcast to `n_tgt` and given the target
patches' positional embeddings. After the transformer, only the last `n_tgt`
tokens (the mask-token outputs) are kept and projected — these are the
*predictions* for the masked target positions.

## The model (nanojepa.py)

`NanoJEPA` assembles a **context pathway** (trained) and a **target pathway**
(EMA, not trained):

```python
from stable_worldmodel.wm.nanojepa import NanoJEPA, count_parameters

model = NanoJEPA(
    image_size=64,
    patch_size=8,
    embed_dim=192,
    encoder_depth=6,
    encoder_heads=3,
    predictor_depth=4,
    predictor_heads=3,
    obs_dim=25,
    act_dim=4,
)
count_parameters(model)
# {'context_encoder': ..., 'predictor': ..., 'target_encoder': ...,
#  'total': ..., 'trainable': ...}   # ~4.6M trainable
```

You will usually build it straight from a config instead (the C6 schema from
chapter 1):

```python
import yaml

cfg = yaml.safe_load(open('scripts/train/config/nanojepa_flat/quick.yaml'))
model = NanoJEPA.from_config(cfg)
```

### The training forward pass

`forward` takes the C2 batch dict and returns predicted vs. target embeddings —
the comparison happens entirely in embedding space:

```python
out = model(batch)
out['predicted'].shape   # [B, n_tgt, D]  — predictor output (context side)
out['target'].shape      # [B, n_tgt, D]  — EMA target-encoder output, detached
```

Three things happen inside:

1. The **context encoder** encodes only `image_t[context_indices]`, plus the
   proprio and action tokens, and the **predictor** predicts the embeddings at
   `target_indices`.
2. The **target encoder** (under `torch.no_grad`) encodes the *full*
   `image_tp1` and we slice out the same `target_indices`.
3. The targets are **layer-normed** so the loss is not dominated by raw
   magnitude. The loss (chapter 3) regresses `predicted` onto `target.detach()`.

### The EMA target encoder

The target encoder is a `deepcopy` of the context encoder with
`requires_grad=False`. It is never touched by backprop — only by an exponential
moving average of the context encoder:

```python
# target = momentum * target + (1 - momentum) * context
model.update_target_encoder(momentum)   # called once per step in the trainer
```

The momentum is scheduled from `0.996 -> 1.0`, so targets start responsive and
become increasingly stable. This is the mechanism that gives the predictor a
*stable* regression target: if the target encoder moved every step (or equalled
the context encoder), training would chase a moving objective and the easiest
"solution" would be for both encoders to output a constant. That degenerate
solution is **representation collapse**, the subject of chapter 3.

## Conceptual contrast with the swm baselines

NanoJEPA is one of several world models in `stable_worldmodel`, and the way each
one *avoids collapse* is the most instructive way to tell them apart:

| Model                  | Visual backbone        | How it avoids collapse                          |
| ---------------------- | ---------------------- | ----------------------------------------------- |
| **NanoJEPA**           | ViT trained **from scratch** | EMA target encoder **+ VICReg** (toggleable) |
| **PreJEPA** (DINO-WM)  | **frozen** pretrained DINO   | the backbone is frozen — targets cannot collapse |
| **PLDM / LeWM**        | trained                | **detach** the target (no EMA, no moving copy)  |

Because NanoJEPA trains its ViT from scratch, it has no frozen anchor and no
detached target to lean on — so it must *actively* prevent collapse with the EMA
target plus VICReg. There is no equivalent of `update_target_encoder` anywhere
else in swm: PLDM/LeWM detach their targets, PreJEPA freezes a pretrained
backbone. This is precisely why NanoJEPA is the right model to teach the collapse
experiment on.

## Inference helpers (used later)

Two methods produce embeddings outside of training:

```python
# fused vision + proprio embedding, mean-pooled -> [B, D]  (used by probes)
z = model.encode(images, proprios)

# one-step pooled next-state embedding -> [B, D]  (used by the planning demo)
z_next = model.predict_next(images, proprios, actions)
```

A subtle but important point you will meet again in chapter 5: `encode()`
produces a **fused** embedding space (vision + proprio), used by the diagnostics.
Planning instead lives in the **predictor's layer-normed target patch space**.
They are different spaces on purpose.

Next: [training and the collapse experiment](3_train_and_collapse.md).
