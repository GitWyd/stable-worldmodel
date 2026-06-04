---
title: 3. Training & The Collapse Experiment
summary: The JEPA loss, VICReg, and toggling collapse prevention on and off
sidebar_title: 3. Train & Collapse
---

# NanoJEPA — 3. Training & The Collapse Experiment

This is the centrepiece of NanoJEPA. We train the model from
[chapter 2](2_model.md), then run the same training twice — with and without
collapse prevention — and watch the representation either stay healthy or fall
apart. The trainer is plain PyTorch in `scripts/train/nanojepa.py`; you can read
the whole loop.

## The loss

The objective has three terms (`wm/nanojepa/losses.py`):

```python
from stable_worldmodel.wm.nanojepa import (
    jepa_loss, variance_loss, covariance_loss,
)

loss = jepa_loss(predicted, target) \
     + variance_weight   * variance_loss(predicted) \   # TOGGLEABLE
     + covariance_weight * covariance_loss(predicted)    # TOGGLEABLE
```

**`jepa_loss`** is the core prediction loss: `F.smooth_l1_loss(predicted,
target.detach())`. The `.detach()` is critical — gradients must not flow into the
target encoder (it is updated only by EMA). Without it, both encoders could
trivially converge to the same constant and drive the loss to zero.

**`variance_loss`** (the VICReg variance term) pushes each embedding dimension's
standard deviation toward `>= 1.0`: `relu(1 - std).mean()`. A collapsed
representation has near-zero std, so this term punishes collapse directly.

**`covariance_loss`** (the VICReg covariance term) penalises the squared
off-diagonal covariance, decorrelating the dimensions so they each carry
independent information rather than redundantly encoding one feature.

!!! info "What the library ships"
    These three functions are kept hand-written because reading them *is* the
    lesson. The framework also ships a production twin,
    `stable_worldmodel.wm.loss.VCReg`, which packages the same variance +
    covariance idea (returning `std_loss` / `cov_loss` and their temporal
    variants), plus `stable_worldmodel.wm.loss.SIGReg`, a sketched
    isotropic-Gaussian regulariser
    ([arXiv:2511.08544](https://arxiv.org/abs/2511.08544)). When you graduate
    from the tutorial, reach for those.

## The training loop

Everything that teaches JEPA is in `scripts/train/nanojepa.py`. The per-step
core, stripped to essentials:

```python
out = model(batch)
pred, target = out['predicted'], out['target']

loss_j = jepa_loss(pred, target)
loss_v = variance_loss(pred) if use_vicreg else torch.zeros((), device=device)
loss_c = covariance_loss(pred) if use_vicreg else torch.zeros((), device=device)
loss = loss_j + var_w * loss_v + cov_w * loss_c

optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(trainable_params, grad_clip)
optimizer.step()
model.update_target_encoder(ema)        # EMA, after the optimizer step
```

Around that core the loop schedules three things over training (all from the
`training:` config block): a warmup-then-cosine **learning rate**
(`lr -> lr_min`), a linear **EMA momentum** ramp (`0.996 -> 1.0`), and a linear
**weight decay** ramp (`weight_decay_start -> weight_decay_end`). After every
step it logs `compute_collapse_metrics(pred)` so you can watch the health of the
representation live.

Note the `--no-vicreg` switch simply zeros `var_w` and `cov_w` — the prediction
loss is untouched. That is the whole experiment.

## Run it: VICReg ON

```bash
# ~15 min on a laptop GPU/MPS; collects the dataset first if it is missing
python scripts/train/nanojepa.py \
    --config scripts/train/config/nanojepa_flat/quick.yaml
```

Checkpoints land in `checkpoints/nanojepa/epoch_*.pt`. The per-step log shows the
embedding std staying high and the effective rank staying large:

```
  step   250 | loss 0.31 (jepa 0.28, var 0.02, cov 0.00) | std 0.74 | rank 96.3 | lr 9.9e-04
```

## Run it: VICReg OFF (the collapse)

```bash
python scripts/train/nanojepa.py \
    --config scripts/train/config/nanojepa_flat/quick.yaml --no-vicreg
```

This run is written to a *separate* directory, `checkpoints/nanojepa_no_vicreg/`,
so the two checkpoints do not clobber each other. With the variance/covariance
pressure removed, the embeddings drift toward a constant vector — the std and
rank crater even as the jepa loss happily falls:

```
  step   250 | loss 0.04 (jepa 0.04, var 0.00, cov 0.00) | std 0.06 | rank 3.1 | lr 9.9e-04
```

The trap: a low prediction loss looks like success, but the model achieved it by
predicting (nearly) the same vector everywhere. The representation is useless.

## The ON/OFF table

Once both runs finish, the difference is stark. Chapter 4 computes this table
automatically with `scripts/eval/nanojepa.py --compare`, but conceptually:

| Metric                      | VICReg ON | VICReg OFF |
| --------------------------- | --------- | ---------- |
| linear-probe R²             | `> 0.80`  | `~ 0.00`   |
| embedding std (mean per-dim)| `> 0.50`  | `< 0.10`   |
| effective rank              | `~ 100`   | `< 5`      |

- **R²** (chapter 4's linear probe): with collapse prevention the embeddings
  linearly decode the end-effector position; collapsed embeddings decode nothing.
- **Embedding std**: healthy representations spread mass across dimensions;
  collapsed ones squeeze toward a point.
- **Effective rank** (`exp(entropy(singular values))`): how many dimensions are
  actually used — close to `embed_dim` when healthy, a handful when collapsed.

## Checkpoints

The trainer saves self-contained checkpoints via `model.save_checkpoint(path,
epoch=..., optimizer=..., history=..., vicreg=...)`. Each one stores its own
`model_cfg`, so reloading needs no config file:

```python
from stable_worldmodel.wm.nanojepa import NanoJEPA

model, ckpt = NanoJEPA.from_checkpoint(
    'checkpoints/nanojepa/epoch_10.pt', device='cpu',
)
ckpt['vicreg']    # True / False — which arm of the experiment this was
ckpt['history']   # per-step loss / std / rank curves for plotting
```

Next: [evaluate the two checkpoints](4_evaluate.md) and produce the table for
real.
