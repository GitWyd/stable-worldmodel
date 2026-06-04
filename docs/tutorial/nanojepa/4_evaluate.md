---
title: 4. Evaluate
summary: Linear probes, cross-modal tests, and collapse metrics on a checkpoint
sidebar_title: 4. Evaluate
---

# NanoJEPA — 4. Evaluate

In [chapter 3](3_train_and_collapse.md) we trained two checkpoints — one with
VICReg, one without. Now we measure what the embeddings actually learned. Three
diagnostics live in `wm/nanojepa/metrics.py`, and `scripts/eval/nanojepa.py`
drives them.

These probes stay hand-written because `stable_worldmodel` ships only a probe
*registry* (`wm/probes.py`: `attach_probe` / `get_probe` / `load_probe`), not a
probe *trainer* or a collapse metric. They are what make the collapse experiment
legible.

## The three diagnostics

All three consume the **fused** `encode()` embedding space (vision + proprio) and
read frames from a swm dataset through `TransitionView.single_step`.

**`linear_probe`** — fit a single linear layer mapping frozen embeddings to a
proprio target (the end-effector xyz, `proprio[:3]`), report R² on an 80/20
split. If a *linear* model can decode useful state, the encoder learned
meaningful features.

```python
import torch
import stable_worldmodel as swm
from stable_worldmodel.wm.nanojepa import (
    NanoJEPA, linear_probe, cross_modal_eval, compute_collapse_metrics,
)

device = torch.device('cpu')
ds = swm.data.load_dataset(
    'nanojepa_fetchpush_quick.lance',
    num_steps=2,
    keys_to_load=['pixels', 'proprio', 'action'],
)
model, _ = NanoJEPA.from_checkpoint('checkpoints/nanojepa/epoch_10.pt', device)

probe = linear_probe(model, ds, device)            # default target_slice=slice(0,3)
probe['r2'], probe['mse']                          # VICReg ON: r2 > 0.8
```

**`cross_modal_eval`** — zero out one modality and compare (cosine) the
single-modal embedding to the full-modal one. High similarity for *both* means
the model genuinely fused vision and proprioception; a low score for one means
that modality is being ignored.

```python
cross = cross_modal_eval(model, ds, device)
cross['cos_vision'], cross['cos_proprio']
```

**`compute_collapse_metrics`** — the quantitative collapse readout from chapter
3, computed over a batch of embeddings:

```python
# encode a slab of frames, then measure
from stable_worldmodel.wm.nanojepa import TransitionView

view = TransitionView(ds)
imgs, pros = zip(*(view.single_step(i) for i in range(1000)))
z = model.encode(torch.stack(imgs).to(device), torch.stack(pros).to(device))
compute_collapse_metrics(z)
# {'embedding_std_mean': float, 'effective_rank': float}
```

## Evaluate one checkpoint

```bash
python scripts/eval/nanojepa.py \
    --checkpoint checkpoints/nanojepa/epoch_10.pt \
    --config scripts/train/config/nanojepa_flat/quick.yaml
```

prints a summary block:

```
=== checkpoints/nanojepa/epoch_10.pt ===
  linear-probe R^2      : 0.8xxx
  cross-modal (vision)  : 0.9xxx
  cross-modal (proprio) : 0.8xxx
  embedding std         : 0.7xxx
  effective rank        : 9x.x
```

## The collapse table side-by-side (`--compare`)

This is where chapter 3's experiment pays off. Point `--compare` at the two
checkpoints (VICReg ON first, OFF second) and the script prints the ON/OFF table:

```bash
python scripts/eval/nanojepa.py \
    --config scripts/train/config/nanojepa_flat/quick.yaml \
    --compare checkpoints/nanojepa/epoch_10.pt \
              checkpoints/nanojepa_no_vicreg/epoch_10.pt
```

```
+--------------------+--------------+--------------+
| Metric             | VICReg ON    | VICReg OFF   |
+--------------------+--------------+--------------+
| R^2                |       0.8xxx |       0.0xxx |
| cosine_vision      |       0.9xxx |       0.9xxx |
| cosine_proprio     |       0.8xxx |       0.9xxx |
| embedding_std      |       0.7xxx |       0.0xxx |
| effective_rank     |      9x.xxxx |       3.xxxx |
+--------------------+--------------+--------------+
```

The headline numbers reproduce the contract:

- **R²** drops from `> 0.8` to `~ 0` — the collapsed encoder cannot decode state.
- **embedding std** drops from `> 0.5` to `< 0.1`.
- **effective rank** drops from `~ 100` to `< 5`.

(Cosine cross-modal similarity is *not* a reliable collapse detector: a constant
vector is trivially cosine-similar to everything, so that row can look fine even
when the model has collapsed. That is exactly why the probe R², std, and rank
matter.)

## Reading the result

A JEPA's loss going down is necessary but not sufficient. The probe answers the
real question — *did the embedding capture the world's state?* With VICReg on, a
linear map recovers the end-effector position; with it off, the model "solved"
prediction by collapsing, and the probe exposes the emptiness behind a deceptively
low loss.

Next: [planning](5_planning.md) — putting the (uncollapsed) world model to work.
