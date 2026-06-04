"""Representation-quality diagnostics: collapse metrics, a linear probe, and a
cross-modal fusion test.

``stable_worldmodel`` ships only a probe *registry* (``wm/probes.py``:
``attach_probe`` / ``get_probe`` / ``load_probe``), not a probe *trainer* or a
collapse metric — so these stay hand-written. They are what make the collapse
experiment legible: with VICReg on, ``r2`` is high and ``effective_rank`` is
large; with VICReg off, both crater.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .masking import TransitionView


@torch.no_grad()
def compute_collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    """Quantify representation collapse from a batch of embeddings.

    Args:
        z: ``[..., D]`` embeddings.

    Returns:
        ``embedding_std_mean`` (mean per-dim std; healthy > 0.5, collapsing
        < 0.1) and ``effective_rank`` (``exp(entropy(singular values))``;
        healthy ~ ``D``, collapsed < 5).
    """
    z_flat = z.reshape(-1, z.shape[-1]).float()
    stds = z_flat.std(dim=0)
    std_mean = stds.mean().item()

    _, s, _ = torch.linalg.svd(
        z_flat - z_flat.mean(dim=0), full_matrices=False
    )
    s_norm = s / s.sum()
    s_norm = s_norm[s_norm > 1e-8]
    entropy = -(s_norm * s_norm.log()).sum()
    eff_rank = entropy.exp().item()
    return {'embedding_std_mean': std_mean, 'effective_rank': eff_rank}


def _as_view(dataset) -> TransitionView:
    return (
        dataset if hasattr(dataset, 'single_step') else TransitionView(dataset)
    )


def _gather_frames(
    dataset, num_samples: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise ``(images [N,3,H,W], proprios [N,P])`` from a dataset."""
    view = _as_view(dataset)
    n = len(view) if num_samples is None else min(num_samples, len(view))
    images, proprios = [], []
    for i in range(n):
        img, pro = view.single_step(i)
        images.append(img)
        proprios.append(pro)
    return torch.stack(images), torch.stack(proprios)


@torch.no_grad()
def _encode_frames(
    model,
    images: torch.Tensor,
    proprios: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode frames in batches; returns ``[N, D]`` on CPU."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        img_b = images[i : i + batch_size].to(device)
        pro_b = proprios[i : i + batch_size].to(device)
        out.append(model.encode(img_b, pro_b).cpu())
        if device.type == 'mps':
            torch.mps.empty_cache()
    return torch.cat(out)


def linear_probe(
    model,
    dataset,
    device: torch.device,
    target_slice: slice = slice(0, 3),
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_samples: int | None = 4000,
) -> dict[str, float]:
    """Fit a linear layer mapping frozen embeddings -> a proprio target.

    The standard self-supervised probe: if a linear model can decode useful
    state (here the end-effector xyz = ``proprio[:3]``) from the embeddings,
    the encoder learned meaningful features. Reports R^2 on an 80/20 split.

    Expected: VICReg on -> R^2 > 0.8; collapsed -> R^2 ~ 0.
    """
    images, proprios = _gather_frames(dataset, num_samples)
    targets = proprios[:, target_slice]

    n = len(images)
    split = int(0.8 * n)
    train_emb = _encode_frames(
        model, images[:split], proprios[:split], device, batch_size
    )
    val_emb = _encode_frames(
        model, images[split:], proprios[split:], device, batch_size
    )
    train_tgt, val_tgt = targets[:split], targets[split:]

    probe = nn.Linear(train_emb.shape[1], train_tgt.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    dl = DataLoader(
        TensorDataset(train_emb.to(device), train_tgt.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    probe.train()
    for _ in range(epochs):
        for emb_b, tgt_b in dl:
            loss = F.mse_loss(probe(emb_b), tgt_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(val_emb.to(device))
        tgt = val_tgt.to(device)
        mse = F.mse_loss(pred, tgt).item()
        ss_res = ((pred - tgt) ** 2).sum().item()
        ss_tot = ((tgt - tgt.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {'mse': mse, 'r2': r2}


def cross_modal_eval(
    model,
    dataset,
    device: torch.device,
    num_samples: int = 1000,
    batch_size: int = 256,
) -> dict[str, float]:
    """Measure how much each modality contributes to the fused embedding.

    Zero out one modality and compare (cosine) the single-modal embedding to
    the full-modal one. High similarity for both means the model fused vision
    and proprioception; low for one means it is being ignored.
    """
    images, proprios = _gather_frames(dataset, num_samples)
    full = _encode_frames(model, images, proprios, device, batch_size)
    vision = _encode_frames(
        model, images, torch.zeros_like(proprios), device, batch_size
    )
    proprio = _encode_frames(
        model, torch.zeros_like(images), proprios, device, batch_size
    )
    cos_vision = F.cosine_similarity(vision, full, dim=1).mean().item()
    cos_proprio = F.cosine_similarity(proprio, full, dim=1).mean().item()
    return {'cos_vision': cos_vision, 'cos_proprio': cos_proprio}


__all__ = [
    'compute_collapse_metrics',
    'linear_probe',
    'cross_modal_eval',
]
