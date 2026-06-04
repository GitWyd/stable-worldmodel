"""JEPA prediction loss + toggleable VICReg collapse-prevention terms.

These three functions are the heart of the collapse experiment. They are kept
hand-written (rather than reusing :class:`stable_worldmodel.wm.loss.VCReg`)
because reading them *is* the lesson:

    loss = jepa_loss(predicted, target)
         + variance_weight   * variance_loss(predicted)    # TOGGLEABLE
         + covariance_weight * covariance_loss(predicted)   # TOGGLEABLE

Turning the VICReg terms off (``--no-vicreg``) makes the representation collapse
to a constant vector. The library twin ``swm.wm.loss.VCReg`` packages the same
variance + covariance idea (and ``SIGReg`` an isotropic-Gaussian variant) for
production use — the tutorial points there as "what the framework ships."
"""

import torch
import torch.nn.functional as F


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Core JEPA loss: smooth L1 between predicted and target embeddings.

    ``target.detach()`` is critical — gradients must NOT flow through the target
    encoder. The target encoder is updated only via EMA, not backprop. Without
    detach, both encoders would converge to a trivial constant solution.
    """
    return F.smooth_l1_loss(predicted, target.detach())


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg variance term: push each dimension's std toward >= 1.0.

    Without this, all embeddings can collapse to a single point (the trivial
    minimiser of the prediction error). Requiring every dimension's standard
    deviation to be at least 1.0 keeps the embeddings spread across dimensions.

    Args:
        z: ``[..., D]`` predicted embeddings; std is computed over the batch.

    Returns:
        Scalar loss; 0 when all dims have std >= 1.0, positive otherwise.
    """
    z_flat = z.reshape(-1, z.shape[-1])
    std = z_flat.std(dim=0)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: decorrelate embedding dimensions.

    Even with high variance, dimensions could still be redundant (all encoding
    the same feature). Penalising the off-diagonal covariance pushes dimensions
    to carry independent information, maximising the embedding's content.

    Args:
        z: ``[..., D]`` predicted embeddings.

    Returns:
        Scalar loss; 0 when all dimensions are perfectly uncorrelated.
    """
    z_flat = z.reshape(-1, z.shape[-1])
    z_centered = z_flat - z_flat.mean(dim=0)
    n = z_flat.shape[0]
    cov = (z_centered.T @ z_centered) / (n - 1)
    d = cov.shape[0]
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d


__all__ = ['jepa_loss', 'variance_loss', 'covariance_loss']
