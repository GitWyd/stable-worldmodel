"""NanoJEPA — an educational, from-scratch multimodal JEPA world model.

A teaching companion to the ``stable_worldmodel`` baselines: read it top to
bottom to learn how a JEPA predicts in embedding space, why it would collapse,
and how an EMA target encoder + VICReg prevent that. Unlike ``wm/prejepa``
(frozen pretrained backbone) it trains its ViT from scratch, and unlike
``wm/pldm`` / ``wm/lewm`` it uses an EMA target — the single mechanism the
collapse experiment hinges on.

See ``CONTRACTS.md`` for the frozen interface contracts and
``docs/tutorial/nanojepa/`` for the curriculum.
"""

from .losses import covariance_loss, jepa_loss, variance_loss
from .masking import MaskCollator, TransitionView, make_loader
from .metrics import (
    compute_collapse_metrics,
    cross_modal_eval,
    linear_probe,
)
from .module import (
    ActionEncoder,
    Predictor,
    ProprioEncoder,
    VisionEncoder,
)
from .nanojepa import NanoJEPA, count_parameters
from .planning import build_cem_policy, latent_shooting


__all__ = [
    'NanoJEPA',
    'count_parameters',
    'VisionEncoder',
    'ProprioEncoder',
    'ActionEncoder',
    'Predictor',
    'MaskCollator',
    'TransitionView',
    'make_loader',
    'jepa_loss',
    'variance_loss',
    'covariance_loss',
    'compute_collapse_metrics',
    'linear_probe',
    'cross_modal_eval',
    'build_cem_policy',
    'latent_shooting',
]
