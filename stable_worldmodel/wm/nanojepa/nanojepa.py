"""The NanoJEPA multimodal world model.

Training path (unchanged from the standalone NanoJEPA):
    context encoder sees a masked ``image_t`` (+ proprio + action), the EMA
    target encoder sees the full ``image_{t+1}``, the predictor maps context ->
    target embeddings, and the loss is computed in embedding space.

Planning path (new — the ``Costable`` bridge to swm solvers):
    ``get_cost`` encodes the goal image, autoregressively rolls the predictor
    forward over a candidate action sequence in latent space, and scores the
    squared distance to the goal embedding. This makes NanoJEPA drivable by
    ``CEMSolver`` / ``WorldModelPolicy`` / ``World.evaluate`` exactly like the
    ``PLDM`` / ``LeWM`` baselines.

Conceptual contrast worth teaching: ``wm/prejepa`` (DINO-WM) avoids collapse by
*freezing* a pretrained backbone; NanoJEPA trains its ViT *from scratch* and so
must actively prevent collapse with an EMA target encoder + VICReg.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    ActionEncoder,
    ProprioEncoder,
    VisionEncoder,
    Predictor,
)


class NanoJEPA(nn.Module):
    """Multimodal JEPA world model (from-scratch ViT + EMA target + VICReg)."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 3,
        predictor_depth: int = 4,
        predictor_heads: int = 3,
        obs_dim: int = 25,
        act_dim: int = 4,
    ):
        super().__init__()
        # Keep constructor args so checkpoints are self-describing (C8).
        self._model_cfg = dict(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
        num_patches = (image_size // patch_size) ** 2

        # Context pathway (trained).
        self.context_encoder = VisionEncoder(
            image_size, patch_size, embed_dim, encoder_depth, encoder_heads
        )
        self.proprio_encoder = ProprioEncoder(obs_dim, embed_dim)
        self.action_encoder = ActionEncoder(act_dim, embed_dim)
        self.predictor = Predictor(
            embed_dim, predictor_depth, predictor_heads, num_patches
        )

        # Target pathway (EMA, not trained).
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        """ViT-style init: trunc_normal_(std=0.02) on Linear/Conv weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """EMA update: ``target = m * target + (1 - m) * context``.

        The target encoder provides stable regression targets. Without EMA the
        targets would change every step and training would be unstable.
        Momentum rises 0.996 -> 1.0, so the target starts responsive and
        becomes increasingly stable. (No equivalent exists in swm: PLDM/LeWM
        detach their targets, PreJEPA freezes a pretrained backbone.)
        """
        for q, k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            k.data.mul_(momentum).add_(q.data, alpha=1.0 - momentum)

    # --------------------------------------------------------------- training
    def forward(self, batch: dict) -> dict:
        """Training forward pass.

        Args:
            batch: dict with ``image_t, proprio_t, action_t, image_tp1,
                context_indices, target_indices`` (see CONTRACTS.md C2).

        Returns:
            ``{"predicted": [B, n_tgt, D], "target": [B, n_tgt, D]}``.
        """
        context_tokens = self.context_encoder(
            batch['image_t'], batch['context_indices']
        )
        proprio_token = self.proprio_encoder(batch['proprio_t'])
        action_token = self.action_encoder(batch['action_t'])

        predicted = self.predictor(
            context_tokens,
            proprio_token,
            action_token,
            batch['target_indices'],
        )

        with torch.no_grad():
            target_full = self.target_encoder(
                batch['image_tp1'], patch_indices=None
            )
            target = target_full[:, batch['target_indices']]
            # Layer-normalise targets (prevents loss dominated by magnitude).
            target = F.layer_norm(target, (target.shape[-1],))

        return {'predicted': predicted, 'target': target}

    # -------------------------------------------------------------- inference
    def encode(
        self, images: torch.Tensor, proprios: torch.Tensor
    ) -> torch.Tensor:
        """Fused inference embedding (vision + proprio), mean-pooled ``[B, D]``.

        Used by the linear probe / cross-modal diagnostics. Distinct from the
        planning-cost space (see ``get_cost``), which lives in the predictor's
        target patch space.
        """
        vision_tokens = self.context_encoder(images, patch_indices=None)
        proprio_token = self.proprio_encoder(proprios)
        all_tokens = torch.cat([vision_tokens, proprio_token], dim=1)
        return all_tokens.mean(dim=1)

    def predict_next(
        self,
        images: torch.Tensor,
        proprios: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """One-step pooled next-state embedding ``[B, D]`` (full image input).

        The simple API used by the hand-written planning demo in the tutorial;
        ``get_cost`` uses the token-level :meth:`_latent_rollout` instead.
        """
        num_patches = self.context_encoder.num_patches
        all_indices = torch.arange(num_patches, device=images.device)
        context_tokens = self.context_encoder(images, patch_indices=None)
        proprio_token = self.proprio_encoder(proprios)
        action_token = self.action_encoder(actions)
        predicted_all = self.predictor(
            context_tokens, proprio_token, action_token, all_indices
        )
        return predicted_all.mean(dim=1)

    # ---------------------------------------------------------------- planning
    def _latent_rollout(
        self,
        context_tokens: torch.Tensor,
        proprio_token: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive latent rollout over an action sequence.

        The predictor emits a full ``[B, N, D]`` next-state token map; we feed
        it back as the next context and condition on the next action. The final
        state is layer-normed + mean-pooled into the predictor's target space
        (matching the training target), so it is comparable to an encoded goal.

        Args:
            context_tokens: ``[B, N, D]`` initial state tokens.
            proprio_token:  ``[B, 1, D]`` (held fixed across the rollout).
            actions:        ``[B, H, A]`` action sequence.

        Returns:
            ``[B, D]`` rolled-out state embedding.
        """
        n = context_tokens.shape[1]
        all_idx = torch.arange(n, device=context_tokens.device)
        tokens = context_tokens
        for h in range(actions.shape[1]):
            action_token = self.action_encoder(actions[:, h])
            tokens = self.predictor(
                tokens, proprio_token, action_token, all_idx
            )
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        return tokens.mean(dim=1)

    def _encode_goal_z(self, goal_images: torch.Tensor) -> torch.Tensor:
        """Encode goal images into the target patch space ``[B, D]``."""
        with torch.no_grad():
            g = self.target_encoder(goal_images, patch_indices=None)
            g = F.layer_norm(g, (g.shape[-1],))
        return g.mean(dim=1)

    def rollout(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Roll the model forward for every action candidate.

        ``info_dict`` tensors are ``[B, S, T, ...]`` (B envs, S samples, T
        history); the current state is identical across S (the solver
        broadcasts it), so we encode it once per env and expand. Adds
        ``predicted_z`` ``[B, S, D]``.
        """
        b, s, h = action_candidates.shape[:3]
        pixels = info_dict['pixels'][:, 0, -1].float()  # (B, 3, H, W)
        proprio = info_dict['proprio'][:, 0, -1].float()  # (B, P)

        ctx = self.context_encoder(pixels, patch_indices=None)  # (B, N, D)
        proprio_token = self.proprio_encoder(proprio)  # (B, 1, D)
        n, d = ctx.shape[1], ctx.shape[2]

        ctx = ctx.unsqueeze(1).expand(b, s, n, d).reshape(b * s, n, d)
        proprio_token = (
            proprio_token.unsqueeze(1).expand(b, s, 1, d).reshape(b * s, 1, d)
        )
        actions = action_candidates.reshape(b * s, h, -1)

        z = self._latent_rollout(ctx, proprio_token, actions)  # (B*S, D)
        info_dict['predicted_z'] = z.reshape(b, s, d)
        return info_dict

    def criterion(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Squared latent distance between rolled-out and goal embeddings.

        Returns ``[B, S]`` (override for custom planning costs).
        """
        pred = info_dict['predicted_z']  # (B, S, D)
        goal = info_dict['goal_z']  # (B, S, D)
        return (pred - goal).pow(2).sum(dim=-1)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Planning cost for action candidates (the ``Costable`` entry point).

        Args:
            info_dict: planning info (see CONTRACTS.md C3); must contain a
                ``goal`` image.
            action_candidates: ``[B, S, H, A]``.

        Returns:
            ``[B, S]`` cost (lower is better).
        """
        assert 'goal' in info_dict, 'goal not in info_dict'
        b, s = action_candidates.shape[:2]
        goal = info_dict['goal'][:, 0, -1].float()  # (B, 3, H, W)
        z_goal = self._encode_goal_z(goal)  # (B, D)
        info_dict['goal_z'] = z_goal.unsqueeze(1).expand(b, s, -1)
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict, action_candidates)

    # ----------------------------------------------------------- (de)serialise
    @classmethod
    def from_config(cls, cfg: dict) -> 'NanoJEPA':
        """Build a model from a flat config dict (CONTRACTS.md C6)."""
        m = cfg['model']
        env = cfg.get('env', {})
        return cls(
            image_size=env.get('image_size', 64),
            patch_size=m['patch_size'],
            embed_dim=m['embed_dim'],
            encoder_depth=m['encoder_depth'],
            encoder_heads=m['encoder_heads'],
            predictor_depth=m['predictor_depth'],
            predictor_heads=m['predictor_heads'],
            obs_dim=m.get('obs_dim', 25),
            act_dim=m.get('act_dim', 4),
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str | torch.device = 'cpu'
    ) -> tuple['NanoJEPA', dict]:
        """Load ``(model, ckpt)`` from a self-contained checkpoint (C8)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt['model_cfg']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt

    def save_checkpoint(
        self,
        path: str,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        history: dict | None = None,
        vicreg: bool = True,
    ) -> None:
        """Save a self-contained checkpoint (CONTRACTS.md C8)."""
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': (
                    optimizer.state_dict() if optimizer is not None else None
                ),
                'history': history,
                'vicreg': vicreg,
                'model_cfg': self._model_cfg,
            },
            path,
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters per top-level child plus total/trainable."""
    counts = {}
    for name, child in model.named_children():
        counts[name] = sum(p.numel() for p in child.parameters())
    counts['total'] = sum(p.numel() for p in model.parameters())
    counts['trainable'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return counts


__all__ = ['NanoJEPA', 'count_parameters']
