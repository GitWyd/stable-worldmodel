"""NanoJEPA neural building blocks: a from-scratch ViT, modality encoders,
and the predictor.

Everything here is deliberately hand-written and readable. ``stable_worldmodel``
ships production ViTs (see ``wm/pldm/module.py`` and the HuggingFace
``vit_hf`` backbones); we re-implement a tiny one so a learner can read the
whole encoder — including the I-JEPA-style patch-index masking, which the
library backbones do not expose.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def sinusoidal_position_embedding_2d(
    grid_size: int, embed_dim: int
) -> torch.Tensor:
    """Fixed 2D sin/cos positional embeddings. Returns ``[grid_size^2, D]``.

    For a 2D grid we need position info along both axes. We split ``embed_dim``
    into 4 equal parts: ``[sin(h), cos(h), sin(w), cos(w)]``, each of size
    ``embed_dim // 4``. This lets the transformer distinguish any (row, col)
    position. Fixed (not learned) because the grid is small (8x8 = 64 patches)
    and sinusoidal embeddings generalise better.
    """
    assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4 for 2D sincos'
    d = embed_dim // 4
    positions_h = torch.arange(grid_size, dtype=torch.float32)
    positions_w = torch.arange(grid_size, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))

    out_h = positions_h.unsqueeze(1) * omega.unsqueeze(0)  # [grid, d]
    out_w = positions_w.unsqueeze(1) * omega.unsqueeze(0)

    pe = torch.cat(
        [
            out_h.sin().unsqueeze(1).expand(-1, grid_size, -1),
            out_h.cos().unsqueeze(1).expand(-1, grid_size, -1),
            out_w.sin().unsqueeze(0).expand(grid_size, -1, -1),
            out_w.cos().unsqueeze(0).expand(grid_size, -1, -1),
        ],
        dim=-1,
    )
    return pe.reshape(grid_size * grid_size, embed_dim)


class PatchEmbed(nn.Module):
    """Convert an image to patch embeddings using a strided Conv2d."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W] -> [B, num_patches, D]``."""
        x = self.proj(x)  # [B, D, gh, gw]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        return x


class Attention(nn.Module):
    """Multi-head attention — manual implementation (MPS-safe, no SDPA).

    We avoid ``F.scaled_dot_product_attention`` because it has known bugs on
    the MPS backend (Apple Silicon). Instead we do explicit QKV matmuls with
    ``.contiguous()`` on every operand to prevent MPS stride errors.
    """

    def __init__(self, dim: int, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Manual attention — all contiguous for MPS
        attn = (
            torch.matmul(q.contiguous(), k.contiguous().transpose(-2, -1))
            * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn.contiguous(), v.contiguous())

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block: LN -> Attn -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim: int, num_heads: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class VisionEncoder(nn.Module):
    """ViT-Tiny: patch embeddings + fixed positional embeddings + transformer.

    ``patch_indices`` selects a subset of patches *before* adding positional
    embeddings — this is how I-JEPA context masking is implemented: only the
    visible context patches are encoded.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        pos_embed = sinusoidal_position_embedding_2d(self.grid_size, embed_dim)
        self.register_buffer('pos_embed', pos_embed)  # [num_patches, D]

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        patch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode images, optionally keeping only ``patch_indices``.

        Args:
            images: ``[B, 3, H, W]``.
            patch_indices: ``[num_selected]`` long; if given, only these
                patches are encoded.

        Returns:
            ``[B, num_selected_or_all, D]``.
        """
        x = self.patch_embed(images)  # [B, num_patches, D]

        if patch_indices is not None:
            x = x[:, patch_indices]  # [B, num_selected, D]
            pos = self.pos_embed[patch_indices].unsqueeze(0)
        else:
            pos = self.pos_embed.unsqueeze(0)

        x = x + pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ProprioEncoder(nn.Module):
    """MLP encoder for proprioception: ``obs_dim -> embed_dim``."""

    def __init__(
        self, obs_dim: int = 25, embed_dim: int = 192, hidden: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, obs_dim] -> [B, 1, embed_dim]``."""
        return self.net(x).unsqueeze(1)


class ActionEncoder(nn.Module):
    """MLP encoder for actions: ``act_dim -> embed_dim``."""

    def __init__(
        self, act_dim: int = 4, embed_dim: int = 192, hidden: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, act_dim] -> [B, 1, embed_dim]``."""
        return self.net(x).unsqueeze(1)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor(nn.Module):
    """Transformer predictor: context + mask tokens -> predicted targets.

    Input sequence: ``[context_vision_tokens, proprio_token, action_token,
    mask_tokens]``. Output: the last ``num_target`` tokens (predictions for the
    masked positions). The action token is what bridges the temporal gap from
    ``image_t`` to ``image_{t+1}``.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        num_patches: int = 64,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pos_embed = sinusoidal_position_embedding_2d(
            int(math.sqrt(num_patches)), embed_dim
        )
        self.register_buffer('pos_embed', pos_embed)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        proprio_token: torch.Tensor,
        action_token: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict embeddings at ``target_indices``.

        Args:
            context_tokens: ``[B, num_context, D]`` from the context encoder.
            proprio_token:  ``[B, 1, D]``.
            action_token:   ``[B, 1, D]``.
            target_indices: ``[num_target]`` patch positions to predict.

        Returns:
            ``[B, num_target, D]`` predicted target embeddings.
        """
        B = context_tokens.shape[0]
        num_target = target_indices.shape[0]

        mask_tokens = self.mask_token.expand(B, num_target, -1)
        mask_pos = self.pos_embed[target_indices].unsqueeze(0)
        mask_tokens = mask_tokens + mask_pos

        x = torch.cat(
            [context_tokens, proprio_token, action_token, mask_tokens], dim=1
        )
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        predicted = x[:, -num_target:]  # only the mask-token predictions
        return self.proj(predicted)


__all__ = [
    'sinusoidal_position_embedding_2d',
    'PatchEmbed',
    'Attention',
    'MLP',
    'TransformerBlock',
    'VisionEncoder',
    'ProprioEncoder',
    'ActionEncoder',
    'Predictor',
]
