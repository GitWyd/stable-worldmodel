"""I-JEPA multiblock mask collator + the swm-dataset adapter.

``TransitionView`` is the *only* bridge between a ``stable_worldmodel`` dataset
and NanoJEPA's (otherwise unchanged) model: it turns a 2-step window into the
``(image_t, proprio_t, action_t, image_tp1, proprio_tp1)`` tuple the model
expects. ``MaskCollator`` then samples I-JEPA context/target patch masks per
batch and assembles the training batch dict (see ``CONTRACTS.md`` C2).
"""

import math
import random

import torch
from torch.utils.data import DataLoader, Dataset


class TransitionView(Dataset):
    """Adapt a swm 2-step ``Dataset`` to NanoJEPA transitions.

    A swm sample ``ds[i]`` is a dict of 2-step windows with ``pixels``
    (``uint8 [2, 3, H, W]``, already CHW), ``proprio`` (``[2, P]``) and
    ``action`` (``[2, A]``). Index 0 is ``t``, index 1 is ``t+1``; ``action[0]``
    is the action taken at ``t`` (World.collect aligns actions to transitions).
    """

    def __init__(self, dataset: Dataset):
        assert getattr(dataset, 'num_steps', None) == 2, (
            'TransitionView expects a swm dataset loaded with num_steps=2 '
            f'(got num_steps={getattr(dataset, "num_steps", None)}).'
        )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _img(frame: torch.Tensor) -> torch.Tensor:
        """``uint8 [3, H, W] -> float32 [3, H, W]`` in [0, 1] (already CHW)."""
        return frame.float().div(255.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        w = self.dataset[idx]
        pixels = w['pixels']  # [2, 3, H, W] uint8
        proprio = w['proprio'].float()  # [2, P]
        action = w['action'].float()  # [2, A]
        return (
            self._img(pixels[0]),
            proprio[0],
            action[0],
            self._img(pixels[1]),
            proprio[1],
        )

    def single_step(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image_t [3,H,W] in [0,1], proprio_t [P])`` — for probes."""
        w = self.dataset[idx]
        return self._img(w['pixels'][0]), w['proprio'][0].float()


class MaskCollator:
    """Collate function generating I-JEPA multiblock masks per batch.

    Following I-JEPA (Assran et al. 2023): sample rectangular target blocks,
    use the complement as context. All samples in a batch share the same mask
    positions (but different image content) so context/target tensors have a
    uniform size across the batch.

    Args:
        patch_size: Patch size in pixels (8 -> 8x8 grid for 64x64 images).
        image_size: Image resolution.
        num_targets: Number of target blocks to sample.
        target_scale: ``(min, max)`` fraction of total patches per target block.
        target_aspect_ratio: ``(min, max)`` aspect ratio of target blocks.
        min_context_patches: Minimum context patches to retain.
    """

    def __init__(
        self,
        patch_size: int = 8,
        image_size: int = 64,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_context_patches: int = 16,
    ):
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.min_context_patches = min_context_patches

    def _sample_block(self) -> set[int]:
        """Sample one rectangular block of patch indices (row-major).

        Following I-JEPA Section 3.2: each block is a contiguous rectangle in
        the patch grid, sized as a fraction of total patches with a random
        aspect ratio. This spatial structure forces the model to learn local
        coherence rather than memorising individual patches.
        """
        g = self.grid_size
        n = self.num_patches

        scale = random.uniform(*self.target_scale)
        num_target = max(1, int(scale * n))

        aspect = random.uniform(*self.target_aspect_ratio)
        h = max(1, min(g, int(round(math.sqrt(num_target * aspect)))))
        w = max(1, min(g, int(round(num_target / h))))

        top = random.randint(0, g - h)
        left = random.randint(0, g - w)

        indices = set()
        for r in range(top, top + h):
            for c in range(left, left + w):
                indices.add(r * g + c)
        return indices

    def _generate_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ``(context_indices, target_indices)`` for this batch."""
        all_target: set[int] = set()
        for _ in range(self.num_targets):
            block = self._sample_block()
            if (
                len(all_target | block)
                > self.num_patches - self.min_context_patches
            ):
                break
            all_target |= block

        all_indices = set(range(self.num_patches))
        context_set = all_indices - all_target

        # Safety: if blocks overlapped badly, reclaim patches for context.
        if len(context_set) < self.min_context_patches:
            excess = self.min_context_patches - len(context_set)
            target_list = sorted(all_target)
            random.shuffle(target_list)
            for i in range(excess):
                idx = target_list[i]
                all_target.discard(idx)
                context_set.add(idx)

        context_indices = torch.tensor(sorted(context_set), dtype=torch.long)
        target_indices = torch.tensor(sorted(all_target), dtype=torch.long)
        return context_indices, target_indices

    def __call__(self, batch: list) -> dict:
        """Collate a batch of ``TransitionView`` tuples + add mask indices."""
        imgs_t, pros_t, acts_t, imgs_tp1, pros_tp1 = zip(*batch)
        context_indices, target_indices = self._generate_masks()
        return {
            'image_t': torch.stack(imgs_t),
            'proprio_t': torch.stack(pros_t),
            'action_t': torch.stack(acts_t),
            'image_tp1': torch.stack(imgs_tp1),
            'proprio_tp1': torch.stack(pros_tp1),
            'context_indices': context_indices,
            'target_indices': target_indices,
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    masking_cfg: dict | None = None,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """Wrap a swm 2-step dataset in a NanoJEPA training ``DataLoader``.

    Args:
        dataset: a swm ``Dataset`` loaded with ``num_steps=2``.
        batch_size: batch size.
        masking_cfg: kwargs for :class:`MaskCollator` (patch_size, image_size,
            num_targets, target_scale, target_aspect_ratio,
            min_context_patches).
    """
    view = TransitionView(dataset)
    collate = MaskCollator(**(masking_cfg or {}))
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )


__all__ = ['TransitionView', 'MaskCollator', 'make_loader']
