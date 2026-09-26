"""Tensor logic for frame chaining, free of any ComfyUI imports so it can be unit tested.

ComfyUI passes images as float tensors shaped ``[batch, height, width, channels]``
with values in ``0..1``. A generated video clip is one such batch: one entry per frame.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

RESIZE_ERROR = "error"
RESIZE_TO_FIRST = "resize to first clip"
RESIZE_MODES = (RESIZE_ERROR, RESIZE_TO_FIRST)


def as_image_batch(images: torch.Tensor, name: str = "images") -> torch.Tensor:
    """Return ``images`` as a ``[B, H, W, C]`` batch. A single ``[H, W, C]`` frame is promoted."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(images).__name__}")
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4:
        raise ValueError(f"{name} must be shaped [batch, height, width, channels], got {tuple(images.shape)}")
    if images.shape[0] < 1:
        raise ValueError(f"{name} is empty: the batch has no frames")
    return images


def last_frame(images: torch.Tensor, offset_from_end: int = 0) -> torch.Tensor:
    """Return one frame as a ``[1, H, W, C]`` batch.

    ``offset_from_end=0`` is the final frame, ``1`` the one before it, and so on.
    Offsets past the start clamp to the first frame, so short clips never fail.
    """
    images = as_image_batch(images)
    if offset_from_end < 0:
        raise ValueError("offset_from_end must be 0 or greater")
    index = max(images.shape[0] - 1 - offset_from_end, 0)
    return images[index : index + 1].clone()


def chain_append(chain: Sequence[torch.Tensor] | None, clip: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return a new chain with ``clip`` added. The input chain is never mutated (ComfyUI caches outputs)."""
    clip = as_image_batch(clip, "clip")
    return (*(chain or ()), clip)


def resample(frames: torch.Tensor, speed: float) -> torch.Tensor:
    """Retime a frame batch by picking frames: ``speed=2`` keeps every other frame, ``0.5`` doubles them.

    Always returns at least one frame.
    """
    if speed <= 0 or not math.isfinite(speed):
        raise ValueError("speed must be a positive number")
    n = frames.shape[0]
    if n == 0 or speed == 1:
        return frames
    count = max(1, round(n / speed))
    index = torch.clamp((torch.arange(count, dtype=torch.float64) * speed).floor().long(), max=n - 1)
    return frames[index]


def speed_ramp(frames: torch.Tensor, fast: float = 2.0, hold: float = 1.0, split: float = 0.5) -> torch.Tensor:
    """Two-speed ramp: the first ``split`` fraction of the clip plays at ``fast``, the rest at ``hold``.

    With ``fast=2, hold=1`` the camera "travel" is punchy and the "hold" at the end
    plays at normal speed so the viewer can take in the detail.
    """
    if not 0 <= split <= 1:
        raise ValueError("split must be between 0 and 1")
    frames = as_image_batch(frames, "frames")
    cut = round(frames.shape[0] * split)
    parts = []
    if cut > 0:
        parts.append(resample(frames[:cut], fast))
    if cut < frames.shape[0]:
        parts.append(resample(frames[cut:], hold))
    return torch.cat(parts, dim=0)


def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # [B, H, W, C] -> [B, C, H, W] for interpolate, then back.
    out = F.interpolate(frames.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
    return out.movedim(1, -1).clamp(0, 1)


def stitch(
    clips: Iterable[torch.Tensor],
    drop_seam_frames: bool = True,
    fast: float = 1.0,
    hold: float = 1.0,
    split: float = 0.5,
    resize: str = RESIZE_ERROR,
) -> torch.Tensor:
    """Concatenate clips head-to-tail into one frame batch.

    - ``drop_seam_frames``: in a frame chain, clip N+1 starts on the last frame of clip N,
      so that frame appears twice and the playback stutters. Dropping the first frame of
      every clip after the first removes the duplicate.
    - ``fast`` / ``hold`` / ``split``: optional speed ramp applied to each clip (see
      :func:`speed_ramp`). ``fast=1, hold=1`` (the default) leaves timing untouched.
    - ``resize``: clips of different sizes either raise (``"error"``) or are resized to
      the first clip's size (``"resize to first clip"``).
    """
    if resize not in RESIZE_MODES:
        raise ValueError(f"resize must be one of {RESIZE_MODES}")
    clips = [as_image_batch(c, f"clip {i + 1}") for i, c in enumerate(clips)]
    if not clips:
        raise ValueError("nothing to stitch: connect a frame chain or at least one image batch")

    _, height, width, channels = clips[0].shape
    out = []
    for i, clip in enumerate(clips):
        if clip.shape[-1] != channels:
            raise ValueError(f"clip {i + 1} has {clip.shape[-1]} channels, clip 1 has {channels}")
        if clip.shape[1:3] != (height, width):
            if resize == RESIZE_ERROR:
                raise ValueError(
                    f"clip {i + 1} is {clip.shape[2]}x{clip.shape[1]}, clip 1 is {width}x{height}. "
                    f"Set resize to '{RESIZE_TO_FIRST}' or generate every clip at the same size."
                )
            clip = _resize(clip, height, width)
        if drop_seam_frames and i > 0 and clip.shape[0] > 1:
            clip = clip[1:]
        if fast != 1 or hold != 1:
            clip = speed_ramp(clip, fast=fast, hold=hold, split=split)
        out.append(clip.to(device=clips[0].device, dtype=clips[0].dtype))
    return torch.cat(out, dim=0)
