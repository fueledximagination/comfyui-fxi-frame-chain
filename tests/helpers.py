import torch


def make_clip(values, height=2, width=3, channels=3):
    """A tiny [B, H, W, C] clip whose frame i is filled with values[i] / 100.

    Frame identity survives slicing, reordering and concatenation, so tests can
    assert on exactly which frames came out.
    """
    frames = [torch.full((height, width, channels), v / 100) for v in values]
    return torch.stack(frames)


def frame_ids(batch):
    """The id (fill value * 100) of every frame in a batch."""
    return [round(float(f.mean()) * 100) for f in batch]
