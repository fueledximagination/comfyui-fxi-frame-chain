"""ComfyUI node definitions (V1 node API: INPUT_TYPES / RETURN_TYPES / FUNCTION)."""

from __future__ import annotations

from .core import RESIZE_MODES, chain_append, last_frame, stitch

CATEGORY = "FXI/frame chain"
FRAME_CHAIN = "FXI_FRAME_CHAIN"
MAX_EXTRA_INPUTS = 4


def _video_frames(video):
    """Frames from a core VIDEO input (LoadVideo / CreateVideo) as an IMAGE batch."""
    get_components = getattr(video, "get_components", None)
    if get_components is None:
        raise TypeError("video input does not expose get_components(); connect an IMAGE batch instead")
    return get_components().images


class FXILastFrameExtract:
    DESCRIPTION = (
        "Pull the exact last frame out of a clip (an IMAGE batch or a VIDEO). "
        "Feed it to the start image of the next image-to-video generation to chain clips "
        "into one continuous shot."
    )
    CATEGORY = CATEGORY
    FUNCTION = "extract"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("last_frame", "frame_count")
    OUTPUT_TOOLTIPS = ("The selected frame as a one-image batch.", "Number of frames in the input clip.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offset_from_end": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "0 = the last frame. Raise it to skip a soft or blurred final frame.",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "A clip as an IMAGE batch, e.g. the output of VAE Decode."}),
                "video": ("VIDEO", {"tooltip": "Or a VIDEO, e.g. from Load Video. Used when images is not connected."}),
            },
        }

    def extract(self, offset_from_end, images=None, video=None):
        if images is None:
            if video is None:
                raise ValueError("Last Frame Extract: connect either images or video")
            images = _video_frames(video)
        return (last_frame(images, offset_from_end), int(images.shape[0]))


class FXIFrameChainStep:
    DESCRIPTION = (
        "One link of a frame chain. Collects this clip into the chain and outputs its last frame "
        "as the start image for the next clip. Wire Step -> next generation -> Step -> ... and "
        "finish with Stitch Clips. Stateless, so it also works inside loop nodes."
    )
    CATEGORY = CATEGORY
    FUNCTION = "step"
    RETURN_TYPES = (FRAME_CHAIN, "IMAGE", "INT")
    RETURN_NAMES = ("chain", "next_start_image", "clip_count")
    OUTPUT_TOOLTIPS = (
        "The chain so far, including this clip. Connect to the next Step or to Stitch Clips.",
        "Last frame of this clip. Connect to the start image of the next generation.",
        "Number of clips in the chain.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "The clip just generated, as an IMAGE batch."}),
                "offset_from_end": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "tooltip": "Which frame to hand on. 0 = the last frame."},
                ),
            },
            "optional": {
                "chain": (FRAME_CHAIN, {"tooltip": "The chain from the previous Step. Leave empty on the first clip."}),
            },
        }

    def step(self, clip, offset_from_end, chain=None):
        new_chain = chain_append(chain, clip)
        return (new_chain, last_frame(clip, offset_from_end), len(new_chain))


class FXIStitchClips:
    DESCRIPTION = (
        "Join clips head-to-tail into one IMAGE batch, ready for Create Video or Save Animated WEBP. "
        "Drops the duplicated seam frame between chained clips and can apply a two-speed ramp "
        "(e.g. 2x travel, 1x hold) to every clip."
    )
    CATEGORY = CATEGORY
    FUNCTION = "stitch"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "chain": (FRAME_CHAIN, {"tooltip": "Clips collected by Frame Chain Step. Stitched first, in order."}),
        }
        for i in range(1, MAX_EXTRA_INPUTS + 1):
            optional[f"images_{i}"] = ("IMAGE", {"tooltip": "Extra clip, appended after the chain in input order."})
        return {
            "required": {
                "drop_seam_frames": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Drop the first frame of every clip after the first. In a frame chain "
                        "it repeats the previous clip's last frame and causes a stutter.",
                    },
                ),
                "ramp_fast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Speed of the first part of each clip (the camera travel). "
                        "1 = normal, 2 = twice as fast.",
                    },
                ),
                "ramp_hold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Speed of the rest of each clip (the hold). 1 = normal speed.",
                    },
                ),
                "ramp_split": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Fraction of each clip played at ramp_fast before switching to ramp_hold.",
                    },
                ),
                "resize": (list(RESIZE_MODES), {"tooltip": "What to do when clips have different sizes."}),
            },
            "optional": optional,
        }

    def stitch(self, drop_seam_frames, ramp_fast, ramp_hold, ramp_split, resize, chain=None, **extra):
        clips = list(chain or ())
        clips += [extra[f"images_{i}"] for i in range(1, MAX_EXTRA_INPUTS + 1) if extra.get(f"images_{i}") is not None]
        frames = stitch(
            clips,
            drop_seam_frames=drop_seam_frames,
            fast=ramp_fast,
            hold=ramp_hold,
            split=ramp_split,
            resize=resize,
        )
        return (frames, int(frames.shape[0]))


NODE_CLASS_MAPPINGS = {
    "FXILastFrameExtract": FXILastFrameExtract,
    "FXIFrameChainStep": FXIFrameChainStep,
    "FXIStitchClips": FXIStitchClips,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXILastFrameExtract": "Last Frame Extract (FXI)",
    "FXIFrameChainStep": "Frame Chain Step (FXI)",
    "FXIStitchClips": "Stitch Clips (FXI)",
}
