"""FXI Frame Chain: seamless AI video in ComfyUI by starting every clip on the last frame of the previous one."""

try:
    from .fxi_frame_chain import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported as a top-level module (e.g. by pytest), not as a ComfyUI custom node package
    from fxi_frame_chain import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
