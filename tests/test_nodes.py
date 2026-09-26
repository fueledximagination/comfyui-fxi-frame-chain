import json
from pathlib import Path

import pytest
from helpers import frame_ids, make_clip

from fxi_frame_chain.nodes import (
    FRAME_CHAIN,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    FXIFrameChainStep,
    FXILastFrameExtract,
    FXIStitchClips,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeComponents:
    def __init__(self, images):
        self.images = images


class FakeVideo:
    """Duck-types ComfyUI's VIDEO input: get_components().images."""

    def __init__(self, images):
        self._images = images

    def get_components(self):
        return FakeComponents(self._images)


def test_every_node_is_registered_with_a_display_name():
    assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)
    for cls in NODE_CLASS_MAPPINGS.values():
        spec = cls.INPUT_TYPES()
        assert "required" in spec
        assert callable(getattr(cls(), cls.FUNCTION))
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert cls.CATEGORY.startswith("FXI")


def test_last_frame_extract_from_images():
    frame, count = FXILastFrameExtract().extract(0, images=make_clip([1, 2, 3]))
    assert frame_ids(frame) == [3]
    assert count == 3


def test_last_frame_extract_from_video():
    frame, count = FXILastFrameExtract().extract(1, video=FakeVideo(make_clip([4, 5, 6])))
    assert frame_ids(frame) == [5]
    assert count == 3


def test_last_frame_extract_needs_an_input():
    with pytest.raises(ValueError):
        FXILastFrameExtract().extract(0)


def test_chain_step_then_stitch_end_to_end():
    step = FXIFrameChainStep()
    chain, start2, n = step.step(make_clip([1, 2, 3]), 0)
    assert frame_ids(start2) == [3] and n == 1
    # "Generate" clip 2 from start2: its first frame is the handed-on frame.
    clip2 = make_clip([frame_ids(start2)[0], 4, 5])
    chain, start3, n = step.step(clip2, 0, chain=chain)
    assert frame_ids(start3) == [5] and n == 2
    frames, count = FXIStitchClips().stitch(True, 1.0, 1.0, 0.5, "error", chain=chain)
    assert frame_ids(frames) == [1, 2, 3, 4, 5]
    assert count == 5


def test_stitch_appends_extra_image_inputs_after_chain():
    chain, _, _ = FXIFrameChainStep().step(make_clip([1, 2]), 0)
    frames, _ = FXIStitchClips().stitch(
        False, 1.0, 1.0, 0.5, "error", chain=chain, images_2=make_clip([9]), images_1=make_clip([8])
    )
    assert frame_ids(frames) == [1, 2, 8, 9]


def test_stitch_without_chain_uses_image_inputs():
    frames, _ = FXIStitchClips().stitch(
        True, 1.0, 1.0, 0.5, "error", images_1=make_clip([1, 2]), images_2=make_clip([2, 3])
    )
    assert frame_ids(frames) == [1, 2, 3]


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.json")), ids=lambda p: p.name)
def test_example_workflows_are_wired_consistently(path):
    workflow = json.loads(path.read_text())
    ours = {n["class_type"] for n in workflow.values() if n["class_type"].startswith("FXI")}
    assert ours <= set(NODE_CLASS_MAPPINGS)
    for node_id, node in workflow.items():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in workflow, f"node {node_id} links to missing node {value[0]}"
        if node["class_type"] in NODE_CLASS_MAPPINGS:
            spec = NODE_CLASS_MAPPINGS[node["class_type"]].INPUT_TYPES()
            known = {**spec["required"], **spec.get("optional", {})}
            assert set(node["inputs"]) <= set(known), f"node {node_id} has unknown inputs"
            assert set(spec["required"]) <= set(node["inputs"]), f"node {node_id} is missing required inputs"


def test_frame_chain_type_name_is_namespaced():
    assert FRAME_CHAIN.startswith("FXI_")
