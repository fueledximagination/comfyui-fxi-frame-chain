import pytest
import torch
from helpers import frame_ids, make_clip

from fxi_frame_chain.core import (
    RESIZE_TO_FIRST,
    as_image_batch,
    chain_append,
    last_frame,
    resample,
    speed_ramp,
    stitch,
)


class TestLastFrame:
    def test_returns_final_frame_as_single_batch(self):
        out = last_frame(make_clip([1, 2, 3]))
        assert out.shape == (1, 2, 3, 3)
        assert frame_ids(out) == [3]

    def test_offset_from_end(self):
        assert frame_ids(last_frame(make_clip([1, 2, 3]), 1)) == [2]

    def test_offset_past_start_clamps_to_first_frame(self):
        assert frame_ids(last_frame(make_clip([1, 2]), 10)) == [1]

    def test_single_hwc_frame_is_promoted(self):
        out = last_frame(torch.zeros(4, 5, 3))
        assert out.shape == (1, 4, 5, 3)

    def test_result_does_not_alias_input(self):
        clip = make_clip([1, 2])
        out = last_frame(clip)
        out.fill_(0.99)
        assert frame_ids(clip) == [1, 2]

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            last_frame(make_clip([1]), -1)

    @pytest.mark.parametrize("bad", [torch.zeros(3), torch.zeros(1, 1, 1, 1, 1), torch.zeros(0, 2, 2, 3)])
    def test_bad_shapes_rejected(self, bad):
        with pytest.raises(ValueError):
            as_image_batch(bad)

    def test_non_tensor_rejected(self):
        with pytest.raises(TypeError):
            as_image_batch([[0.0]])


class TestChain:
    def test_append_builds_ordered_tuple_without_mutating(self):
        first = chain_append(None, make_clip([1]))
        second = chain_append(first, make_clip([2]))
        assert len(first) == 1
        assert [frame_ids(c) for c in second] == [[1], [2]]


class TestResample:
    def test_double_speed_keeps_every_other_frame(self):
        assert frame_ids(resample(make_clip(range(8)), 2)) == [0, 2, 4, 6]

    def test_half_speed_duplicates_frames(self):
        assert frame_ids(resample(make_clip([1, 2]), 0.5)) == [1, 1, 2, 2]

    def test_speed_one_is_identity(self):
        clip = make_clip([1, 2, 3])
        assert resample(clip, 1) is clip

    def test_never_returns_empty(self):
        assert frame_ids(resample(make_clip([7]), 8)) == [7]

    @pytest.mark.parametrize("speed", [0, -1, float("inf"), float("nan")])
    def test_invalid_speed(self, speed):
        with pytest.raises(ValueError):
            resample(make_clip([1, 2]), speed)


class TestSpeedRamp:
    def test_fast_travel_then_normal_hold(self):
        out = speed_ramp(make_clip(range(8)), fast=2, hold=1, split=0.5)
        assert frame_ids(out) == [0, 2, 4, 5, 6, 7]

    def test_split_zero_is_all_hold(self):
        assert frame_ids(speed_ramp(make_clip(range(4)), fast=2, hold=1, split=0)) == [0, 1, 2, 3]

    def test_split_one_is_all_fast(self):
        assert frame_ids(speed_ramp(make_clip(range(4)), fast=2, hold=1, split=1)) == [0, 2]

    def test_invalid_split(self):
        with pytest.raises(ValueError):
            speed_ramp(make_clip([1]), split=1.5)


class TestStitch:
    def test_drops_duplicated_seam_frame(self):
        # Chained: clip 2 starts on clip 1's last frame (3), clip 3 on clip 2's (5).
        clips = [make_clip([1, 2, 3]), make_clip([3, 4, 5]), make_clip([5, 6])]
        assert frame_ids(stitch(clips)) == [1, 2, 3, 4, 5, 6]

    def test_keeps_seam_frames_when_asked(self):
        clips = [make_clip([1, 2]), make_clip([2, 3])]
        assert frame_ids(stitch(clips, drop_seam_frames=False)) == [1, 2, 2, 3]

    def test_single_frame_clip_is_never_dropped_entirely(self):
        clips = [make_clip([1]), make_clip([2])]
        assert frame_ids(stitch(clips)) == [1, 2]

    def test_applies_speed_ramp_per_clip(self):
        clips = [make_clip(range(0, 8)), make_clip(range(7, 16))]
        out = stitch(clips, fast=2, hold=1, split=0.5)
        # clip 2 loses its seam frame (7) first, leaving 8..15, then ramps.
        assert frame_ids(out) == [0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]

    def test_size_mismatch_errors_by_default(self):
        with pytest.raises(ValueError, match="resize"):
            stitch([make_clip([1]), make_clip([2], height=4, width=6)])

    def test_size_mismatch_resizes_to_first_clip(self):
        out = stitch([make_clip([1, 2]), make_clip([2, 3], height=4, width=6)], resize=RESIZE_TO_FIRST)
        assert out.shape == (3, 2, 3, 3)
        assert frame_ids(out) == [1, 2, 3]

    def test_channel_mismatch_errors(self):
        with pytest.raises(ValueError, match="channels"):
            stitch([make_clip([1]), make_clip([2], channels=4)])

    def test_nothing_to_stitch(self):
        with pytest.raises(ValueError):
            stitch([])

    def test_unknown_resize_mode(self):
        with pytest.raises(ValueError):
            stitch([make_clip([1])], resize="stretch")

    def test_output_matches_first_clip_dtype(self):
        out = stitch([make_clip([1]), make_clip([1, 2]).double()])
        assert out.dtype == torch.float32
