import pytest
from bluesky.run_engine import RunEngine, call_in_bluesky_event_loop
from ophyd_async.core import (
    EnableDisable,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import NDROIIO, ADBaseColorMode, NDPluginBaseIO

from xpd_tools.detectors.utils import NDROIStartIO, plugin_array_description, set_roi


@pytest.mark.parametrize(
    ("color_mode", "size_z", "expected_shape"),
    [
        # RGB1 is pixel-interleaved, so the colour axis is last on disk.
        (ADBaseColorMode.RGB1, 3, [2336, 1752, 3]),
        (ADBaseColorMode.MONO, 0, [2336, 1752, 0]),
    ],
)
async def test_plugin_array_description(
    color_mode: ADBaseColorMode, size_z: int, expected_shape: list[int]
):
    async with init_devices(mock=True):
        plugin = NDPluginBaseIO("PLUGIN:")
    set_mock_value(plugin.array_size_x, 1752)
    set_mock_value(plugin.array_size_y, 2336)
    set_mock_value(plugin.array_size_z, size_z)
    set_mock_value(plugin.color_mode, color_mode)

    description = plugin_array_description(plugin)

    shape = [await s.get_value() for s in description.shape_signals]
    assert shape == expected_shape
    # Reported as Mono so ophyd-async does not prepend a second colour axis.
    assert await description.color_mode_signal.get_value() == ADBaseColorMode.MONO


async def test_roi_description_uses_settings_not_last_frame():
    async with init_devices(mock=True):
        roi = NDROIIO("ROI:")
    # Output size left over from the previous frame, before the ROI was changed.
    set_mock_value(roi.array_size_x, 1388)
    set_mock_value(roi.array_size_y, 1173)
    set_mock_value(roi.size_x, 2777)
    set_mock_value(roi.size_y, 1173)
    set_mock_value(roi.bin_x, 1)
    set_mock_value(roi.bin_y, 2)

    description = plugin_array_description(roi)

    shape = [await s.get_value() for s in description.shape_signals]
    assert shape == [586, 2777, 0]


def await_value(signal):
    return call_in_bluesky_event_loop(signal.get_value())


@pytest.fixture
async def roi_with_start() -> NDROIStartIO:
    async with init_devices(mock=True):
        roi = NDROIStartIO("ROI:")
    set_mock_value(roi.max_size_x, 2848)
    set_mock_value(roi.max_size_y, 2848)
    return roi


def test_set_roi_sets_region(RE: RunEngine, roi_with_start: NDROIStartIO):
    readback = RE(set_roi(roi_with_start, 40, 343, 1000, 1500, bin_x=2, bin_y=2))

    assert readback.plan_result == {
        "min_x": 40,
        "min_y": 343,
        "size_x": 1000,
        "size_y": 1500,
        "bin_x": 2,
        "bin_y": 2,
    }
    # Binned pixels are averaged so they can't overflow the data type
    assert await_value(roi_with_start.enable_scale) == EnableDisable.ENABLE
    assert await_value(roi_with_start.scale) == 4.0

    RE(set_roi(roi_with_start, 0, 0, 1000, 1500))
    assert await_value(roi_with_start.enable_scale) == EnableDisable.DISABLE


def test_set_roi_rejects_region_outside_input(
    RE: RunEngine, roi_with_start: NDROIStartIO
):
    with pytest.raises(ValueError, match="does not fit"):
        RE(set_roi(roi_with_start, 2000, 0, 1000, 1500))
    assert get_mock_put(roi_with_start.min_x).call_count == 0
