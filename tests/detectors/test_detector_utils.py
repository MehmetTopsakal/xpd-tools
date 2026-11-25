import pytest
from ophyd_async.core import init_devices, set_mock_value
from ophyd_async.epics.adcore import NDROIIO, ADBaseColorMode, NDPluginBaseIO

from xpd_tools.detectors.utils import plugin_array_description


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
