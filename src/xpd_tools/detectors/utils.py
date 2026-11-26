"""Utility functions for xpd_tools detectors."""

from collections.abc import Generator, Sequence
from typing import Annotated as A
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.utils import Msg
from ophyd_async.core import EnableDisable, SignalR, SignalRW, derived_signal_r
from ophyd_async.epics.adcore import (
    NDROIIO,
    ADBaseColorMode,
    AreaDetector,
    NDArrayBaseIO,
    NDArrayDescription,
    NDPluginBaseIO,
    NDProcessFilterType,
    NDProcessIO,
)
from ophyd_async.epics.core import PvSuffix


class NDROIStartIO(NDROIIO):
    """ROI plugin with the start, input size and scaling that `NDROIIO` lacks."""

    min_x: A[SignalRW[int], PvSuffix.rbv("MinX")]
    min_y: A[SignalRW[int], PvSuffix.rbv("MinY")]
    max_size_x: A[SignalR[int], PvSuffix("MaxSizeX_RBV")]
    max_size_y: A[SignalR[int], PvSuffix("MaxSizeY_RBV")]
    enable_scale: A[SignalRW[EnableDisable], PvSuffix.rbv("EnableScale")]
    scale: A[SignalRW[float], PvSuffix.rbv("Scale")]


def set_roi(
    roi: NDROIStartIO,
    min_x: int,
    min_y: int,
    size_x: int,
    size_y: int,
    bin_x: int = 1,
    bin_y: int = 1,
) -> Generator[Msg, Any, dict[str, int]]:
    """Set the region of an ROI plugin.

    For example ``RE(set_roi(avt1_roi1, 0, 0, 1000, 1500, bin_x=2, bin_y=2))``.
    The next scan's declared frame shape follows the new ROI, because
    `plugin_array_description` reads it from ``SizeX/BinX`` and ``SizeY/BinY``.

    Binned pixels are averaged rather than summed (``Scale`` is set to
    ``bin_x * bin_y``), so they keep the camera's range and data type: a sum
    would overflow, e.g. 4x4 binning of 8-bit pixels.

    Parameters
    ----------
    roi : NDROIStartIO
        The ROI plugin.
    min_x, min_y : int
        Start of the region (pixels of the plugin's input).
    size_x, size_y : int
        Size of the region before binning.
    bin_x, bin_y : int, optional
        Binning, by default 1.

    Returns
    -------
    dict[str, int]
        The settings read back from the IOC.
    """
    if min(min_x, min_y) < 0 or min(size_x, size_y, bin_x, bin_y) < 1:
        raise ValueError(
            "min_x and min_y must be >= 0, and size_x, size_y, bin_x and bin_y >= 1"
        )
    max_x: int = yield from bps.rd(roi.max_size_x)
    max_y: int = yield from bps.rd(roi.max_size_y)
    if min_x + size_x > max_x or min_y + size_y > max_y:
        raise ValueError(
            f"Region x={min_x}..{min_x + size_x}, y={min_y}..{min_y + size_y} "
            f"does not fit the {max_x} x {max_y} input of {roi.name}"
        )
    # Start first, so the size is never checked against the old start
    yield from bps.mv(roi.min_x, min_x, roi.min_y, min_y)
    yield from bps.mv(
        roi.size_x, size_x, roi.size_y, size_y, roi.bin_x, bin_x, roi.bin_y, bin_y
    )
    num_binned = bin_x * bin_y
    yield from bps.mv(
        roi.scale,
        float(num_binned),
        roi.enable_scale,
        EnableDisable.ENABLE if num_binned > 1 else EnableDisable.DISABLE,
    )
    readback: dict[str, int] = {}
    for key in ("min_x", "min_y", "size_x", "size_y", "bin_x", "bin_y"):
        readback[key] = yield from bps.rd(getattr(roi, key))
    print(
        f"{roi.name}: start ({readback['min_x']}, {readback['min_y']}), "
        f"size {readback['size_x']} x {readback['size_y']}, "
        f"bin {readback['bin_x']} x {readback['bin_y']} -> frames "
        f"{readback['size_x'] // readback['bin_x']} x "
        f"{readback['size_y'] // readback['bin_y']} (x by y)"
    )
    return readback


def _roi_output_size(size: int, binning: int) -> int:
    return size // max(binning, 1)


def plugin_array_description(plugin: NDArrayBaseIO) -> NDArrayDescription:
    """Describe the frames a file writer receives from ``plugin``.

    Use as ``ADWriterFactory.hdf(..., array_description=...)`` when the file
    writer is fed by a plugin (ROI, transform, ...) that changes the frame
    size, so the declared shape matches what is written.

    The shape is ``(y, x[, ArraySizeZ])``, which is the on-disk order for both
    Mono and pixel-interleaved RGB1 (``(y, x, 3)``). ophyd-async would put the
    RGB1 colour axis first, so RGB1 is reported as Mono here; the colour axis
    is already in the shape. Other colour modes are still rejected by
    ophyd-async.

    For an `NDROIIO`, x and y come from ``SizeX/BinX`` and ``SizeY/BinY``,
    which update as soon as they are written. For other plugins they come from
    ``ArraySizeX/Y``, which only update when a frame passes through, so take
    a frame after changing such a plugin.

    The plugin must be passed to the detector's ``plugins`` so its signals
    connect.
    """

    def _color_mode(color_mode: ADBaseColorMode) -> ADBaseColorMode:
        if color_mode == ADBaseColorMode.RGB1:
            return ADBaseColorMode.MONO
        return color_mode

    color_mode_signal: SignalR[ADBaseColorMode] = derived_signal_r(
        _color_mode, color_mode=plugin.color_mode
    )
    if isinstance(plugin, NDROIIO):
        size_y: SignalR[int] = derived_signal_r(
            _roi_output_size, size=plugin.size_y, binning=plugin.bin_y
        )
        size_x: SignalR[int] = derived_signal_r(
            _roi_output_size, size=plugin.size_x, binning=plugin.bin_x
        )
    else:
        size_y, size_x = plugin.array_size_y, plugin.array_size_x
    return NDArrayDescription(
        shape_signals=[size_y, size_x, plugin.array_size_z],
        data_type_signal=plugin.data_type,
        color_mode_signal=color_mode_signal,
    )


def get_detector_acq_times(
    detectors: list[AreaDetector],
) -> Generator[Msg, Any, list[float]]:
    """Get the acquisition times and periods for a list of detectors.

    Parameters
    ----------
    detectors : list[AreaDetector]
        List of area detectors to query.

    Returns
    -------
    list[float]
        List of tuples containing the acquisition time and period
        for each detector. If the acquisition period is less than
        the acquisition time, it will be set to the acquisition time.
    """
    acquisition_periods: list[float] = []
    for det in detectors:
        acq_time: float = yield from bps.rd(det.driver.acquire_time)
        acq_period: float = yield from bps.rd(det.driver.acquire_period)
        if acq_period < acq_time:
            acq_period = acq_time
        acquisition_periods.append(acq_period)
    return acquisition_periods


def set_proc_filter_type(
    dets: Sequence[AreaDetector], filter_type: NDProcessFilterType | str
) -> Generator[Msg, Any, None]:
    """Set the Proc plugin's filter type on every detector that has one.

    ophyd-async's prepare leaves the filter type to the IOC, so without this an
    averaged acquisition uses whatever the Proc plugin was last left on.

    Parameters
    ----------
    dets : Sequence[AreaDetector]
        The detectors; those without a ``proc`` plugin are skipped.
    filter_type : NDProcessFilterType | str
        The filter type, or its EPICS name (e.g. ``"Average"``).
    """
    filter_type = NDProcessFilterType(filter_type)
    sigs: list[Any] = []
    for det in dets:
        proc = getattr(det, "proc", None)
        if isinstance(proc, NDProcessIO):
            sigs.extend([proc.filter_type, filter_type])
    if sigs:
        yield from bps.mv(*sigs)


def set_plugin_queue_sizes(
    dets: AreaDetector | Sequence[AreaDetector], queue_size: int
) -> Generator[Msg, Any, None]:
    """Set the queue size of every AreaDetector plugin to a given value.

    A flyscan sets this to the number of exposures, so the plugins can buffer the
    whole scan without dropping frames.

    Parameters
    ----------
    dets : AreaDetector | Sequence[AreaDetector]
        The detector or list of detectors whose plugin queue sizes are to be set.
    queue_size : int
        The desired queue size for all plugins.
    """
    if isinstance(dets, AreaDetector):
        dets = [dets]
    sigs: list[Any] = []
    for det in dets:
        for _, child in det.children():
            if isinstance(child, NDPluginBaseIO):
                sigs.extend([child.queue_size, queue_size])
    if sigs:
        yield from bps.mv(*sigs)
