"""Vimba (Allied Vision) camera interface for XPD beamline at NSLS-II."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated as A

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorTriggerLogic,
    DeviceMock,
    OnOff,
    SignalDict,
    SignalR,
    SignalRW,
    StrictEnum,
    TriggerInfo,
    callback_on_mock_put,
    default_mock_class,
    set_and_wait_for_value,
    set_mock_put_proceeds,
    set_mock_value,
    soft_signal_rw,
    wait_for_value,
)
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseDataType,
    ADBaseIO,
    ADImageMode,
    ADWriterFactory,
    AreaDetector,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDProcessIO,
    prepare_exposures,
    prepare_filter,
    trigger_info_from_driver,
)
from ophyd_async.epics.core import PvSuffix

# Default frame readout time (s) for Allied Vision Alvium cameras; tune it at
# runtime through ``driver.readout_time``.
VIMBA_READOUT_TIME = 0.01
# Measured time (s) per software-triggered frame beyond the exposure is ~0.12 s;
# allow more in timeouts
VIMBA_TRIGGER_OVERHEAD = 0.5


class VimbaTriggerSourceAlvium(StrictEnum):
    """Trigger source options for Allied Vision Alvium cameras.

    Matches actual detector: XF:28ID2-ES:2{Det:AV1}cam1:TriggerSource_RBV
    with choices ('Software', 'Line 0', 'Line 1', 'Line 2', 'Line 3',
    'Timer0End', 'Timer1End')
    """

    SOFTWARE = "Software"
    LINE_0 = "Line 0"
    LINE_1 = "Line 1"
    LINE_2 = "Line 2"
    LINE_3 = "Line 3"
    TIMER_0_END = "Timer0End"
    TIMER_1_END = "Timer1End"


class VimbaOverlapAlvium(StrictEnum):
    """Trigger overlap options for Allied Vision Alvium cameras.

    Matches actual detector: XF:28ID2-ES:2{Det:AV1}cam1:TriggerOverlap_RBV
    with choices ('N.A.',)
    """

    NOT_AVAILABLE = "N.A."


class VimbaExposeOutModeAlvium(StrictEnum):
    """Exposure mode options for Allied Vision Alvium cameras.

    Matches actual detector: XF:28ID2-ES:2{Det:AV1}cam1:ExposureMode_RBV
    with choices ('Timed', 'Trigger Width', 'Trigger Controlled')
    """

    TIMED = "Timed"
    TRIGGER_WIDTH = "Trigger Width"
    TRIGGER_CONTROLLED = "Trigger Controlled"


class VimbaDriverIO(ADBaseIO):
    """Vimba driver interface for Allied Vision cameras.

    Custom version that matches actual Alvium detector capabilities.
    PV names follow the ADVimba driver template.
    """

    trigger_source: A[SignalRW[VimbaTriggerSourceAlvium], PvSuffix.rbv("TriggerSource")]
    trigger_mode: A[SignalRW[OnOff], PvSuffix.rbv("TriggerMode")]
    trigger_overlap: A[SignalRW[VimbaOverlapAlvium], PvSuffix.rbv("TriggerOverlap")]
    exposure_mode: A[SignalRW[VimbaExposeOutModeAlvium], PvSuffix.rbv("ExposureMode")]
    # bo record ('Done', 'Trigger'); writing True takes one frame when the camera
    # is acquiring with TriggerMode=On and TriggerSource=Software
    trigger_software: A[SignalRW[bool], PvSuffix("TriggerSoftware")]

    def __init__(self, prefix: str, name: str = ""):
        super().__init__(prefix, name=name)
        # Not a PV: the minimum gap between frames (s), used as the deadtime
        self.readout_time = soft_signal_rw(float, initial_value=VIMBA_READOUT_TIME)


@dataclass
class VimbaTriggerLogic(DetectorTriggerLogic):
    """Trigger logic for Vimba cameras.

    Parameters
    ----------
    driver : VimbaDriverIO
        The camera driver.
    trigger_input : VimbaTriggerSourceAlvium, optional
        External trigger source line, by default ``SOFTWARE``. Must be set
        to the wired hardware line before using external triggers.
    process_plugin : NDProcessIO, optional
        Combines ``exposures_per_collection`` exposures into each written
        frame by its filter (see ``prepare_filter``), by default None.
    """

    driver: VimbaDriverIO
    trigger_input: VimbaTriggerSourceAlvium = VimbaTriggerSourceAlvium.SOFTWARE
    process_plugin: NDProcessIO | None = None

    def config_sigs(self) -> set[SignalR]:
        return {self.driver.readout_time}

    def get_deadtime(self, config_values: SignalDict) -> float:
        return config_values[self.driver.readout_time]

    def _check_external_input(self) -> None:
        """Refuse external triggering when no hardware line is configured."""
        if self.trigger_input == VimbaTriggerSourceAlvium.SOFTWARE:
            raise ValueError(
                "External triggering needs trigger_input set to the wired "
                "hardware line (e.g. VimbaTriggerSourceAlvium.LINE_0), "
                "not SOFTWARE"
            )

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the camera for software-triggered acquisition of ``num`` frames.

        With no requested deadtime the frame period is set to the exposure
        plus ``driver.readout_time``, the shortest gap the camera supports.

        Free-running modes deliver extra frames on Alvium cameras: ADVimba runs
        ``ImageMode=Multiple`` as ``Continuous`` and overshoots, and
        ``ImageMode=Single`` sometimes starts a second exposure by itself. So
        the camera acquires continuously with ``TriggerMode=On`` and
        ``TriggerSource=Software``, and `VimbaAcquireLogic` sends one
        ``TriggerSoftware`` pulse per frame, ``num`` (``NumImages``) per
        trigger or kickoff.
        """
        if livetime and not deadtime:
            deadtime = await self.driver.readout_time.get_value()
        coros = [
            self.driver.trigger_source.set(VimbaTriggerSourceAlvium.SOFTWARE),
            self.driver.trigger_mode.set(OnOff.ON),
            self.driver.exposure_mode.set(VimbaExposeOutModeAlvium.TIMED),
            self.driver.image_mode.set(ADImageMode.CONTINUOUS),
            self.driver.num_images.set(num),
        ]
        if livetime:
            coros.append(self.driver.acquire_time.set(livetime))
            if deadtime:
                coros.append(self.driver.acquire_period.set(livetime + deadtime))
        await asyncio.gather(*coros)

    async def prepare_edge(self, num: int, livetime: float):
        """Prepare the camera to take one frame per rising trigger edge."""
        self._check_external_input()
        await asyncio.gather(
            self.driver.trigger_source.set(self.trigger_input),
            self.driver.trigger_mode.set(OnOff.ON),
            self.driver.exposure_mode.set(VimbaExposeOutModeAlvium.TIMED),
            prepare_exposures(self.driver, num, livetime),
        )

    async def prepare_level(self, num: int):
        """Prepare the camera to expose while the trigger input is high."""
        self._check_external_input()
        await asyncio.gather(
            self.driver.trigger_source.set(self.trigger_input),
            self.driver.trigger_mode.set(OnOff.ON),
            self.driver.exposure_mode.set(VimbaExposeOutModeAlvium.TRIGGER_WIDTH),
            prepare_exposures(self.driver, num),
        )

    async def prepare_exposures_per_collection(self, exposures_per_collection: int):
        await prepare_filter(self.process_plugin, num_filter=exposures_per_collection)

    async def default_trigger_info(self):
        """Get default trigger info based on the number of images to acquire.

        The collection timeout allows for every exposure that the process
        plugin averages into one frame, plus the software-trigger overhead.
        """
        info, acquire_time = await asyncio.gather(
            trigger_info_from_driver(self.driver.num_images, self.process_plugin),
            self.driver.acquire_time.get_value(),
        )
        return TriggerInfo(
            collections_per_event=info.collections_per_event,
            exposures_per_collection=info.exposures_per_collection,
            trigger=info.trigger,
            collection_timeout=(acquire_time + VIMBA_TRIGGER_OVERHEAD)
            * info.exposures_per_collection
            + DEFAULT_TIMEOUT,
        )


class VimbaAcquireLogic(ADAcquireLogic):
    """Acquire logic that takes one frame per ``TriggerSoftware`` pulse.

    When the camera is set up for software triggering (``TriggerMode=On`` and
    ``TriggerSource=Software``, see `VimbaTriggerLogic.prepare_internal`), it
    keeps acquiring between triggers and each `start_acquiring` sends
    ``NumImages`` pulses. Other modes, such as external triggering, use
    `ADAcquireLogic`.
    """

    def __init__(self, driver: VimbaDriverIO):
        super().__init__(driver)
        self.vimba_driver = driver
        self.trigger_status: AsyncStatus | None = None

    async def _software_triggered(self) -> bool:
        mode, source = await asyncio.gather(
            self.vimba_driver.trigger_mode.get_value(),
            self.vimba_driver.trigger_source.get_value(),
        )
        return mode == OnOff.ON and source == VimbaTriggerSourceAlvium.SOFTWARE

    async def _send_triggers(self, num: int) -> None:
        acquire_time = await self.driver.acquire_time.get_value()
        for _ in range(num):
            frames = await self.driver.array_counter.get_value(cached=False)
            await self.vimba_driver.trigger_software.set(True, timeout=None)
            # The camera ignores pulses while it is exposing, so wait for the frame
            await wait_for_value(
                self.driver.array_counter,
                lambda count, frames=frames: count > frames,
                timeout=acquire_time + VIMBA_TRIGGER_OVERHEAD + DEFAULT_TIMEOUT,
            )

    async def start_acquiring(self):
        if not await self._software_triggered():
            await super().start_acquiring()
            return
        if not await self.driver.acquire.get_value():
            # Completes when acquisition is stopped in ensure_stopped
            self.acquire_status = await set_and_wait_for_value(
                self.driver.acquire, True, wait_for_set_completion=False
            )
        num = await self.driver.num_images.get_value()
        self.trigger_status = AsyncStatus(self._send_triggers(num))

    async def wait_for_idle(self):
        if not await self._software_triggered():
            await super().wait_for_idle()
        elif self.trigger_status:
            # Acquisition keeps running until ensure_stopped
            await self.trigger_status

    async def ensure_stopped(self):
        if self.trigger_status and not self.trigger_status.done:
            self.trigger_status.task.cancel()
        await super().ensure_stopped()
        self.acquire_status = None
        self.trigger_status = None
        if await self._software_triggered():
            # Let Acquire from the camera GUI take frames again
            await self.vimba_driver.trigger_mode.set(OnOff.OFF)


class VimbaDetectorMock(DeviceMock["VimbaDetector"]):
    """Mock behaviour that simulates Vimba internally triggered acquisition."""

    async def connect(self, device: "VimbaDetector") -> None:
        """Mock signals to simulate Vimba camera acquisition."""
        # Alvium 1800 U-811m specs
        set_mock_value(device.driver.array_size_x, 1024)
        set_mock_value(device.driver.array_size_y, 768)
        set_mock_value(device.driver.data_type, ADBaseDataType.UINT8)

        try:
            hdf = device.get_plugin("hdf", NDFileHDF5IO)
            set_mock_value(hdf.array_size0, 768)
            set_mock_value(hdf.array_size1, 1024)
            set_mock_value(hdf.file_path_exists, True)
        except (AttributeError, TypeError):
            hdf = None

        try:
            proc = device.get_plugin("proc", NDProcessIO)
        except (AttributeError, TypeError, KeyError):
            proc = None
        # Frames the process plugin has averaged since its last output or reset
        filtered = [0]

        set_mock_value(device.driver.acquire_time, 0.1)
        set_mock_value(device.driver.acquire_period, 0.1 + VIMBA_READOUT_TIME)
        set_mock_value(device.driver.num_images, 1)
        set_mock_value(device.driver.trigger_source, VimbaTriggerSourceAlvium.SOFTWARE)

        async def _capture_frame():
            await asyncio.sleep(await device.driver.acquire_time.get_value())
            frames = await device.driver.array_counter.get_value()
            set_mock_value(device.driver.array_counter, frames + 1)
            if proc is not None and await proc.enable_filter.get_value():
                filtered[0] += 1
                if filtered[0] < await proc.num_filter.get_value():
                    return
                filtered[0] = 0
            if hdf is not None:
                num_captured = await hdf.num_captured.get_value()
                set_mock_value(hdf.num_captured, num_captured + 1)

        async def _do_acquisition():
            num_images = await device.driver.num_images.get_value()
            acquire_time = await device.driver.acquire_time.get_value()
            acquire_period = await device.driver.acquire_period.get_value()

            for i in range(num_images):
                await _capture_frame()
                if i < num_images - 1:
                    await asyncio.sleep(max(acquire_period - acquire_time, 0))

            set_mock_value(device.driver.acquire, False)
            set_mock_put_proceeds(device.driver.acquire, True)

        async def _on_acquire_write(value: bool) -> None:
            if not value:
                set_mock_put_proceeds(device.driver.acquire, True)
                return
            mode, source = await asyncio.gather(
                device.driver.trigger_mode.get_value(),
                device.driver.trigger_source.get_value(),
            )
            if mode == OnOff.OFF:
                set_mock_put_proceeds(device.driver.acquire, False)
                asyncio.ensure_future(_do_acquisition())
            elif source == VimbaTriggerSourceAlvium.SOFTWARE:
                # Frames come from TriggerSoftware pulses; the put completes when
                # acquisition is stopped
                set_mock_put_proceeds(device.driver.acquire, False)
            else:
                set_mock_put_proceeds(device.driver.acquire, True)

        def _on_trigger_software_write(value: bool) -> None:
            if value:
                asyncio.ensure_future(_capture_frame())

        callback_on_mock_put(device.driver.acquire, _on_acquire_write)
        callback_on_mock_put(device.driver.trigger_software, _on_trigger_software_write)
        if proc is not None:
            # prepare_filter resets the filter whenever it sets num_filter
            callback_on_mock_put(proc.num_filter, lambda _: filtered.__setitem__(0, 0))


@default_mock_class(VimbaDetectorMock)
class VimbaDetector(AreaDetector[VimbaDriverIO]):
    """Create a Vimba AreaDetector instance for Allied Vision cameras.

    Parameters
    ----------
    prefix : str
        EPICS PV prefix for the camera
    writer_factories : ADWriterFactory
        Factories for file writer plugins and their data logics
    driver_suffix : str, optional
        Suffix for the driver PV, by default "cam1:"
    trigger_input : VimbaTriggerSourceAlvium, optional
        External trigger source line, by default ``SOFTWARE``
    proc_suffix : str, optional
        If provided, an NDProcessIO plugin is created at this suffix and used
        to average ``exposures_per_collection`` exposures per frame. The file
        writer must be fed from this plugin, directly or through other plugins
        such as ROI. By default None.
    plugins : dict[str, NDPluginBaseIO], optional
        Additional areaDetector plugins to include, by default None
    config_sigs : Sequence[SignalR], optional
        Additional signals to include in configuration, by default ()
    name : str, optional
        Name for the camera device, by default ""
    """

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADWriterFactory,
        driver_suffix="cam1:",
        trigger_input: VimbaTriggerSourceAlvium = VimbaTriggerSourceAlvium.SOFTWARE,
        proc_suffix: str | None = None,
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = VimbaDriverIO(prefix + driver_suffix)
        proc_plugin = NDProcessIO(prefix + proc_suffix) if proc_suffix else None
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=VimbaAcquireLogic(driver),
            trigger_logic=VimbaTriggerLogic(driver, trigger_input, proc_plugin),
            plugins=(plugins or {}) | ({"proc": proc_plugin} if proc_plugin else {}),
            config_sigs=config_sigs,
            name=name,
        )
