"""Vimba (Allied Vision) camera interface for XPD beamline at NSLS-II."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated as A

from ophyd_async.core import (
    DetectorTriggerLogic,
    DeviceMock,
    OnOff,
    SignalDict,
    SignalR,
    SignalRW,
    StrictEnum,
    callback_on_mock_put,
    default_mock_class,
    set_mock_put_proceeds,
    set_mock_value,
    soft_signal_rw,
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
        """Prepare the camera for internally triggered acquisition.

        With no requested deadtime the frame period is set to the exposure
        plus ``driver.readout_time``, the shortest gap the camera supports.

        Alvium cameras lack the GenICam ``MultiFrame`` mode, so ADVimba runs
        ``ImageMode=Multiple`` as ``Continuous`` and delivers extra frames
        before it stops the camera. A single frame therefore uses
        ``ImageMode=Single`` (``SingleFrame``), and multi-frame internal
        acquisitions are refused; use ``bp.count(..., num=N)`` instead.
        """
        if num > 1:
            raise ValueError(
                f"Internal triggering of {num} frames per prepare is not "
                "supported: the camera delivers extra frames in "
                "ImageMode=Multiple. Use one frame per trigger, e.g. "
                "num_images=1 with bp.count([det], num=N)."
            )
        if livetime and not deadtime:
            deadtime = await self.driver.readout_time.get_value()
        image_mode = ADImageMode.SINGLE if num == 1 else ADImageMode.CONTINUOUS
        coros = [
            self.driver.trigger_source.set(VimbaTriggerSourceAlvium.SOFTWARE),
            self.driver.trigger_mode.set(OnOff.OFF),
            self.driver.exposure_mode.set(VimbaExposeOutModeAlvium.TIMED),
            self.driver.image_mode.set(image_mode),
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
        """Get default trigger info based on the number of images to acquire."""
        return await trigger_info_from_driver(
            self.driver.num_images, self.process_plugin
        )


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

        set_mock_value(device.driver.acquire_time, 0.1)
        set_mock_value(device.driver.acquire_period, 0.1 + VIMBA_READOUT_TIME)
        set_mock_value(device.driver.num_images, 1)
        set_mock_value(device.driver.trigger_source, VimbaTriggerSourceAlvium.SOFTWARE)

        async def _do_acquisition():
            trigger_source = await device.driver.trigger_source.get_value()
            if trigger_source != VimbaTriggerSourceAlvium.SOFTWARE:
                set_mock_put_proceeds(device.driver.acquire, True)
                return

            num_images = await device.driver.num_images.get_value()
            acquire_time = await device.driver.acquire_time.get_value()
            acquire_period = await device.driver.acquire_period.get_value()

            for i in range(num_images):
                await asyncio.sleep(acquire_time)
                if hdf is not None:
                    num_captured = await hdf.num_captured.get_value()
                    set_mock_value(hdf.num_captured, num_captured + 1)
                if i < num_images - 1:
                    await asyncio.sleep(max(acquire_period - acquire_time, 0))

            set_mock_value(device.driver.acquire, False)
            set_mock_put_proceeds(device.driver.acquire, True)

        def _on_acquire_write(value: bool) -> None:
            if value:
                set_mock_put_proceeds(device.driver.acquire, False)
                asyncio.ensure_future(_do_acquisition())

        callback_on_mock_put(device.driver.acquire, _on_acquire_write)


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
        to combine ``exposures_per_collection`` exposures per frame. The file
        writer must be fed from this plugin. By default None.
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
            acquire_logic=ADAcquireLogic(driver),
            trigger_logic=VimbaTriggerLogic(driver, trigger_input, proc_plugin),
            plugins=(plugins or {}) | ({"proc": proc_plugin} if proc_plugin else {}),
            config_sigs=config_sigs,
            name=name,
        )
