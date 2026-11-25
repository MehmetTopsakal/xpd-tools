from pathlib import Path

import pytest
from ophyd_async.core import (
    DetectorTrigger,
    OnOff,
    StaticPathProvider,
    TriggerInfo,
    UUIDFilenameProvider,
    init_devices,
)
from ophyd_async.epics.adcore import ADImageMode, ADWriterFactory, NDProcessIO

from xpd_tools.detectors.vimba import (
    VIMBA_READOUT_TIME,
    VimbaDetector,
    VimbaExposeOutModeAlvium,
    VimbaTriggerSourceAlvium,
)


async def _make(tmp_path: Path, trigger_input: VimbaTriggerSourceAlvium):
    path_provider = StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    async with init_devices(mock=True):
        avt = VimbaDetector(
            "AVT:",
            ADWriterFactory.hdf(path_provider),
            trigger_input=trigger_input,
            name="avt",
        )
    return avt


@pytest.fixture
async def avt(tmp_path: Path) -> VimbaDetector:
    return await _make(tmp_path, VimbaTriggerSourceAlvium.LINE_0)


@pytest.fixture
async def avt_software(tmp_path: Path) -> VimbaDetector:
    return await _make(tmp_path, VimbaTriggerSourceAlvium.SOFTWARE)


async def test_zero_deadtime_resets_period(avt: VimbaDetector):
    driver = avt.driver
    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.02, deadtime=0.98))
    assert await driver.acquire_period.get_value() == pytest.approx(1.0)

    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.02, deadtime=0))
    assert await driver.acquire_period.get_value() == pytest.approx(
        0.02 + VIMBA_READOUT_TIME
    )
    assert await driver.trigger_mode.get_value() == OnOff.OFF


@pytest.mark.parametrize(
    ("trigger_info", "exposure_mode"),
    [
        (
            TriggerInfo(
                number_of_events=3,
                trigger=DetectorTrigger.EXTERNAL_EDGE,
                livetime=0.02,
                deadtime=VIMBA_READOUT_TIME,
            ),
            VimbaExposeOutModeAlvium.TIMED,
        ),
        (
            TriggerInfo(
                number_of_events=3,
                trigger=DetectorTrigger.EXTERNAL_LEVEL,
                deadtime=VIMBA_READOUT_TIME,
            ),
            VimbaExposeOutModeAlvium.TRIGGER_WIDTH,
        ),
    ],
)
async def test_external_prepare_uses_hardware_line(
    avt: VimbaDetector,
    trigger_info: TriggerInfo,
    exposure_mode: VimbaExposeOutModeAlvium,
):
    await avt.prepare(trigger_info)
    driver = avt.driver
    assert await driver.trigger_source.get_value() == VimbaTriggerSourceAlvium.LINE_0
    assert await driver.trigger_mode.get_value() == OnOff.ON
    assert await driver.exposure_mode.get_value() == exposure_mode


@pytest.mark.parametrize(
    "trigger", [DetectorTrigger.EXTERNAL_EDGE, DetectorTrigger.EXTERNAL_LEVEL]
)
async def test_external_prepare_rejects_software_source(
    avt_software: VimbaDetector, trigger: DetectorTrigger
):
    with pytest.raises(ValueError, match="trigger_input"):
        await avt_software.prepare(
            TriggerInfo(
                number_of_events=1,
                trigger=trigger,
                livetime=0.02 if trigger == DetectorTrigger.EXTERNAL_EDGE else 0,
                deadtime=VIMBA_READOUT_TIME,
            )
        )


async def test_readout_time_sets_deadtime(avt: VimbaDetector):
    await avt.driver.readout_time.set(0.005)
    _, deadtime = await avt.get_trigger_deadtime()
    assert deadtime == 0.005
    assert "avt-driver-readout_time" in await avt.read_configuration()
    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.02))
    assert await avt.driver.acquire_period.get_value() == pytest.approx(0.025)


async def test_single_internal_frame_uses_single_image_mode(avt: VimbaDetector):
    # ImageMode=Multiple runs the Alvium in Continuous and overshoots NumImages.
    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.02))
    assert await avt.driver.image_mode.get_value() == ADImageMode.SINGLE
    assert await avt.driver.num_images.get_value() == 1


async def test_multi_frame_internal_prepare_rejected(avt: VimbaDetector):
    with pytest.raises(ValueError, match="extra frames"):
        await avt.prepare(TriggerInfo(number_of_events=3, livetime=0.02))


async def _make_with_proc(tmp_path: Path) -> VimbaDetector:
    path_provider = StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    async with init_devices(mock=True):
        avt = VimbaDetector(
            "AVT:",
            ADWriterFactory.hdf(path_provider),
            trigger_input=VimbaTriggerSourceAlvium.LINE_0,
            proc_suffix="Proc1:",
            name="avt",
        )
    return avt


async def test_proc_plugin_combines_external_exposures(tmp_path: Path):
    avt = await _make_with_proc(tmp_path)
    proc = avt.get_plugin("proc", NDProcessIO)

    await avt.prepare(
        TriggerInfo(
            number_of_events=2,
            exposures_per_collection=5,
            trigger=DetectorTrigger.EXTERNAL_EDGE,
            livetime=0.02,
            deadtime=VIMBA_READOUT_TIME,
        )
    )

    assert await proc.enable_filter.get_value()
    assert await proc.num_filter.get_value() == 5
    assert await avt.driver.num_images.get_value() == 10


async def test_internal_multi_exposure_collection_rejected(tmp_path: Path):
    # Combining exposures needs several frames per prepare, which overshoot
    # in ImageMode=Multiple on this camera.
    avt = await _make_with_proc(tmp_path)
    with pytest.raises(ValueError, match="extra frames"):
        await avt.prepare(
            TriggerInfo(number_of_events=1, exposures_per_collection=5, livetime=0.02)
        )
