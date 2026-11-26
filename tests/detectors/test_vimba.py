from pathlib import Path

import pytest
from ophyd_async.core import (
    DetectorTrigger,
    OnOff,
    StaticPathProvider,
    TriggerInfo,
    UUIDFilenameProvider,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import (
    ADImageMode,
    ADWriterFactory,
    NDFileHDF5IO,
    NDProcessIO,
)

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


async def test_internal_prepare_uses_software_trigger(avt: VimbaDetector):
    # Free-running ImageMode=Single/Multiple deliver extra frames on the Alvium.
    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.02))
    driver = avt.driver
    assert await driver.trigger_mode.get_value() == OnOff.ON
    assert await driver.trigger_source.get_value() == VimbaTriggerSourceAlvium.SOFTWARE
    assert await driver.image_mode.get_value() == ADImageMode.CONTINUOUS
    assert await driver.num_images.get_value() == 1


async def test_internal_trigger_takes_one_frame_per_software_pulse(
    avt: VimbaDetector,
):
    hdf = avt.get_plugin("hdf", NDFileHDF5IO)
    await avt.stage()
    await avt.prepare(TriggerInfo(number_of_events=1, livetime=0.01))
    for _ in range(3):
        await avt.trigger()
    assert await hdf.num_captured.get_value() == 3
    assert get_mock_put(avt.driver.trigger_software).call_count == 3
    # The camera keeps acquiring between triggers
    assert await avt.driver.acquire.get_value()

    await avt.unstage()
    assert not await avt.driver.acquire.get_value()
    # So Acquire from the camera GUI takes frames again
    assert await avt.driver.trigger_mode.get_value() == OnOff.OFF


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


async def test_internal_trigger_averages_exposures_in_proc_plugin(tmp_path: Path):
    avt = await _make_with_proc(tmp_path)
    hdf = avt.get_plugin("hdf", NDFileHDF5IO)
    await avt.stage()
    await avt.prepare(
        TriggerInfo(number_of_events=1, exposures_per_collection=3, livetime=0.01)
    )
    for _ in range(2):
        await avt.trigger()
    assert await hdf.num_captured.get_value() == 2
    assert get_mock_put(avt.driver.trigger_software).call_count == 6
    await avt.unstage()


async def test_unprepared_trigger_averages_proc_filter_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # With preserved detector state, trigger() reads exposures per collection
    # from the process plugin filter, as in the beamline profile.
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    avt = await _make_with_proc(tmp_path)
    proc = avt.get_plugin("proc", NDProcessIO)
    set_mock_value(proc.enable_filter, True)
    set_mock_value(proc.num_filter, 4)
    set_mock_value(avt.driver.acquire_time, 0.01)
    hdf = avt.get_plugin("hdf", NDFileHDF5IO)
    await avt.stage()
    await avt.trigger()
    assert await hdf.num_captured.get_value() == 1
    assert get_mock_put(avt.driver.trigger_software).call_count == 4
    await avt.unstage()
