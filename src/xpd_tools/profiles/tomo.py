"""XPD-D Bluesky profile for the 28-ID-2 beamline at NSLS-II."""

import os

# Remove PYEPICS_LIBCA set by the conda epics-base activation script.
# It points to conda's libca.so which conflicts with epicscorelibs' libca.so
# (used by aioca/ophyd-async), causing PV connections to fail and the process
# to hang on exit with "double free or corruption".
os.environ.pop("PYEPICS_LIBCA", None)

import asyncio
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import (
    RunEngine,
    autoawait_in_bluesky_event_loop,
    call_in_bluesky_event_loop,  # noqa: F401 - operator namespace
)
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import BufferingWrapper, TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.devices import Xspress3Detector, Xspress3HDFWriterFactory
from nslsii.ophyd_async.providers import NSLS2PathProvider
from nslsii.sync_experiment import sync_experiment
from ophyd_async.core import StandardReadable, init_devices, print_connection_status, PathInfo, PathProvider
from ophyd_async.epics.adcore import (
    ADWriterFactory,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDProcessIO,
    NDStatsIO,
    PluginSignalDataLogic,
)
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.core import epics_signal_rw
from ophyd_async.sim import SimMotor
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from ophyd_async.fastcs.panda import HDFPanda
from ophyd_async.plan_stubs import retrieve_settings, store_settings
from rich import print as rprint
from tiled.client import from_uri

from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    SignalR,
    StandardReadable,
    StrictEnum,
    default_mock_class,
    derived_signal_r,
    observe_signals_value,
    wait_for_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)


from xpd_tools.detectors import (
    EmergentVisionDetector,
    ProsilicaLegacyDetector,
    VimbaDetector,
)
from xpd_tools.detectors.dexela import DexelaDetector
from xpd_tools.detectors.pilatus4 import Pilatus4Detector
from xpd_tools.detectors.utils import NDROIStartIO, plugin_array_description, set_roi
from xpd_tools.machine import NSLS2OpsMode, NSLS2StorageRing
from xpd_tools.motors import RotationMotor, SampleStack, Slits
from xpd_tools.plans import *
from xpd_tools.psylotech import Psylotech
from xpd_tools.shutter import FastShutter, PPSShutter, Shutter
from xpd_tools.suspenders import SuspendFloorUnlessOpsMode
from xpd_tools.tomo import *
from xpd_tools.tomo import LastDarkFlatUIDTracker
from xpd_tools.utils import (
    ProposalIDPrompt,
    initialize_run_engine,
    show_docs,
)


def start_beamtime(proposal_id: int, verbose: bool = True) -> None:
    """Start a beamtime for the given proposal ID."""
    md = sync_experiment(proposal_id, "XPD", redis_ssl=True)

    lines = [
        f"Started beamtime for proposal ID [bold][blue]{proposal_id}[/blue][/bold].",
        f"Current time: [italic]{datetime.now():%Y-%m-%d %H:%M:%S}[/italic]",
    ]
    proposal_md = md.get("proposal", {})
    if proposal_md and verbose:
        lines += [
            f"Proposal title: [italic]{proposal_md['title']}[/italic]",
            f"Proposal type: [italic]{proposal_md['type']}[/italic], "
            f"Proposal PI: [italic]{proposal_md['pi_name']}[/italic]",
        ]
    # One rprint call: in Jupyter each call is a separate, padded output block.
    rprint("\n".join(lines))


# Allow for using env var to run in simulation mode
XPDTOOLS_RUNNING_IN_CI = (
    os.environ.get("XPDTOOLS_RUNNING_IN_CI", "false").lower() == "true"
)

TILED_BACKUP_ROOT = "/nsls2/data/xpd-new/xpdd/shared/config/bluesky/tiled_backup"
TILED_LOCAL_BACKUP_ROOT = "/tmp/bluesky-docs"

os.environ["REDIS_HOST"] = "xf28id2-xpdd-redis1.nsls2.bnl.gov"
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

rprint("[bold][blue]Starting the XPDD profile collection...[/blue][/bold]")

# Create our RunEngine instance
RE: RunEngine = initialize_run_engine()

# Set some metadata that never changes.
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "XPD-D"
RE.md["beamline_id"] = "28-ID-2"

# Display current proposal information, if set. Use one rprint call: in Jupyter
# each call is a separate, padded output block.
_banner = ["Initialized RunEngine...", ""]
_banner.append("[bold][underline]Current Proposal Information[/underline][/bold]")
_proposal_md = RE.md.get("proposal", {})
if _proposal_md:
    _banner += [
        "[bold]  - Title: [green]"
        f"[italic]{_proposal_md['title']}[/italic][/green][/bold]",
        f"[bold]  - Type:  [italic]{_proposal_md['type']}[/italic][/bold]",
        f"[bold]  - PI:    [italic]{_proposal_md['pi_name']}[/italic][/bold]",
    ]
else:
    _banner.append("[bold][red]No active proposal![/red][/bold]")
_banner.append(
    "[bold]To switch proposals, use start_beamtime(123456) replacing the number "
    "with your proposal ID.[/bold]"
)
rprint("\n".join(_banner))
del _banner, _proposal_md

# Setup terminal progress bars
RE.waiting_hook = ProgressBarManager()  # type: ignore[assignment]

if not XPDTOOLS_RUNNING_IN_CI:
    # Initialize our tiled writing client
    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_XPDD", ""),
    )["xpdd"]["migration"]
    tw = TiledWriter(tiled_writing_client)
else:
    # When testing, spin up an in-process tiled server, and use that instead
    from tiled.client import simple

    tiled_reading_client = tiled_writing_client = c = simple(
        # Tiled refuses API keys that are not alphanumeric.
        "/tmp/xpd_tools",
        api_key="xpdtools",
        readable_storage="/tmp/xpd_tools",
    )
    tw = BufferingWrapper(
        TiledWriter(
            tiled_writing_client,
            backup_directory=f"{TILED_LOCAL_BACKUP_ROOT}/migration",
        )
    )
RE.subscribe(tw)

# Setup BEC for live table, header, and live plots
bec = BestEffortCallback()
RE.subscribe(bec)

RE.subscribe(LastDarkFlatUIDTracker(RE.md))

# Setup the proposal ID prompt, and let `await coro()` run in the IPython shell
ipython = get_ipython()
if ipython is not None:
    # Run `await coro()` typed in the shell/notebook on the RunEngine's event loop.
    autoawait_in_bluesky_event_loop(ipython)
    # The custom prompt only applies to the terminal; Jupyter uses ZMQInteractiveShell.
    if isinstance(ipython, TerminalInteractiveShell):
        ipython.prompts = ProposalIDPrompt(RE, ipython)
    # ipython.run_line_magic("matplotlib", "")
    if not XPDTOOLS_RUNNING_IN_CI:
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["xpdd"][
            "migration"
        ]


################
# FILE PATHS   #
################
PROPOSALS_DIR = PurePosixPath("/nsls2/data/xpd-new/proposals")


def _windows_path_provider() -> NSLS2PathProvider:
    """Path provider for IOCs on Windows, which see the proposals dir as Z:."""
    return NSLS2PathProvider(
        RE.md,  # type: ignore (nslsii types metadata_dict as dict; RE.md is a RedisJSONDict)
        base_data_dir=PROPOSALS_DIR,
        base_write_dir=PureWindowsPath("Z:\\proposals"),
        granularity="year",
    )


path_provider = NSLS2PathProvider(
    RE.md,  # type: ignore (nslsii types metadata_dict as dict; RE.md is a RedisJSONDict)
    base_data_dir=PROPOSALS_DIR,
)
dexela1_path_provider = _windows_path_provider()
avt1_path_provider = _windows_path_provider()
evt1_path_provider = _windows_path_provider()


class FixedAssetDirPathProvider(PathProvider):
    """Use a fixed asset directory name, independent of the device name."""

    def __init__(self, provider: PathProvider, asset_dir_name: str):
        self._provider = provider
        self._asset_dir_name = asset_dir_name

    def __call__(self, datakey_name: str | None = None) -> PathInfo:
        return self._provider(self._asset_dir_name)

xs3_path_provider = FixedAssetDirPathProvider(path_provider, "xspress3-1")

prosilica1_path_provider = FixedAssetDirPathProvider(path_provider, "default")


#############
# DETECTORS #
#############
# Stats plugins are created outside init_devices so they take their name from the
# parent detector (e.g. pilatus1-stats1) and can be used in the data logics below.
pilatus1_stats1 = NDStatsIO("XF:28ID2-ES{Pilatus4-Det:1}Stats1:")
iris1_stats1 = NDStatsIO("XF:28IDD-ES{Iris-Det:1}Stats1:")
dexela1_stats1 = NDStatsIO("XF:28IDD-ES:2{Det:DEX}Stats1:")

# The *_fly detectors are copies without the stats data logic, which is too slow
# for fly scans. They share the step-scan detector's name so data keys and asset
# directories match.
# TODO: Remove them once a new ophyd-async release auto-disables step-scan-only
# data logic in fly scans.

# If an IOC was down at startup, connect its detector later without re-running
# the profile, e.g. call_in_bluesky_event_loop(avt1.connect())
with init_devices(
    mock=XPDTOOLS_RUNNING_IN_CI, timeout=2.0, on_connect_hook=print_connection_status
):
    panda1 = HDFPanda("XF:28ID2-ES{PANDA:1}:", path_provider)

    # --- Pilatus4 ---
    pilatus1 = Pilatus4Detector(
        "XF:28ID2-ES{Pilatus4-Det:1}",
        ADWriterFactory.hdf(path_provider),
        plugins={"stats1": pilatus1_stats1},
    )
    # Include the stats total in step-scan readings
    pilatus1.add_detector_logics(
        PluginSignalDataLogic(pilatus1.driver, pilatus1_stats1.total)
    )
    pilatus1_fly = Pilatus4Detector(  # practical at ~450 Hz
        "XF:28ID2-ES{Pilatus4-Det:1}", ADWriterFactory.hdf(path_provider)
    )
    pilatus1_fly.set_name("pilatus1")

    # --- Dexela ---
    dexela1 = DexelaDetector(
        "XF:28IDD-ES:2{Det:DEX}",
        ADWriterFactory.hdf(dexela1_path_provider),
        plugins={"stats1": dexela1_stats1},
        proc_suffix="Proc1:",
    )
    dexela1.add_detector_logics(
        PluginSignalDataLogic(dexela1.driver, dexela1_stats1.total)
    )
    dexela1_fly = DexelaDetector(
        "XF:28IDD-ES:2{Det:DEX}",
        ADWriterFactory.hdf(dexela1_path_provider),
        proc_suffix="Proc1:",
    )
    dexela1_fly.set_name("dexela1")
    # Live view branch (DEX1 -> ROI1 -> PVA1), wired by dexela1_set_plugin_chain().
    # Files are written by DEX1 -> PROC1 -> HDF1, so this ROI doesn't change them.
    dexela1_roi1 = NDROIStartIO("XF:28IDD-ES:2{Det:DEX}ROI1:")
    dexela1_pva1 = NDPluginBaseIO("XF:28IDD-ES:2{Det:DEX}Pva1:")

    # --- Iris (Kinetix) ---
    # HDF1 is fed by ROI1, so describe the written frames from ROI1's output
    # rather than the full-sensor driver size.
    iris1_roi1 = NDROIStartIO("XF:28IDD-ES{Iris-Det:1}ROI1:")
    iris1 = KinetixDetector(
        "XF:28IDD-ES{Iris-Det:1}",
        ADWriterFactory.hdf(
            path_provider, array_description=plugin_array_description(iris1_roi1)
        ),
        proc_suffix="Proc1:",
        plugins={"stats1": iris1_stats1, "roi1": iris1_roi1},
    )
    iris1.add_detector_logics(PluginSignalDataLogic(iris1.driver, iris1_stats1.total))
    # A device can only have one parent, so the fly detector gets its own ROI1.
    iris1_fly_roi1 = NDROIStartIO("XF:28IDD-ES{Iris-Det:1}ROI1:")
    iris1_fly = KinetixDetector(
        "XF:28IDD-ES{Iris-Det:1}",
        ADWriterFactory.hdf(
            path_provider, array_description=plugin_array_description(iris1_fly_roi1)
        ),
        proc_suffix="Proc1:",
        plugins={"roi1": iris1_fly_roi1},
    )
    iris1_fly.set_name("iris1")

    # --- Prosilica ---
    # The chain is CAM -> TRANS1 -> ROI1 -> PROC1 -> HDF1 (wired by
    # prosilica1_set_plugin_chain() below). ROI1 crops the transformed RGB1
    # frames and PROC1 averages exposures_per_collection frames (its NumFilter)
    # into each written frame without changing its size, so describe the
    # written frames from ROI1's output rather than the full-sensor size.
    prosilica1_trans1 = NDPluginBaseIO("XF:28IDD-BI{Det-Sample:1}Trans1:")
    prosilica1_roi1 = NDROIStartIO("XF:28IDD-BI{Det-Sample:1}ROI1:")
    prosilica1 = ProsilicaLegacyDetector(
        "XF:28IDD-BI{Det-Sample:1}",
        ADWriterFactory.hdf(
            prosilica1_path_provider,
            array_description=plugin_array_description(prosilica1_roi1),
        ),
        proc_suffix="Proc1:",
        plugins={"trans1": prosilica1_trans1, "roi1": prosilica1_roi1},
        name="prosilica1",
    )

    # --- AVT (Vimba) ---
    # The chain is VMB1 -> ROI1 -> PROC1 -> HDF1 (wired by avt1_set_plugin_chain()
    # below). PROC1 averages exposures_per_collection frames (its NumFilter) of
    # the cropped image into each written frame without changing its size, so
    # describe the written frames from ROI1's output rather than the full-sensor
    # driver size.
    avt1_roi1 = NDROIStartIO("XF:28ID2-ES:2{Det:AV1}ROI1:")
    avt1_trans1 = NDPluginBaseIO("XF:28ID2-ES:2{Det:AV1}Trans1:")
    avt1 = VimbaDetector(
        "XF:28ID2-ES:2{Det:AV1}",
        ADWriterFactory.hdf(
            avt1_path_provider, array_description=plugin_array_description(avt1_roi1)
        ),
        proc_suffix="Proc1:",
        plugins={"roi1": avt1_roi1, "trans1": avt1_trans1},
        name="avt1",
    )

    # --- Emergent Vision ---
    # The chain is EVT1 -> TRANS1 -> ROI1 -> PROC1 -> HDF1 (wired by
    # evt1_set_plugin_chain() below). PROC1 averages exposures_per_collection
    # frames (its NumFilter) into each written frame without changing its size,
    # so describe the written frames from ROI1's output rather than the
    # full-sensor driver size.
    evt1_roi1 = NDROIStartIO("XF:28IDD-EM1{EVT-Cam:1}ROI1:")
    evt1_trans1 = NDPluginBaseIO("XF:28IDD-EM1{EVT-Cam:1}Trans1:")
    evt1 = EmergentVisionDetector(
        "XF:28IDD-EM1{EVT-Cam:1}",
        ADWriterFactory.hdf(
            evt1_path_provider, array_description=plugin_array_description(evt1_roi1)
        ),
        proc_suffix="Proc1:",
        plugins={"roi1": evt1_roi1, "trans1": evt1_trans1},
        name="evt1",
    )

    # --- Xspress3 fluorescence detector
    xs3 = Xspress3Detector(
        "XF:28ID2-ES{Xsp:1}:",
        Xspress3HDFWriterFactory(xs3_path_provider, hinted_streams=[]),
        channel_numbers=(1,),
        mca_roi_numbers=(1,2),
        name="xs3", # was "xspress3-1"
    )


async def _avt1_set_plugin_chain() -> None:
    # Upstream first, so no plugin is ever fed from its own output
    await avt1_roi1.nd_array_port.set("VMB1")
    await avt1.get_plugin("proc", NDProcessIO).nd_array_port.set("ROI1")
    await avt1.get_plugin("hdf", NDFileHDF5IO).nd_array_port.set("PROC1")


def avt1_set_plugin_chain() -> None:
    """Wire the avt1 plugins as VMB1 -> ROI1 -> PROC1 -> HDF1.

    The IOC does not restore this wiring on restart, so call this again after
    restarting the AVT IOC. Call it from a notebook cell that has no ``await``.
    """
    call_in_bluesky_event_loop(_avt1_set_plugin_chain())


try:
    avt1_set_plugin_chain()
except Exception as _exc:
    rprint(f"[bold][red]Could not set the avt1 plugin chain: {_exc!r}[/red][/bold]")


async def _prosilica1_set_plugin_chain() -> None:
    # Upstream first, so no plugin is ever fed from its own output
    await prosilica1_trans1.nd_array_port.set("CAM")
    await prosilica1_roi1.nd_array_port.set("TRANS1")
    await prosilica1.get_plugin("proc", NDProcessIO).nd_array_port.set("ROI1")
    await prosilica1.get_plugin("hdf", NDFileHDF5IO).nd_array_port.set("PROC1")


def prosilica1_set_plugin_chain() -> None:
    """Wire the prosilica1 plugins as CAM -> TRANS1 -> ROI1 -> PROC1 -> HDF1.

    Call this again after restarting the Prosilica IOC. Call it from a notebook
    cell that has no ``await``.
    """
    call_in_bluesky_event_loop(_prosilica1_set_plugin_chain())


try:
    prosilica1_set_plugin_chain()
except Exception as _exc:
    rprint(
        f"[bold][red]Could not set the prosilica1 plugin chain: {_exc!r}[/red][/bold]"
    )


async def _dexela1_set_plugin_chain() -> None:
    # Upstream first, so no plugin is ever fed from its own output
    await asyncio.gather(
        dexela1.get_plugin("proc", NDProcessIO).nd_array_port.set("DEX1"),
        dexela1_roi1.nd_array_port.set("DEX1"),
    )
    await asyncio.gather(
        dexela1.get_plugin("hdf", NDFileHDF5IO).nd_array_port.set("PROC1"),
        dexela1_pva1.nd_array_port.set("ROI1"),
    )


def dexela1_set_plugin_chain() -> None:
    """Wire the dexela1 plugins as DEX1 -> PROC1 -> HDF1 and DEX1 -> ROI1 -> PVA1.

    HDF1 writes the (averaged) full frames; PVA1 serves the ROI for live viewing.
    Call this again after restarting the Dexela IOC. Call it from a notebook
    cell that has no ``await``.
    """
    call_in_bluesky_event_loop(_dexela1_set_plugin_chain())


try:
    dexela1_set_plugin_chain()
except Exception as _exc:
    rprint(f"[bold][red]Could not set the dexela1 plugin chain: {_exc!r}[/red][/bold]")


async def _evt1_set_plugin_chain() -> None:
    # Upstream first, so no plugin is ever fed from its own output
    await evt1_trans1.nd_array_port.set("EVT1")
    await evt1_roi1.nd_array_port.set("TRANS1")
    await evt1.get_plugin("proc", NDProcessIO).nd_array_port.set("ROI1")
    await evt1.get_plugin("hdf", NDFileHDF5IO).nd_array_port.set("PROC1")


def evt1_set_plugin_chain() -> None:
    """Wire the evt1 plugins as EVT1 -> TRANS1 -> ROI1 -> PROC1 -> HDF1.

    Call this again after restarting the Emergent Vision IOC. Call it from a
    notebook cell that has no ``await``.
    """
    call_in_bluesky_event_loop(_evt1_set_plugin_chain())


try:
    evt1_set_plugin_chain()
except Exception as _exc:
    rprint(f"[bold][red]Could not set the evt1 plugin chain: {_exc!r}[/red][/bold]")

##########
# MOTORS #
##########


with init_devices(mock=XPDTOOLS_RUNNING_IN_CI):
    # psylo = Psylotech("XF:28ID2-ES{Psylotech:1}", name="psylo")

    # The beamline names these motors in mixedCase, which the users rely on.
    # amPhi is a RotationMotor so it can be fly-scanned with the PandA encoder;
    # the others are plain AsyncEpicsMotors.

    mBaseX = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:Xbase}Mtr", name="amBaseX"
    )
    mBaseY = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:Y}Mtr", name="amBaseY"
    )
    amPhi = RotationMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:Phi2}Mtr", name="amPhi", panda_encoder=1
    )
    mTopX = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:Xfine}Mtr", name="amTopX"
    )
    mTopZ = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:Z}Mtr", name="amTopZ"
    )

    slits = Slits("XF:28IDD-ES:2{Slt-Ax:", name="slits")

    sample_stack = SampleStack(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Stack-Ax:", name="sample_stack"
    )

    mDetX = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Cam:Mnt-Ax:X}Mtr", name="mDetX"
    )

    mSigrayZ = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Sigray-Ax:Z}Mtr", name="mSigrayZ"
    )
    mVisibleX = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Sigray-Ax:X}Mtr", name="mVisibleX"
    )
    mEmergentY = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDD-ES:2{Stg:Sigray-Ax:Y}Mtr", name="mEmergentY"
    )

    mFastShutter = AsyncEpicsMotor(  # noqa: N816
        "XF:28IDC-ES:1{Sh2:Exp-Ax:5}Mtr", name="mFastShutter"
    )

    # mMirrorBendDS = AsyncEpicsMotor(  # noqa: N816
    #     "XF:28IDA-OP:1{Mir:VFM-Ax:BndD}Mtr", name="mMirrorBendDS"
    # )
    # mMirrorBendUS = AsyncEpicsMotor(  # noqa: N816
    #     "XF:28IDA-OP:1{Mir:VFM-Ax:BndU}Mtr", name="mMirrorBendUS"
    # )
    # mMirrorY = AsyncEpicsMotor(  # noqa: N816
    #     "XF:28IDA-OP:1{Mir:VFM-Ax:Y}Mtr", name="mMirrorY"
    # )

    # A few fake motors, useful for testing
    fake_motor_x = SimMotor(name="fake_motor_x", instant=False)
    fake_motor_y = SimMotor(name="fake_motor_y", instant=False)
    fake_motor_z = SimMotor(name="fake_z", instant=False)


    # Storage ring information
    storage_ring = NSLS2StorageRing()



# # Fast shutter
# mFastShutter = EpicsMotor("XF:28IDC-ES:1{Sh2:Exp-Ax:5}Mtr", name="mFastShutter")

# # Mirror
# mMirrorY = EpicsMotor("XF:28IDA-OP:1{Mir:VFM-Ax:Y}Mtr", name="mMirrorY")
# mMirrorBendDS = EpicsMotor("XF:28IDA-OP:1{Mir:VFM-Ax:BndD}Mtr", name="mMirrorBendDS")
# mMirrorBendUS = EpicsMotor("XF:28IDA-OP:1{Mir:VFM-Ax:BndU}Mtr", name="mMirrorBendUS")

###########
# FILTERS #
###########


class FilterBank(StandardReadable):
    """Four-slot filter bank."""

    def __init__(self, prefix: str, name: str = ""):
        with self.add_children_as_readables():
            self.flt1 = epics_signal_rw(str, f"{prefix}1-Cmd")
            self.flt2 = epics_signal_rw(str, f"{prefix}2-Cmd")
            self.flt3 = epics_signal_rw(str, f"{prefix}3-Cmd")
            self.flt4 = epics_signal_rw(str, f"{prefix}4-Cmd")
        super().__init__(name=name)


with init_devices(mock=XPDTOOLS_RUNNING_IN_CI):
    filters = FilterBank("XF:28IDC-OP:1{Fltr}Cmd:Opn", name="fb")  # noqa: N816
    storage_ring = NSLS2StorageRing()








############
# SHUTTERS #
############
with init_devices(mock=True):
    # The tomo plans only check that the front-end and photon shutters are open,
    # and open and close the fast shutter. Pass use_shutter=False to skip them.
    fe_shutter = Shutter("XF:28ID-PPS{Sh:FE}", name="fe_shutter")
    photon_shutter = Shutter("XF:28IDA-PPS{PSh}", name="photon_shutter")
    fast_shutter = FastShutter(name="fast_shutter")



############
# baseline #
############
# Configure baseline supplemental data to include in the metadata of every run.
sd = bpp.SupplementalData(
    baseline=[
        slits,
        sample_stack,
        filters,
        mDetX,
        mSigrayZ,
        mVisibleX,
        mEmergentY,
        mFastShutter,
        storage_ring.beam_current,
    ]
)
RE.preprocessors.append(sd)

bec.disable_baseline()

# Pause the RunEngine if the beam current drops below 100 mA and resume above
# 390 mA. Always installed; it ignores low beam while the ring is in maintenance
# or shutdown, checking the mode each time rather than once at startup.
RE.install_suspender(
    SuspendFloorUnlessOpsMode(
        storage_ring.beam_current,
        100,
        resume_thresh=390,
        sleep=600,
        ops_mode=storage_ring.operating_mode,
        skip_modes=(NSLS2OpsMode.MAINTENANCE,
                    NSLS2OpsMode.SHUTDOWN,
                    NSLS2OpsMode.ACCEL_STUDIES
                    ),
    )
)
