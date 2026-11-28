"""Additional motor classes and helper functions for XPD beamline at NSLS-II."""

import asyncio
from typing import Protocol

from bluesky.protocols import Flyable, Preparable
from ophyd_async.core import (
    DeviceMock,
    SignalR,
    StandardReadable,
    callback_on_mock_put,
    default_mock_class,
    derived_signal_r,
    derived_signal_rw,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


class FlyscanMotor(Flyable, Preparable, Protocol):
    """What :func:`xpd_tools.plans.single_axis_flyscan` needs of a motor.

    Any flyable motor with these members can be flown, e.g. :class:`RotationMotor`
    or the Psylotech rotation axis. The members are read-only properties so that
    a motor may provide a writable signal where a readable one is asked for.
    """

    @property
    def panda_encoder(self) -> int:
        """PandA encoder axis the motor's encoder is wired to."""
        ...

    @property
    def encoder_resolution(self) -> SignalR[float]:
        """Resolution of the encoder, in degrees per count."""
        ...

    @property
    def user_readback(self) -> SignalR[float]:
        """Position of the motor, in degrees."""
        ...

    @property
    def max_velocity(self) -> SignalR[float]:
        """Fastest the motor may be flown, in degrees per second."""
        ...

    @property
    def acceleration_time(self) -> SignalR[float]:
        """Time the motor takes to reach its velocity, in seconds."""
        ...


# TODO: Get this upstreamed to ophyd_async and remove it from here.
# It is a general utility that is not specific to XPD.
class VelocityRespectingMotorMock(DeviceMock[AsyncEpicsMotor]):
    """Mock behaviour that respects motor velocity and acceleration time."""

    async def connect(self, device: AsyncEpicsMotor) -> None:
        """Mock signals to simulate a move respecting velocity and acceleration."""
        set_mock_value(device.velocity, 10)
        set_mock_value(device.max_velocity, 100)
        set_mock_value(device.acceleration_time, 1)

        # Motor starts in "done" state (not moving)
        set_mock_value(device.motor_done_move, 1)

        async def _do_move(target: float):
            current = await device.user_readback.get_value()
            velocity = await device.velocity.get_value()
            acceleration_time = await device.acceleration_time.get_value()
            move_time = abs(target - current) / velocity + 2 * acceleration_time
            set_mock_value(device.motor_done_move, 0)
            elapsed = 0.0
            while elapsed < move_time:
                await asyncio.sleep(min(1.0, move_time - elapsed))
                elapsed += 1.0
                fraction = min(elapsed / move_time, 1.0)
                position = current + (target - current) * fraction
                set_mock_value(device.user_readback, position)
            set_mock_value(device.user_readback, target)
            set_mock_value(device.motor_done_move, 1)
            set_mock_put_proceeds(device.user_setpoint, True)

        def _on_setpoint_write(value):
            set_mock_put_proceeds(device.user_setpoint, False)
            asyncio.ensure_future(_do_move(value))

        callback_on_mock_put(device.user_setpoint, _on_setpoint_write)


@default_mock_class(VelocityRespectingMotorMock)
class RotationMotor(AsyncEpicsMotor):
    """A motor that can be used for rotation scans.

    This class is a subclass of the AsyncEpicsMotor class and is used to represent
    a motor that can be used for rotation scans. It has additional attributes and
    methods that are specific to rotation scans.

    Parameters
    ----------
    prefix : str
        PV prefix of the motor record.
    name : str, optional
        Name of the device.
    panda_encoder : int, default 1
        PandA encoder axis this motor's encoder is wired to. Flyscans use the
        INENC, CALC and PCOMP blocks with this number, and the shared PULSE1
        block to drive the capture.
    """

    def __init__(self, prefix: str, name: str = "", *, panda_encoder: int = 1):
        super().__init__(prefix, name=name)
        self.panda_encoder = panda_encoder
        self.encoder_counts_per_rev = derived_signal_r(
            self.get_encoder_counts_per_rev,
            derived_units="counts",
            derived_precision=0,
            encoder_resolution=self.encoder_resolution,
        )
        self.encoder_counts = epics_signal_r(int, prefix + ".REP")

    def get_encoder_counts_per_rev(self, encoder_resolution: float) -> int:
        """Calculate the number of encoder counts per revolution.

        Parameters
        ----------
        encoder_resolution : float
            The resolution of the encoder in degrees per count.

        Returns
        -------
        int
            The number of encoder counts per revolution.
        """
        return round(360.0 / encoder_resolution)


class Slits(StandardReadable):
    """Four-blade slits with derived gap and center signals."""

    def __init__(self, prefix: str, name: str = ""):
        with self.add_children_as_readables():
            self.top = AsyncEpicsMotor(f"{prefix}T}}Mtr")
            self.bottom = AsyncEpicsMotor(f"{prefix}B}}Mtr")
            self.outboard = AsyncEpicsMotor(f"{prefix}O}}Mtr")
            self.inboard = AsyncEpicsMotor(f"{prefix}I}}Mtr")

        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.v_gap = derived_signal_rw(
                self._get_v_gap,
                self._set_v_gap,
                top=self.top.user_readback,
                bottom=self.bottom.user_readback,
            )
            self.h_gap = derived_signal_rw(
                self._get_h_gap,
                self._set_h_gap,
                outboard=self.outboard.user_readback,
                inboard=self.inboard.user_readback,
            )

        with self.add_children_as_readables():
            self.v_center = derived_signal_rw(
                self._get_v_center,
                self._set_v_center,
                top=self.top.user_readback,
                bottom=self.bottom.user_readback,
            )
            self.h_center = derived_signal_rw(
                self._get_h_center,
                self._set_h_center,
                outboard=self.outboard.user_readback,
                inboard=self.inboard.user_readback,
            )

        super().__init__(name=name)

    def _get_v_gap(self, top: float, bottom: float) -> float:
        return top - bottom

    async def _set_v_gap(self, value: float) -> None:
        top, bottom = await asyncio.gather(
            self.top.user_readback.get_value(),
            self.bottom.user_readback.get_value(),
        )
        center = (top + bottom) / 2.0
        await asyncio.gather(
            self.top.set(center + value / 2.0),
            self.bottom.set(center - value / 2.0),
        )

    def _get_v_center(self, top: float, bottom: float) -> float:
        return (top + bottom) / 2.0

    async def _set_v_center(self, value: float) -> None:
        top, bottom = await asyncio.gather(
            self.top.user_readback.get_value(),
            self.bottom.user_readback.get_value(),
        )
        gap = top - bottom
        await asyncio.gather(
            self.top.set(value + gap / 2.0),
            self.bottom.set(value - gap / 2.0),
        )

    def _get_h_gap(self, outboard: float, inboard: float) -> float:
        return outboard - inboard

    async def _set_h_gap(self, value: float) -> None:
        outboard, inboard = await asyncio.gather(
            self.outboard.user_readback.get_value(),
            self.inboard.user_readback.get_value(),
        )
        center = (outboard + inboard) / 2.0
        await asyncio.gather(
            self.outboard.set(center + value / 2.0),
            self.inboard.set(center - value / 2.0),
        )

    def _get_h_center(self, outboard: float, inboard: float) -> float:
        return (outboard + inboard) / 2.0

    async def _set_h_center(self, value: float) -> None:
        outboard, inboard = await asyncio.gather(
            self.outboard.user_readback.get_value(),
            self.inboard.user_readback.get_value(),
        )
        gap = outboard - inboard
        await asyncio.gather(
            self.outboard.set(value + gap / 2.0),
            self.inboard.set(value - gap / 2.0),
        )


class SampleStack(StandardReadable):
    """Sample stack motors."""

    def __init__(self, prefix: str, name: str = ""):
        with self.add_children_as_readables():
            self.phi = AsyncEpicsMotor(f"{prefix}Phi}}Mtr")
            self.top_x = AsyncEpicsMotor(f"{prefix}Xfine}}Mtr")
            self.top_z = AsyncEpicsMotor(f"{prefix}Z}}Mtr")
            self.roll = AsyncEpicsMotor(f"{prefix}Roll}}Mtr")
            self.pitch = AsyncEpicsMotor(f"{prefix}Pitch}}Mtr")
            self.base_y = AsyncEpicsMotor(f"{prefix}Y}}Mtr")
            self.base_x = AsyncEpicsMotor(f"{prefix}Xbase}}Mtr")
        super().__init__(name=name)
