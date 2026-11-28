import pytest
from ophyd_async.core import set_mock_value

from xpd_tools.motors import RotationMotor, SampleStack, Slits


@pytest.mark.parametrize(
    "encoder_resolution, expected",
    [
        (0.0009, 400_000),
        (0.01, 36_000),
    ],
)
def test_get_encoder_counts_per_rev(encoder_resolution: float, expected: int):
    motor = RotationMotor("MOT:")
    assert motor.get_encoder_counts_per_rev(encoder_resolution) == expected


def test_panda_encoder_defaults_to_axis_1():
    assert RotationMotor("MOT:").panda_encoder == 1
    assert RotationMotor("MOT:", panda_encoder=3).panda_encoder == 3


async def test_slits_gap_and_center():
    slits = Slits("XF:28IDD-ES:2{Slt-Ax:", name="slits")
    await slits.connect(mock=True)

    set_mock_value(slits.top.user_readback, 1.5)
    set_mock_value(slits.bottom.user_readback, -0.5)
    set_mock_value(slits.outboard.user_readback, 2.0)
    set_mock_value(slits.inboard.user_readback, -1.0)

    assert await slits.v_gap.get_value() == pytest.approx(2.0)
    assert await slits.v_center.get_value() == pytest.approx(0.5)
    assert await slits.h_gap.get_value() == pytest.approx(3.0)
    assert await slits.h_center.get_value() == pytest.approx(0.5)

    await slits.v_gap.set(1.0)
    assert await slits.top.user_setpoint.get_value() == pytest.approx(1.0)
    assert await slits.bottom.user_setpoint.get_value() == pytest.approx(0.0)

    await slits.h_center.set(0.0)
    assert await slits.outboard.user_setpoint.get_value() == pytest.approx(1.5)
    assert await slits.inboard.user_setpoint.get_value() == pytest.approx(-1.5)


async def test_sample_stack_axes():
    stack = SampleStack("XF:28IDD-ES:2{Stg:Stack-Ax:", name="sample_stack")
    await stack.connect(mock=True)

    for axis in (
        stack.phi,
        stack.top_x,
        stack.top_z,
        stack.roll,
        stack.pitch,
        stack.base_y,
        stack.base_x,
    ):
        await axis.set(1.25)
        assert await axis.user_setpoint.get_value() == pytest.approx(1.25)
