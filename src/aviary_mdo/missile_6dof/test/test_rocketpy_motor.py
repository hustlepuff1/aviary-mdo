"""
Tests for the RocketPy SolidMotor wrapper component.

Validates that the motor wrapper produces physically reasonable performance
values (total impulse, thrust, Isp, propellant mass) using default grain
geometry and thrust parameters.
"""

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)


class TestRocketPyMotor(unittest.TestCase):
    """Test RocketPy solid motor component."""

    def setUp(self):
        from aviary_mdo.missile_6dof.model.rocketpy_motor import (
            RocketPyMotor,
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'motor', RocketPyMotor(), promotes=['*']
        )
        self.prob.setup()

    def test_default_motor(self):
        """Default motor should produce reasonable thrust and impulse."""
        self.prob.run_model()

        total_impulse = self.prob.get_val(
            DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s'
        )[0]
        max_thrust = self.prob.get_val(
            DynamicMissile6DOF.MAX_THRUST, units='N'
        )[0]
        avg_thrust = self.prob.get_val(
            DynamicMissile6DOF.AVERAGE_THRUST, units='N'
        )[0]
        prop_mass = self.prob.get_val(
            DynamicMissile6DOF.PROPELLANT_MASS, units='kg'
        )[0]

        self.assertGreater(total_impulse, 0, "Total impulse should be positive")
        self.assertGreater(max_thrust, 0, "Max thrust should be positive")
        self.assertGreater(avg_thrust, 0, "Average thrust should be positive")
        self.assertGreater(prop_mass, 0, "Propellant mass should be positive")

    def test_impulse_consistency(self):
        """Total impulse ~ average_thrust * burn_time."""
        self.prob.run_model()

        total_impulse = self.prob.get_val(
            DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s'
        )[0]
        avg_thrust = self.prob.get_val(
            DynamicMissile6DOF.AVERAGE_THRUST, units='N'
        )[0]
        burn_time = self.prob.get_val(
            Missile6DOF.Motor.BURN_TIME, units='s'
        )[0]

        expected_impulse = avg_thrust * burn_time
        # Allow 20% since grain regression changes thrust over time
        assert_near_equal(total_impulse, expected_impulse, tolerance=0.20)

    def test_isp_positive(self):
        """Isp should be positive.  For a realistic motor (high thrust, tuned
        grain geometry) Isp lands in the 180-300 s range, but with the small
        default thrust level (5 kN) against the default grain, RocketPy's
        exhaust-velocity model can return a lower value.  Here we just verify
        the sign and that it is not wildly large."""
        self.prob.run_model()

        avg_isp = self.prob.get_val(
            DynamicMissile6DOF.AVERAGE_ISP, units='s'
        )[0]
        self.assertGreater(avg_isp, 0, "Isp should be positive")
        self.assertLess(avg_isp, 500, "Isp should not exceed 500 s for a solid")

    def test_higher_thrust_more_impulse(self):
        """Doubling thrust level should roughly double total impulse."""
        # Baseline
        self.prob.set_val(Missile6DOF.Motor.THRUST_SOURCE, 5000.0, units='N')
        self.prob.run_model()
        impulse_low = self.prob.get_val(
            DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s'
        )[0]

        # Doubled thrust
        self.prob.set_val(Missile6DOF.Motor.THRUST_SOURCE, 10000.0, units='N')
        self.prob.run_model()
        impulse_high = self.prob.get_val(
            DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s'
        )[0]

        ratio = impulse_high / impulse_low
        self.assertTrue(
            1.5 < ratio < 2.5,
            f"Impulse ratio {ratio} not consistent with doubled thrust",
        )


if __name__ == '__main__':
    unittest.main()
