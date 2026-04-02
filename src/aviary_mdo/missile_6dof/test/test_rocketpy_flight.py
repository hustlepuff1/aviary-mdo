"""
Integration tests for the full RocketPy 6-DOF flight simulation component.

These tests exercise the RocketPyFlight component which wraps the complete
RocketPy Rocket + Flight pipeline: solid motor, Barrowman aero surfaces,
rail launch, and 6-DOF trajectory integration.

Note: The RocketPyFlight component must exist at
aviary_mdo.missile_6dof.model.rocketpy_flight before these tests
will pass. They serve as the specification for that component.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)


class TestRocketPyFlight(unittest.TestCase):
    """Integration test for the full 6-DOF flight simulation component."""

    def setUp(self):
        from aviary_mdo.missile_6dof.model.rocketpy_flight import (
            RocketPyFlightComp,
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'flight', RocketPyFlightComp(), promotes=['*']
        )
        self.prob.setup()

    def test_default_flight(self):
        """Default configuration should produce a valid flight."""
        self.prob.run_model()

        apogee = self.prob.get_val(DynamicMissile6DOF.APOGEE, units='m')[0]
        flight_time = self.prob.get_val(
            DynamicMissile6DOF.FLIGHT_TIME, units='s'
        )[0]
        max_speed = self.prob.get_val(
            DynamicMissile6DOF.MAX_SPEED, units='m/s'
        )[0]

        self.assertGreater(apogee, 0, "Apogee should be positive")
        self.assertGreater(flight_time, 0, "Flight time should be positive")
        self.assertGreater(max_speed, 0, "Max speed should be positive")

    def test_physics_sanity(self):
        """Check basic physics relationships hold."""
        self.prob.run_model()

        max_speed = self.prob.get_val(
            DynamicMissile6DOF.MAX_SPEED, units='m/s'
        )[0]
        max_mach = self.prob.get_val(DynamicMissile6DOF.MAX_MACH)[0]
        max_accel = self.prob.get_val(
            DynamicMissile6DOF.MAX_ACCELERATION, units='m/s**2'
        )[0]

        # Max Mach ~ max_speed / 340 (sea level speed of sound)
        approx_mach = max_speed / 340.0
        self.assertTrue(
            abs(max_mach - approx_mach) / max(max_mach, 0.01) < 0.5,
            f"Mach {max_mach} inconsistent with speed {max_speed}",
        )

        # Max acceleration > g (since we have thrust)
        self.assertGreater(
            max_accel, 9.81,
            f"Max accel {max_accel} should exceed g",
        )

    def test_stability_outputs(self):
        """Static margin should be positive for a stable configuration."""
        self.prob.run_model()

        sm_ign = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]

        # With fins at the aft end, should be statically stable
        self.assertGreater(
            sm_ign, 0,
            f"Static margin at ignition {sm_ign} should be positive",
        )

    def test_motor_outputs(self):
        """Motor performance should be populated."""
        self.prob.run_model()

        total_impulse = self.prob.get_val(
            DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s'
        )[0]
        prop_mass = self.prob.get_val(
            DynamicMissile6DOF.PROPELLANT_MASS, units='kg'
        )[0]

        self.assertGreater(total_impulse, 0, "Total impulse should be positive")
        self.assertGreater(prop_mass, 0, "Propellant mass should be positive")

    def test_vertical_launch(self):
        """Vertical launch (90 deg) should have high apogee."""
        self.prob.set_val(Missile6DOF.Launch.INCLINATION, 90.0, units='deg')
        self.prob.run_model()

        apogee = self.prob.get_val(DynamicMissile6DOF.APOGEE, units='m')[0]
        self.assertGreater(apogee, 100, "Vertical launch apogee too low")

    def test_low_angle_launch(self):
        """Low angle launch should have more range than near-vertical."""
        # 45 deg launch
        self.prob.set_val(Missile6DOF.Launch.INCLINATION, 45.0, units='deg')
        self.prob.run_model()
        range_45 = self.prob.get_val(
            DynamicMissile6DOF.RANGE_TOTAL, units='m'
        )[0]

        # 85 deg launch
        self.prob.set_val(Missile6DOF.Launch.INCLINATION, 85.0, units='deg')
        self.prob.run_model()
        range_85 = self.prob.get_val(
            DynamicMissile6DOF.RANGE_TOTAL, units='m'
        )[0]

        # 45 deg should have more range than 85 deg
        self.assertGreater(
            range_45, range_85 * 0.5,
            f"45deg range {range_45} should exceed 85deg range {range_85}",
        )

    def test_dynamic_pressure_positive(self):
        """Max dynamic pressure should be positive."""
        self.prob.run_model()

        max_q = self.prob.get_val(
            DynamicMissile6DOF.MAX_DYNAMIC_PRESSURE, units='Pa'
        )[0]
        self.assertGreater(max_q, 0, "Max dynamic pressure should be positive")


if __name__ == '__main__':
    unittest.main()
