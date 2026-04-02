"""
Tests for the RocketPy Barrowman aerodynamics component.

Validates that the RocketPyAero component produces reasonable aerodynamic
outputs (drag coefficients, center of pressure, static margins) using the
Barrowman method implemented in RocketPy.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)


class TestRocketPyAero(unittest.TestCase):
    """Test RocketPy Barrowman aerodynamics component."""

    def setUp(self):
        from aviary_mdo.missile_6dof.model.rocketpy_aero import (
            RocketPyAero,
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'aero', RocketPyAero(), promotes=['*']
        )
        self.prob.setup()

    def test_default_aero(self):
        """Default geometry should produce valid aero outputs."""
        self.prob.run_model()

        cp = self.prob.get_val(DynamicMissile6DOF.CP_POSITION, units='m')[0]
        sm = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]

        # CP should be somewhere along the body
        self.assertGreater(cp, 0, f"CP position {cp} should be positive")
        # With aft fins, should be stable
        self.assertGreater(
            sm, 0,
            f"Static margin {sm} should be positive for aft-finned missile",
        )

    def test_drag_coefficients_positive(self):
        """Both power-on and power-off drag should be positive."""
        self.prob.run_model()

        cd_on = self.prob.get_val(DynamicMissile6DOF.POWER_ON_CD)[0]
        cd_off = self.prob.get_val(DynamicMissile6DOF.POWER_OFF_CD)[0]

        self.assertGreater(cd_on, 0, "Power-on CD should be positive")
        self.assertGreater(cd_off, 0, "Power-off CD should be positive")

    def test_burnout_margin_differs(self):
        """Static margin at burnout should differ from ignition."""
        self.prob.run_model()

        sm_ign = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]
        sm_bo = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT
        )[0]

        # After propellant is consumed, CG shifts and margin changes
        self.assertNotEqual(
            sm_ign, sm_bo,
            "Static margin should change from ignition to burnout",
        )

    def test_fin_effect(self):
        """Larger fins should move CP aft (increase stability)."""
        # Small fins
        self.prob.set_val(Missile6DOF.Fins.SPAN, 0.05, units='m')
        self.prob.run_model()
        sm_small = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]

        # Big fins
        self.prob.set_val(Missile6DOF.Fins.SPAN, 0.25, units='m')
        self.prob.run_model()
        sm_big = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]

        self.assertGreater(
            sm_big, sm_small,
            f"Bigger fins should increase stability: {sm_big} vs {sm_small}",
        )


if __name__ == '__main__':
    unittest.main()
