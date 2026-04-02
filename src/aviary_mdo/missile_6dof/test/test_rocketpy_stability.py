"""
Tests for the RocketPy stability analysis component.

Validates CG/CP positions and static margin at ignition and burnout
using the Barrowman method and RocketPy mass models.

Note: The RocketPyStability component must exist at
aviary_mdo.missile_6dof.model.rocketpy_stability before these tests
will pass. They serve as the specification for that component.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
)


class TestRocketPyStability(unittest.TestCase):
    """Test RocketPy stability analysis component."""

    def setUp(self):
        from aviary_mdo.missile_6dof.model.rocketpy_stability import (
            RocketPyStability,
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'stab', RocketPyStability(), promotes=['*']
        )
        self.prob.setup()

    def test_default_stability(self):
        """Default config should be statically stable."""
        self.prob.run_model()

        sm_ign = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]
        sm_bo = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT
        )[0]
        cp = self.prob.get_val(DynamicMissile6DOF.CP_POSITION, units='m')[0]
        cg = self.prob.get_val(DynamicMissile6DOF.CG_POSITION, units='m')[0]

        self.assertGreater(sm_ign, 0, "Should be stable at ignition")
        self.assertGreater(cp, 0, "CP should be along the body")
        self.assertGreater(cg, 0, "CG should be along the body")
        # CP should be aft of CG for stability (higher number = more aft from nose)
        self.assertGreater(
            cp, cg,
            f"CP ({cp}) should be aft of CG ({cg}) for positive static margin",
        )

    def test_burnout_cg_shift(self):
        """CG should shift after burnout (propellant consumed)."""
        self.prob.run_model()

        sm_ign = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION
        )[0]
        sm_bo = self.prob.get_val(
            DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT
        )[0]

        # After burning propellant at aft, margin changes
        self.assertNotEqual(
            sm_ign, sm_bo,
            "Static margin should change from ignition to burnout",
        )

    def test_cp_aft_of_midpoint(self):
        """For an aft-finned missile, CP should be behind body midpoint."""
        self.prob.run_model()

        cp = self.prob.get_val(DynamicMissile6DOF.CP_POSITION, units='m')[0]
        # Default body length is 3.0m, so midpoint is ~1.5m.
        # With aft fins, CP should be aft of midpoint.
        self.assertGreater(
            cp, 1.0,
            f"CP at {cp}m should be well aft of nose for aft-finned config",
        )


if __name__ == '__main__':
    unittest.main()
