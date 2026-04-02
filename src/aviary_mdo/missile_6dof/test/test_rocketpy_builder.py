"""
Tests for the Missile6DOFBuilder Aviary SubsystemBuilder integration.

Validates that the builder follows the SubsystemBuilder pattern:
instantiation, build_pre_mission, design variables, and a full
problem setup-and-run cycle.

Note: The Missile6DOFBuilder must exist at
aviary_mdo.missile_6dof.missile_6dof_builder before these tests
will pass. They serve as the specification for that builder.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
)


class TestMissile6DOFBuilder(unittest.TestCase):
    """Test the Aviary SubsystemBuilder integration."""

    def test_builder_instantiation(self):
        """Builder should instantiate without error."""
        from aviary_mdo.missile_6dof.missile_6dof_builder import (
            Missile6DOFBuilder,
        )

        builder = Missile6DOFBuilder()
        self.assertEqual(builder.name, 'missile_6dof')

    def test_build_pre_mission(self):
        """Builder should return a valid OpenMDAO Group."""
        from aviary_mdo.missile_6dof.missile_6dof_builder import (
            Missile6DOFBuilder,
        )

        builder = Missile6DOFBuilder()
        pre_mission = builder.build_pre_mission(aviary_inputs=None)
        self.assertIsInstance(pre_mission, om.Group)

    def test_design_vars(self):
        """Builder should return design variables dict."""
        from aviary_mdo.missile_6dof.missile_6dof_builder import (
            Missile6DOFBuilder,
        )

        builder = Missile6DOFBuilder()
        dvs = builder.get_design_vars()
        self.assertIsInstance(dvs, dict)
        self.assertGreater(len(dvs), 0, "Should have at least one design variable")

    def test_full_problem_setup(self):
        """Full problem setup and run should work."""
        from aviary_mdo.missile_6dof.missile_6dof_builder import (
            Missile6DOFBuilder,
        )

        builder = Missile6DOFBuilder()

        prob = om.Problem()
        prob.model.add_subsystem('missile_6dof', builder.build_pre_mission(None))
        prob.setup()
        prob.run_model()

        apogee = prob.get_val(
            f'missile_6dof.{DynamicMissile6DOF.APOGEE}', units='m'
        )[0]
        self.assertGreater(apogee, 0, f"Apogee {apogee} should be positive")

    def test_builder_name_override(self):
        """Builder should accept a custom name."""
        from aviary_mdo.missile_6dof.missile_6dof_builder import (
            Missile6DOFBuilder,
        )

        builder = Missile6DOFBuilder(name='my_missile')
        self.assertEqual(builder.name, 'my_missile')


if __name__ == '__main__':
    unittest.main()
