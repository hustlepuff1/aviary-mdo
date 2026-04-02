"""
Tests for the GuidanceBuilder SubsystemBuilder.

Verifies builder instantiation, pre-mission build, design variable
registration, and a full end-to-end run through the engagement sim.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import Guidance, DynamicGuidance


class TestGuidanceBuilder(unittest.TestCase):

    def test_instantiation(self):
        from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder
        builder = GuidanceBuilder()
        self.assertEqual(builder.name, 'guidance')

    def test_build_pre_mission(self):
        from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder
        builder = GuidanceBuilder()
        pre = builder.build_pre_mission(None)
        self.assertIsInstance(pre, om.Group)

    def test_design_vars(self):
        from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder
        builder = GuidanceBuilder()
        dvs = builder.get_design_vars()
        self.assertTrue(len(dvs) > 0)

    def test_full_run(self):
        from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder
        builder = GuidanceBuilder()
        prob = om.Problem()
        prob.model.add_subsystem('guidance', builder.build_pre_mission(None))
        prob.setup()
        prob.run_model()

        miss = prob.get_val(f'guidance.{DynamicGuidance.MISS_DISTANCE}', units='m')[0]
        self.assertTrue(miss >= 0, "Miss distance should be non-negative")


if __name__ == '__main__':
    unittest.main()
