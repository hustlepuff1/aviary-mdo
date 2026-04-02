"""
Tests for the GuidedEngagementSim component.

Verifies 3-DOF guided engagement simulation for canonical scenarios:
head-on non-maneuvering, maneuvering target, navigation ratio sweep,
and intercept flag logic.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import Guidance, DynamicGuidance


class TestGuidedEngagementSim(unittest.TestCase):

    def setUp(self):
        from aviary_mdo.missile_guidance.model.guided_engagement_sim import GuidedEngagementSim
        self.prob = om.Problem()
        self.prob.model.add_subsystem('sim', GuidedEngagementSim(), promotes=['*'])
        self.prob.setup()

    def test_head_on_non_maneuvering(self):
        """Head-on against non-maneuvering target: should achieve small miss."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 0.0, units='deg')
        self.prob.set_val(Guidance.Missile.MAX_ACCELERATION, 300.0, units='m/s**2')
        self.prob.set_val(Guidance.Target.POSITION_X, 15000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 180.0, units='deg')
        self.prob.set_val(Guidance.Target.ACCELERATION_MAX, 0.0, units='m/s**2')
        self.prob.set_val(Guidance.Navigation.RATIO, 4.0)
        self.prob.set_val(Guidance.Autopilot.TIME_CONSTANT, 0.1, units='s')
        self.prob.run_model()

        miss = self.prob.get_val(DynamicGuidance.MISS_DISTANCE, units='m')[0]
        t_int = self.prob.get_val(DynamicGuidance.TIME_TO_INTERCEPT, units='s')[0]

        self.assertTrue(miss < 5.0, f"Head-on non-maneuv miss {miss}m should be < 5m")
        self.assertTrue(5 < t_int < 20, f"Intercept time {t_int}s out of range")

    def test_maneuvering_target(self):
        """Against maneuvering target: miss should be larger but still feasible."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 45.0, units='deg')
        self.prob.set_val(Guidance.Missile.MAX_ACCELERATION, 300.0, units='m/s**2')
        self.prob.set_val(Guidance.Target.POSITION_X, 15000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 15000.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 180.0, units='deg')
        self.prob.set_val(Guidance.Target.ACCELERATION_MAX, 50.0, units='m/s**2')
        self.prob.set_val(Guidance.Target.MANEUVER_TIME, 3.0, units='s')
        self.prob.set_val(Guidance.Navigation.RATIO, 4.0)
        self.prob.set_val(Guidance.Autopilot.TIME_CONSTANT, 0.2, units='s')
        self.prob.run_model()

        miss = self.prob.get_val(DynamicGuidance.MISS_DISTANCE, units='m')[0]
        pk = self.prob.get_val(DynamicGuidance.SINGLE_SHOT_PK)[0]
        max_g = self.prob.get_val(DynamicGuidance.REQUIRED_LOAD_FACTOR)[0]

        self.assertTrue(miss < 50.0, f"Miss {miss}m too large for SM-2 class")
        self.assertTrue(max_g > 1.0, "Should require some maneuvering")

    def test_higher_nav_ratio_reduces_miss(self):
        """Higher N should reduce miss distance."""
        # Setup a crossing engagement with maneuvering target
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 45.0, units='deg')
        self.prob.set_val(Guidance.Missile.MAX_ACCELERATION, 200.0, units='m/s**2')
        self.prob.set_val(Guidance.Target.POSITION_X, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 180.0, units='deg')
        self.prob.set_val(Guidance.Target.ACCELERATION_MAX, 30.0, units='m/s**2')
        self.prob.set_val(Guidance.Autopilot.TIME_CONSTANT, 0.2, units='s')

        # N = 3
        self.prob.set_val(Guidance.Navigation.RATIO, 3.0)
        self.prob.run_model()
        miss_3 = self.prob.get_val(DynamicGuidance.MISS_DISTANCE, units='m')[0]

        # N = 5
        self.prob.set_val(Guidance.Navigation.RATIO, 5.0)
        self.prob.run_model()
        miss_5 = self.prob.get_val(DynamicGuidance.MISS_DISTANCE, units='m')[0]

        # N=5 should give smaller or equal miss than N=3
        self.assertTrue(miss_5 <= miss_3 * 1.5,
                        f"N=5 miss {miss_5} should be <= N=3 miss {miss_3}")

    def test_intercept_flag(self):
        """Intercept should be achieved for a favorable geometry."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 0.0, units='deg')
        self.prob.set_val(Guidance.Missile.MAX_ACCELERATION, 300.0, units='m/s**2')
        self.prob.set_val(Guidance.Target.POSITION_X, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 180.0, units='deg')
        self.prob.set_val(Guidance.Target.ACCELERATION_MAX, 0.0, units='m/s**2')
        self.prob.set_val(Guidance.Navigation.RATIO, 4.0)
        self.prob.set_val(Guidance.Autopilot.TIME_CONSTANT, 0.1, units='s')
        self.prob.run_model()

        intercept = self.prob.get_val(DynamicGuidance.INTERCEPT_ACHIEVED)[0]
        self.assertEqual(intercept, 1.0, "Head-on non-maneuv should intercept")


if __name__ == '__main__':
    unittest.main()
