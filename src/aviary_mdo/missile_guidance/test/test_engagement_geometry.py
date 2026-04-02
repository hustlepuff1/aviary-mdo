"""
Tests for the EngagementGeometry component.

Verifies planar engagement geometry computations (range, closing
velocity, LOS angle, LOS rate) for canonical engagement scenarios:
head-on, tail chase, and crossing.
"""

import unittest
import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import Guidance


class TestEngagementGeometry(unittest.TestCase):

    def setUp(self):
        from aviary_mdo.missile_guidance.model.engagement_geometry import EngagementGeometry
        self.prob = om.Problem()
        self.prob.model.add_subsystem('eng', EngagementGeometry(), promotes=['*'])
        self.prob.setup(force_alloc_complex=True)

    def test_head_on(self):
        """Head-on engagement: missile at origin heading East, target at (10000,0) heading West."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 0.0, units='deg')
        self.prob.set_val(Guidance.Target.POSITION_X, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 180.0, units='deg')
        self.prob.run_model()

        R = self.prob.get_val(Guidance.Engagement.INITIAL_RANGE, units='m')[0]
        Vc = self.prob.get_val(Guidance.Engagement.CLOSING_VELOCITY, units='m/s')[0]

        self.assertAlmostEqual(R, 10000.0, places=0)
        self.assertAlmostEqual(Vc, 1300.0, places=0)  # Vm + Vt head-on

    def test_tail_chase(self):
        """Tail chase: both heading East, missile faster."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 0.0, units='deg')
        self.prob.set_val(Guidance.Target.POSITION_X, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 0.0, units='deg')
        self.prob.run_model()

        Vc = self.prob.get_val(Guidance.Engagement.CLOSING_VELOCITY, units='m/s')[0]
        self.assertAlmostEqual(Vc, 700.0, places=0)  # Vm - Vt

    def test_crossing(self):
        """Crossing engagement: missile heading East, target heading North."""
        self.prob.set_val(Guidance.Missile.POSITION_X, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Missile.VELOCITY, 1000.0, units='m/s')
        self.prob.set_val(Guidance.Missile.HEADING, 0.0, units='deg')
        self.prob.set_val(Guidance.Target.POSITION_X, 10000.0, units='m')
        self.prob.set_val(Guidance.Target.POSITION_Y, 0.0, units='m')
        self.prob.set_val(Guidance.Target.VELOCITY, 300.0, units='m/s')
        self.prob.set_val(Guidance.Target.HEADING, 90.0, units='deg')
        self.prob.run_model()

        R = self.prob.get_val(Guidance.Engagement.INITIAL_RANGE, units='m')[0]
        los_rate = self.prob.get_val(Guidance.Engagement.LOS_RATE, units='rad/s')[0]

        self.assertAlmostEqual(R, 10000.0, places=0)
        # Target moving North should create non-zero LOS rate
        self.assertNotAlmostEqual(los_rate, 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
