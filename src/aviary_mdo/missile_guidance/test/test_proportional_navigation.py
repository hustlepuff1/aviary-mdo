"""
Tests for the ProNavGuidance component.

Verifies proportional navigation guidance law computations:
commanded acceleration, adjoint miss distance, and tau^2 scaling.
"""

import unittest
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import Guidance, DynamicGuidance


class TestProNavGuidance(unittest.TestCase):

    def setUp(self):
        from aviary_mdo.missile_guidance.model.proportional_navigation import ProNavGuidance
        self.prob = om.Problem()
        self.prob.model.add_subsystem('pn', ProNavGuidance(), promotes=['*'])
        self.prob.setup(force_alloc_complex=True)

    def test_zero_los_rate(self):
        """Zero LOS rate should give zero commanded acceleration (collision course)."""
        self.prob.set_val('los_rate', 0.0, units='rad/s')
        self.prob.set_val('closing_velocity', 1300.0, units='m/s')
        self.prob.set_val('nav_ratio', 4.0)
        self.prob.run_model()

        a_cmd = self.prob.get_val('commanded_accel', units='m/s**2')[0]
        self.assertAlmostEqual(a_cmd, 0.0, places=1)

    def test_pn_formula(self):
        """PN command should equal N * Vc * lambda_dot."""
        N = 4.0
        Vc = 1300.0
        lam_dot = 0.01
        self.prob.set_val('nav_ratio', N)
        self.prob.set_val('closing_velocity', Vc, units='m/s')
        self.prob.set_val('los_rate', lam_dot, units='rad/s')
        self.prob.run_model()

        a_cmd = self.prob.get_val('commanded_accel', units='m/s**2')[0]
        expected = N * Vc * lam_dot
        self.assertAlmostEqual(a_cmd, expected, places=0)

    def test_miss_scales_with_tau_squared(self):
        """Miss distance should roughly scale with tau^2."""
        # tau = 0.1
        self.prob.set_val('time_constant', 0.1, units='s')
        self.prob.set_val('target_accel', 50.0, units='m/s**2')
        self.prob.set_val('time_to_go', 10.0, units='s')
        self.prob.run_model()
        miss_01 = self.prob.get_val('adjoint_miss_distance', units='m')[0]

        # tau = 0.2 (double)
        self.prob.set_val('time_constant', 0.2, units='s')
        self.prob.run_model()
        miss_02 = self.prob.get_val('adjoint_miss_distance', units='m')[0]

        # Miss should roughly quadruple (tau^2 scaling)
        ratio = miss_02 / max(miss_01, 0.001)
        self.assertTrue(2.0 < ratio < 6.0,
                        f"Miss ratio {ratio} should be ~4x for tau^2 scaling")


if __name__ == '__main__':
    unittest.main()
