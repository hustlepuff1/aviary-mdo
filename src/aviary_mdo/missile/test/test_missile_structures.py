"""
Tests for TMD structures against spreadsheet baseline values.

Baseline rocket (Sparrow MRAAM):
  - d=8.0 in, l=143.9 in, W_launch=500 lbm
  - MEOP = 2373.6 psi (R19)
  - Estimated weight = 391.28 lbm (R2)
  - Iy_cylinder = 102.6 slug-ft^2 (R8)
  - Iy_nose = 0.061 slug-ft^2 (R12)
  - CG at launch = 76.2 in (R5)
  - Boost fuel length = 20.32 in (R13)
  - Boost fuel CG = 91.48 in (R14)

Values extracted from design_spreadsheet_TMD.xls Structure sheet.
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.model.missile_structures import (
    MotorCaseStress,
    SkinTemperature,
    MissileWeightEstimation,
    MissileStructuresGroup,
)


class TestMotorCaseStress(unittest.TestCase):
    """Test motor case MEOP and stress against TMD spreadsheet."""

    def test_meop_baseline(self):
        """TMD spreadsheet R19: MEOP = 2373.6 psi."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)

        prob.set_val('case.chamber_pressure', 2151.95, units='psi')
        prob.set_val('case.e_multiplier', 1.103)

        prob.run_model()

        MEOP = prob.get_val('case.MEOP', units='psi')
        assert_near_equal(MEOP, 2373.6, tolerance=0.001)

    def test_meop_is_pressure_times_multiplier(self):
        """MEOP should equal Pc * e_multiplier for any inputs."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)

        Pc = 3000.0
        e_mult = 1.2
        prob.set_val('case.chamber_pressure', Pc, units='psi')
        prob.set_val('case.e_multiplier', e_mult)

        prob.run_model()

        MEOP = prob.get_val('case.MEOP', units='psi')
        assert_near_equal(MEOP, Pc * e_mult, tolerance=1e-10)

    def test_wall_thickness_positive(self):
        """Wall thickness must be positive for any positive pressure."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        t = prob.get_val('case.wall_thickness', units='inch')
        self.assertGreater(t, 0.0)

    def test_case_weight_positive(self):
        """Motor case weight must be positive."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        w = prob.get_val('case.case_weight', units='lbm')
        self.assertGreater(w, 0.0)

    def test_higher_pressure_thicker_wall(self):
        """Higher chamber pressure should require thicker wall."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)

        prob.set_val('case.chamber_pressure', 1000.0, units='psi')
        prob.run_model()
        t_low = prob.get_val('case.wall_thickness', units='inch')[0]

        prob.set_val('case.chamber_pressure', 3000.0, units='psi')
        prob.run_model()
        t_high = prob.get_val('case.wall_thickness', units='inch')[0]

        self.assertGreater(t_high, t_low)

    def test_partials(self):
        """Complex-step derivative check."""
        prob = om.Problem()
        prob.model.add_subsystem('case', MotorCaseStress())
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
        for comp_name, comp_data in data.items():
            for (out_name, in_name), deriv_data in comp_data.items():
                if deriv_data['magnitude'].forward > 1e-15:
                    assert_near_equal(
                        deriv_data['rel error'].forward, 0.0, tolerance=1e-6)


class TestSkinTemperature(unittest.TestCase):
    """Test skin temperature recovery calculation."""

    def test_recovery_temp_m2_20kft(self):
        """At M=2.0, h=20000 ft, recovery temp should be significantly above freestream."""
        prob = om.Problem()
        prob.model.add_subsystem('skin', SkinTemperature())
        prob.setup(force_alloc_complex=True)

        prob.set_val('skin.mach', 2.0)
        prob.set_val('skin.alt', 20000.0, units='ft')
        prob.set_val('skin.recovery_factor', 0.9)

        prob.run_model()

        T_free = prob.get_val('skin.T_freestream', units='degR')[0]
        T_rec = prob.get_val('skin.T_recovery', units='degR')[0]
        T_rec_F = prob.get_val('skin.T_recovery_F', units='degF')[0]

        # T_freestream at 20kft ~ 447.4 R
        assert_near_equal(T_free, 447.4, tolerance=0.01)

        # T_recovery = T_atm * (1 + 0.9 * 0.2 * 4) = T_atm * 1.72
        T_expected = T_free * (1.0 + 0.9 * 0.2 * 4.0)
        assert_near_equal(T_rec, T_expected, tolerance=1e-10)

        # Should be well above freestream
        self.assertGreater(T_rec, T_free * 1.5)

        # Fahrenheit conversion check
        assert_near_equal(T_rec_F, T_rec - 459.67, tolerance=1e-10)

        # Recovery temp should be reasonable (~770 R ~ 310 F)
        self.assertGreater(T_rec_F, 200.0)
        self.assertLess(T_rec_F, 500.0)

    def test_higher_mach_higher_temp(self):
        """Higher Mach number should give higher recovery temperature."""
        prob = om.Problem()
        prob.model.add_subsystem('skin', SkinTemperature())
        prob.setup(force_alloc_complex=True)

        prob.set_val('skin.alt', 20000.0, units='ft')
        prob.set_val('skin.recovery_factor', 0.9)

        prob.set_val('skin.mach', 1.5)
        prob.run_model()
        T_low = prob.get_val('skin.T_recovery', units='degR')[0]

        prob.set_val('skin.mach', 3.0)
        prob.run_model()
        T_high = prob.get_val('skin.T_recovery', units='degR')[0]

        self.assertGreater(T_high, T_low)

    def test_zero_mach(self):
        """At M=0, recovery temp should equal freestream."""
        prob = om.Problem()
        prob.model.add_subsystem('skin', SkinTemperature())
        prob.setup(force_alloc_complex=True)

        prob.set_val('skin.mach', 0.0)
        prob.set_val('skin.alt', 20000.0, units='ft')

        prob.run_model()

        T_free = prob.get_val('skin.T_freestream', units='degR')
        T_rec = prob.get_val('skin.T_recovery', units='degR')
        assert_near_equal(T_rec, T_free, tolerance=1e-10)

    def test_partials(self):
        """Complex-step derivative check."""
        prob = om.Problem()
        prob.model.add_subsystem('skin', SkinTemperature())
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
        for comp_name, comp_data in data.items():
            for (out_name, in_name), deriv_data in comp_data.items():
                if deriv_data['magnitude'].forward > 1e-15:
                    assert_near_equal(
                        deriv_data['rel error'].forward, 0.0, tolerance=1e-6)


class TestMissileWeightEstimation(unittest.TestCase):
    """Test weight estimation, CG, and Iy against TMD spreadsheet."""

    def _make_prob(self):
        prob = om.Problem()
        prob.model.add_subsystem('wt', MissileWeightEstimation())
        prob.setup(force_alloc_complex=True)
        return prob

    def test_weight_regression(self):
        """TMD spreadsheet R2: estimated weight = 391.28 lbm for baseline."""
        prob = self._make_prob()

        prob.set_val('wt.missile_diameter', 8.0, units='inch')
        prob.set_val('wt.missile_length', 143.9, units='inch')

        prob.run_model()

        W_est = prob.get_val('wt.W_estimated', units='lbm')
        assert_near_equal(W_est, 391.28, tolerance=0.001)

    def test_boost_fuel_length(self):
        """TMD spreadsheet R13: boost fuel length = 20.32 in."""
        prob = self._make_prob()

        prob.set_val('wt.boost_fuel_weight', 112.0, units='lbm')
        prob.set_val('wt.fuel_density', 0.065, units='lbm/inch**3')
        prob.set_val('wt.fuel_outer_radius', 5.195, units='inch')

        prob.run_model()

        L_boost = prob.get_val('wt.boost_fuel_length', units='inch')
        assert_near_equal(L_boost, 20.32, tolerance=0.005)

    def test_boost_fuel_cg(self):
        """TMD spreadsheet R14: boost fuel CG = 91.48 in."""
        prob = self._make_prob()

        prob.set_val('wt.boost_fuel_weight', 112.0, units='lbm')
        prob.set_val('wt.fuel_density', 0.065, units='lbm/inch**3')
        prob.set_val('wt.fuel_outer_radius', 5.195, units='inch')
        prob.set_val('wt.motor_start_station', 81.32, units='inch')

        prob.run_model()

        cg_boost = prob.get_val('wt.boost_fuel_cg', units='inch')
        assert_near_equal(cg_boost, 91.48, tolerance=0.005)

    def test_iy_cylinder(self):
        """TMD spreadsheet R8: Iy_cylinder = 102.6 slug-ft^2."""
        prob = self._make_prob()

        prob.set_val('wt.missile_diameter', 8.0, units='inch')
        prob.set_val('wt.missile_length', 143.9, units='inch')
        prob.set_val('wt.nose_length', 19.2, units='inch')
        prob.set_val('wt.W_body', 365.7, units='lbm')

        prob.run_model()

        Iy_cyl = prob.get_val('wt.Iy_cylinder', units='slug*ft**2')
        assert_near_equal(Iy_cyl, 102.6, tolerance=0.005)

    def test_iy_nose(self):
        """TMD spreadsheet R12: Iy_nose = 0.061 slug-ft^2."""
        prob = self._make_prob()

        prob.set_val('wt.missile_diameter', 8.0, units='inch')
        prob.set_val('wt.nose_length', 19.2, units='inch')
        prob.set_val('wt.W_nose', 17.42, units='lbm')

        prob.run_model()

        Iy_nose = prob.get_val('wt.Iy_nose', units='slug*ft**2')
        assert_near_equal(Iy_nose, 0.061, tolerance=0.02)

    def test_cg_launch_reasonable(self):
        """CG at launch should be in the forward half of the missile."""
        prob = self._make_prob()
        prob.run_model()

        cg = prob.get_val('wt.cg_launch', units='inch')[0]
        L = prob.get_val('wt.missile_length', units='inch')[0]

        # CG should be between 40% and 60% of length
        self.assertGreater(cg, 0.4 * L)
        self.assertLess(cg, 0.6 * L)

    def test_cg_end_boost_forward_of_launch(self):
        """CG at end boost should shift forward (boost fuel is aft of CG)."""
        prob = self._make_prob()
        prob.run_model()

        cg_launch = prob.get_val('wt.cg_launch', units='inch')[0]
        cg_eb = prob.get_val('wt.cg_end_boost', units='inch')[0]

        # Removing aft fuel should shift CG forward
        self.assertLess(cg_eb, cg_launch)

    def test_weight_regression_scales_with_diameter(self):
        """Larger diameter should give heavier weight estimate."""
        prob = self._make_prob()

        prob.set_val('wt.missile_diameter', 6.0, units='inch')
        prob.run_model()
        W_small = prob.get_val('wt.W_estimated', units='lbm')[0]

        prob.set_val('wt.missile_diameter', 10.0, units='inch')
        prob.run_model()
        W_large = prob.get_val('wt.W_estimated', units='lbm')[0]

        self.assertGreater(W_large, W_small)

    def test_partials(self):
        """Complex-step derivative check."""
        prob = self._make_prob()
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
        for comp_name, comp_data in data.items():
            for (out_name, in_name), deriv_data in comp_data.items():
                if deriv_data['magnitude'].forward > 1e-15:
                    assert_near_equal(
                        deriv_data['rel error'].forward, 0.0, tolerance=1e-6)


class TestMissileStructuresGroup(unittest.TestCase):
    """Integration test for the structures group."""

    def test_group_runs(self):
        """Verify the full structures group runs without error."""
        prob = om.Problem()
        prob.model.add_subsystem('struct', MissileStructuresGroup())
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        MEOP = prob.get_val('struct.MEOP', units='psi')
        T_rec = prob.get_val('struct.T_recovery', units='degR')
        W_est = prob.get_val('struct.W_estimated', units='lbm')

        self.assertGreater(MEOP, 0.0)
        self.assertGreater(T_rec, 0.0)
        self.assertGreater(W_est, 0.0)


if __name__ == '__main__':
    unittest.main()
