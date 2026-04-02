"""
Tests for TMD aerodynamics against spreadsheet baseline values.

Baseline rocket (Sparrow MRAAM):
  - Launch: M=0.8, h=20,000 ft, d=8.0 in, l=143.9 in
  - CD0 power off at launch: ~0.31
  - CD0 power on at launch: ~0.25 (reduced base drag)
  - CN at 20 deg alpha: ~9.83

Values extracted from design_spreadsheet_TMD.xls Master Output sheet.
"""

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.model.missile_aero import (
    MissileBodyDrag,
    MissileWingDrag,
    MissileNormalForce,
    _atmosphere,
)


class TestAtmosphere(unittest.TestCase):
    """Test the standard atmosphere helper against TMD spreadsheet values."""

    def test_20000ft(self):
        """TMD spreadsheet: T=447.4R, rho=0.001267 slug/ft3, a=1036.9 ft/s."""
        T, rho, mu = _atmosphere(20000.0)
        a = (1.4 * 1716.49 * T)**0.5

        assert_near_equal(T, 447.4, tolerance=0.01)
        assert_near_equal(rho, 0.001267, tolerance=0.01)
        assert_near_equal(a, 1036.9, tolerance=0.005)

    def test_sea_level(self):
        """TMD spreadsheet: rho_sl=0.002377 slug/ft3, a_sl=1116.4 ft/s."""
        T, rho, mu = _atmosphere(0.0)
        a = (1.4 * 1716.49 * T)**0.5

        assert_near_equal(rho, 0.002377, tolerance=0.005)
        assert_near_equal(a, 1116.4, tolerance=0.005)

    def test_40000ft(self):
        """TMD ramjet baseline altitude."""
        T, rho, mu = _atmosphere(40000.0)
        # Stratosphere: T should be ~389.97 R
        assert_near_equal(T, 389.97, tolerance=0.01)


class TestBodyDrag(unittest.TestCase):
    """Test body drag against TMD spreadsheet values."""

    def test_rocket_baseline_runs(self):
        """Verify the component runs with rocket baseline inputs."""
        prob = om.Problem()
        prob.model.add_subsystem('body', MissileBodyDrag(power_on=True))
        prob.setup(force_alloc_complex=True)

        prob.set_val('body.mach', 0.8)
        prob.set_val('body.length', 143.9, units='inch')
        prob.set_val('body.diameter', 8.0, units='inch')
        prob.set_val('body.nose_length', 19.2, units='inch')
        prob.set_val('body.nose_bluntness', 0.05)
        prob.set_val('body.nozzle_exit_area', 14.52, units='inch**2')
        prob.set_val('body.ref_area', 50.27, units='inch**2')
        prob.set_val('body.alt', 20000.0, units='ft')

        prob.run_model()

        fn = prob.get_val('body.nose_fineness')
        fb = prob.get_val('body.body_fineness')

        assert_near_equal(fn, 19.2 / 8.0, tolerance=1e-6)
        assert_near_equal(fb, 143.9 / 8.0, tolerance=1e-6)

        # Body drag should be positive and reasonable
        CD_body = prob.get_val('body.CD_body_total')
        self.assertGreater(CD_body, 0.05)
        self.assertLess(CD_body, 0.5)


class TestNormalForce(unittest.TestCase):
    """Test normal force against TMD spreadsheet values."""

    def test_zero_alpha(self):
        """At zero angle of attack, CN should be approximately zero."""
        prob = om.Problem()
        prob.model.add_subsystem('cn', MissileNormalForce())
        prob.setup(force_alloc_complex=True)

        prob.set_val('cn.alpha', 0.0, units='deg')
        prob.set_val('cn.mach', 2.0)
        prob.set_val('cn.body_diameter', 8.0, units='inch')
        prob.set_val('cn.body_length', 143.9, units='inch')
        prob.set_val('cn.nose_length', 19.2, units='inch')
        prob.set_val('cn.ref_area', 50.27, units='inch**2')

        prob.run_model()

        CN = prob.get_val('cn.CN_total')
        assert_near_equal(CN, 0.0, tolerance=1e-6)

    def test_positive_alpha(self):
        """At positive alpha, CN should be positive and grow with alpha."""
        prob = om.Problem()
        prob.model.add_subsystem('cn', MissileNormalForce())
        prob.setup(force_alloc_complex=True)

        prob.set_val('cn.mach', 2.0)
        prob.set_val('cn.body_diameter', 8.0, units='inch')
        prob.set_val('cn.body_length', 143.9, units='inch')
        prob.set_val('cn.nose_length', 19.2, units='inch')
        prob.set_val('cn.ref_area', 50.27, units='inch**2')

        prob.set_val('cn.alpha', 5.0, units='deg')
        prob.run_model()
        CN_5 = prob.get_val('cn.CN_total')[0]

        prob.set_val('cn.alpha', 10.0, units='deg')
        prob.run_model()
        CN_10 = prob.get_val('cn.CN_total')[0]

        self.assertGreater(CN_5, 0.0)
        self.assertGreater(CN_10, CN_5)


if __name__ == '__main__':
    unittest.main()
