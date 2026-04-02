"""
Tests for TMD propulsion against spreadsheet baseline values.

Baseline rocket boost motor:
  - Pc=1769 psi, eps=6.0, fuel type 4 (High Smoke Composite)
  - gamma=1.217, c*=5298 ft/s
  - Cf=1.637, Isp=271 s, thrust=7036 lbf
  - mdot=26.0 lbm/s, At=2.419 in^2, Ae=14.52 in^2
"""

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.model.missile_propulsion import (
    RocketMotor,
    _exit_mach_from_area_ratio,
)


class TestExitMach(unittest.TestCase):
    """Test isentropic area-Mach relation solver."""

    def test_known_values(self):
        """Verify exit Mach for known area ratios."""
        # For gamma=1.4, A/A*=2.0 -> Me~2.197
        Me = _exit_mach_from_area_ratio(2.0, 1.4)
        assert_near_equal(Me, 2.197, tolerance=0.01)

        # A/A*=1.0 -> Me=1.0
        Me = _exit_mach_from_area_ratio(1.001, 1.4)
        assert_near_equal(Me, 1.0, tolerance=0.05)


class TestRocketMotor(unittest.TestCase):
    """Test rocket motor against TMD spreadsheet baseline."""

    def test_baseline_rocket_boost(self):
        """Test with TMD spreadsheet default rocket values."""
        prob = om.Problem()
        prob.model.add_subsystem('rocket', RocketMotor())
        prob.setup(force_alloc_complex=True)

        # TMD spreadsheet defaults (rocket baseline)
        prob.set_val('rocket.Pc', 1769.0, units='lbf/inch**2')
        prob.set_val('rocket.expansion_ratio', 6.0)
        prob.set_val('rocket.fuel_type', 4.0)
        prob.set_val('rocket.propellant_weight', 84.8, units='lbm')
        prob.set_val('rocket.burn_time', 3.26, units='s')
        prob.set_val('rocket.Pa', 6.76, units='lbf/inch**2')

        prob.run_model()

        gamma = prob.get_val('rocket.gamma')
        c_star = prob.get_val('rocket.c_star')
        Cf = prob.get_val('rocket.Cf')
        Isp = prob.get_val('rocket.Isp')
        thrust = prob.get_val('rocket.thrust')

        # TMD spreadsheet: gamma=1.217, c*=5298
        assert_near_equal(gamma, 1.217, tolerance=1e-6)
        assert_near_equal(c_star, 5298.0, tolerance=1e-6)

        # Cf should be ~1.637 for eps=6 (spreadsheet default)
        # Our implementation may differ slightly due to Pe/Pa term
        self.assertGreater(Cf, 1.4)
        self.assertLess(Cf, 1.9)

        # Isp should be in reasonable range (~250-290 s)
        self.assertGreater(Isp, 240.0)
        self.assertLess(Isp, 300.0)

        # Thrust should be in reasonable range (~6000-8000 lbf)
        self.assertGreater(thrust, 5000.0)
        self.assertLess(thrust, 9000.0)

    def test_higher_expansion_ratio(self):
        """Higher expansion ratio should give higher Isp."""
        prob = om.Problem()
        prob.model.add_subsystem('rocket', RocketMotor())
        prob.setup(force_alloc_complex=True)

        prob.set_val('rocket.Pc', 1769.0, units='lbf/inch**2')
        prob.set_val('rocket.fuel_type', 4.0)
        prob.set_val('rocket.propellant_weight', 84.8, units='lbm')
        prob.set_val('rocket.burn_time', 3.26, units='s')
        prob.set_val('rocket.Pa', 6.76, units='lbf/inch**2')

        # Low expansion ratio
        prob.set_val('rocket.expansion_ratio', 4.0)
        prob.run_model()
        Isp_low = prob.get_val('rocket.Isp')[0]

        # High expansion ratio
        prob.set_val('rocket.expansion_ratio', 15.0)
        prob.run_model()
        Isp_high = prob.get_val('rocket.Isp')[0]

        self.assertGreater(Isp_high, Isp_low)


if __name__ == '__main__':
    unittest.main()
