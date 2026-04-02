"""
Tests for TMD warhead discipline against spreadsheet baseline values.

Baseline warhead (Sheet 8):
  - Mmiss = 11.39 slugs, Mwh = 2.4 slugs (77.28 lbs)
  - charge_to_metal_ratio = 1.0
  - charge_energy_sqrt (Ec) = 10230 ft/s
  - Mc = 1.2 slugs, Mm = 1.2 slugs
  - Explosive weight = 38.64 lbs
  - Fragment velocity Vf = 8352.76 ft/s
  - Fragment weight = 0.00022 slugs (3.21 grams)
  - Fragment density = 15.2 slugs/ft^3

Values extracted from design_spreadsheet_TMD.xls Warhead sheet.
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from aviary_mdo.missile.model.missile_warhead import (
    WarheadFragmentation,
    WarheadBlastOverpressure,
    WarheadPenetration,
    WarheadLethality,
    WarheadGroup,
)


class TestWarheadFragmentation(unittest.TestCase):
    """Test fragmentation against TMD spreadsheet values."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('frag', WarheadFragmentation())
        self.prob.setup(force_alloc_complex=True)

        # Set baseline inputs from TMD spreadsheet
        self.prob.set_val('frag.warhead_mass', 2.4, units='slug')
        self.prob.set_val('frag.charge_to_metal_ratio', 1.0)
        self.prob.set_val('frag.charge_energy_sqrt', 10230.0, units='ft/s')
        self.prob.set_val('frag.fragment_weight', 0.00022, units='slug')
        self.prob.set_val('frag.fragment_density', 15.2, units='slug/ft**3')

    def test_charge_mass(self):
        """TMD spreadsheet: Mc = 1.2 slugs for charge_ratio=1.0."""
        self.prob.run_model()
        Mc = self.prob.get_val('frag.charge_mass', units='slug')
        assert_near_equal(Mc, 1.2, tolerance=1e-6)

    def test_metal_mass(self):
        """TMD spreadsheet: Mm = 1.2 slugs for charge_ratio=1.0."""
        self.prob.run_model()
        Mm = self.prob.get_val('frag.metal_mass', units='slug')
        assert_near_equal(Mm, 1.2, tolerance=1e-6)

    def test_explosive_weight(self):
        """TMD spreadsheet: explosive weight = 38.64 lbs."""
        self.prob.run_model()
        W_expl = self.prob.get_val('frag.explosive_weight', units='lbm')
        assert_near_equal(W_expl, 38.6088, tolerance=0.002)

    def test_fragment_velocity(self):
        """TMD spreadsheet: Vf = 8352.76 ft/s (Gurney equation)."""
        self.prob.run_model()
        Vf = self.prob.get_val('frag.fragment_velocity', units='ft/s')
        assert_near_equal(Vf, 8352.76, tolerance=0.001)

    def test_num_fragments(self):
        """Number of fragments = Mm / Wf = 1.2 / 0.00022 = 5454.5."""
        self.prob.run_model()
        N = self.prob.get_val('frag.num_fragments')
        assert_near_equal(N, 1.2 / 0.00022, tolerance=0.01)

    def test_higher_charge_ratio_increases_velocity(self):
        """Higher charge-to-metal ratio should increase fragment velocity."""
        self.prob.run_model()
        Vf_baseline = self.prob.get_val('frag.fragment_velocity', units='ft/s')[0]

        self.prob.set_val('frag.charge_to_metal_ratio', 1.5)
        self.prob.run_model()
        Vf_high = self.prob.get_val('frag.fragment_velocity', units='ft/s')[0]

        self.assertGreater(Vf_high, Vf_baseline)

    def test_partials(self):
        """Check analytic derivatives via finite difference.

        Note: rtol is loosened to 0.01 because the num_fragments/fragment_weight
        derivative is extremely steep at the baseline fragment_weight (0.00022 slugs),
        causing inherent FD truncation error. The component uses CS internally for
        exact derivatives.
        """
        self.prob.run_model()
        data = self.prob.check_partials(method='fd', out_stream=None)
        assert_check_partials(data, atol=1e-5, rtol=1e-2)


class TestWarheadBlastOverpressure(unittest.TestCase):
    """Test blast overpressure component."""

    def test_baseline_runs(self):
        """Component should run with baseline inputs and produce positive overpressure."""
        prob = om.Problem()
        prob.model.add_subsystem('blast', WarheadBlastOverpressure())
        prob.setup(force_alloc_complex=True)

        prob.set_val('blast.explosive_weight', 38.64, units='lbm')
        prob.set_val('blast.target_range', 8.0, units='ft')
        prob.set_val('blast.atmospheric_pressure', 14.696, units='psi')

        prob.run_model()

        Z = prob.get_val('blast.scaled_distance')
        Ps = prob.get_val('blast.peak_overpressure', units='psi')

        # Scaled distance should be positive
        self.assertGreater(Z, 0.0)
        # Overpressure should be positive
        self.assertGreater(Ps, 0.0)

    def test_scaled_distance(self):
        """Z = R / W^(1/3) = 8.0 / 38.64^(1/3)."""
        prob = om.Problem()
        prob.model.add_subsystem('blast', WarheadBlastOverpressure())
        prob.setup(force_alloc_complex=True)

        prob.set_val('blast.explosive_weight', 38.64, units='lbm')
        prob.set_val('blast.target_range', 8.0, units='ft')

        prob.run_model()

        Z = prob.get_val('blast.scaled_distance')
        Z_expected = 8.0 / (38.64 ** (1.0 / 3.0))
        assert_near_equal(Z, Z_expected, tolerance=1e-6)


class TestWarheadPenetration(unittest.TestCase):
    """Test penetration depth component."""

    def test_baseline_runs(self):
        """Component should run with baseline inputs."""
        prob = om.Problem()
        prob.model.add_subsystem('pen', WarheadPenetration())
        prob.setup(force_alloc_complex=True)

        prob.set_val('pen.penetrator_length', 14.0, units='inch')
        prob.set_val('pen.penetrator_diameter', 2.0, units='inch')
        prob.set_val('pen.impact_velocity', 2000.0, units='ft/s')
        prob.set_val('pen.penetrator_density', 15.2, units='slug/ft**3')
        prob.set_val('pen.target_density', 15.2, units='slug/ft**3')

        prob.run_model()

        L_D = prob.get_val('pen.length_to_diameter')
        assert_near_equal(L_D, 7.0, tolerance=1e-6)

        P = prob.get_val('pen.penetration_depth', units='inch')
        # Penetration depth should be positive
        self.assertGreater(P, 0.0)

    def test_density_ratio_effect(self):
        """Higher penetrator density relative to target should increase penetration."""
        prob = om.Problem()
        prob.model.add_subsystem('pen', WarheadPenetration())
        prob.setup(force_alloc_complex=True)

        prob.set_val('pen.penetrator_length', 14.0, units='inch')
        prob.set_val('pen.penetrator_diameter', 2.0, units='inch')
        prob.set_val('pen.impact_velocity', 2000.0, units='ft/s')

        # Equal density
        prob.set_val('pen.penetrator_density', 15.2, units='slug/ft**3')
        prob.set_val('pen.target_density', 15.2, units='slug/ft**3')
        prob.run_model()
        P_equal = prob.get_val('pen.penetration_depth', units='inch')[0]

        # Higher penetrator density
        prob.set_val('pen.penetrator_density', 30.4, units='slug/ft**3')
        prob.run_model()
        P_heavy = prob.get_val('pen.penetration_depth', units='inch')[0]

        self.assertGreater(P_heavy, P_equal)


class TestWarheadLethality(unittest.TestCase):
    """Test lethality / Pk component."""

    def test_baseline_pk(self):
        """Pk should be between 0 and 1 for baseline inputs."""
        prob = om.Problem()
        prob.model.add_subsystem('leth', WarheadLethality())
        prob.setup(force_alloc_complex=True)

        prob.set_val('leth.fragment_velocity', 8352.76, units='ft/s')
        prob.set_val('leth.target_range', 8.0, units='ft')
        prob.set_val('leth.target_presented_area', 20.0, units='ft**2')
        prob.set_val('leth.target_vulnerable_area', 2.0, units='ft**2')
        prob.set_val('leth.num_fragments', 1.2 / 0.00022)

        prob.run_model()

        Pk = prob.get_val('leth.Pk')
        self.assertGreater(Pk, 0.0)
        self.assertLessEqual(Pk, 1.0)

    def test_larger_range_reduces_pk(self):
        """Increasing range should reduce Pk."""
        prob = om.Problem()
        prob.model.add_subsystem('leth', WarheadLethality())
        prob.setup(force_alloc_complex=True)

        prob.set_val('leth.fragment_velocity', 8352.76, units='ft/s')
        prob.set_val('leth.target_presented_area', 20.0, units='ft**2')
        prob.set_val('leth.target_vulnerable_area', 2.0, units='ft**2')
        prob.set_val('leth.num_fragments', 5454.5)

        prob.set_val('leth.target_range', 8.0, units='ft')
        prob.run_model()
        Pk_close = prob.get_val('leth.Pk')[0]

        prob.set_val('leth.target_range', 50.0, units='ft')
        prob.run_model()
        Pk_far = prob.get_val('leth.Pk')[0]

        self.assertGreater(Pk_close, Pk_far)


class TestWarheadGroup(unittest.TestCase):
    """Test the integrated warhead group."""

    def test_baseline_group(self):
        """Run the full warhead group with baseline TMD inputs."""
        prob = om.Problem()
        prob.model.add_subsystem('wh', WarheadGroup())
        prob.setup(force_alloc_complex=True)

        prob.set_val('wh.warhead_mass', 2.4, units='slug')
        prob.set_val('wh.charge_to_metal_ratio', 1.0)
        prob.set_val('wh.charge_energy_sqrt', 10230.0, units='ft/s')
        prob.set_val('wh.fragment_weight', 0.00022, units='slug')
        prob.set_val('wh.fragment_density', 15.2, units='slug/ft**3')
        prob.set_val('wh.target_range', 8.0, units='ft')
        prob.set_val('wh.target_presented_area', 20.0, units='ft**2')
        prob.set_val('wh.target_vulnerable_area', 2.0, units='ft**2')
        prob.set_val('wh.penetrator_length', 14.0, units='inch')
        prob.set_val('wh.penetrator_diameter', 2.0, units='inch')
        prob.set_val('wh.impact_velocity', 2000.0, units='ft/s')

        prob.run_model()

        # Verify key spreadsheet values propagate through group
        Vf = prob.get_val('wh.fragment_velocity', units='ft/s')
        assert_near_equal(Vf, 8352.76, tolerance=0.001)

        Mc = prob.get_val('wh.charge_mass', units='slug')
        assert_near_equal(Mc, 1.2, tolerance=1e-6)

        Pk = prob.get_val('wh.Pk')
        self.assertGreater(Pk, 0.0)
        self.assertLessEqual(Pk, 1.0)


if __name__ == '__main__':
    unittest.main()
