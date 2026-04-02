"""Tests for TPS sizing components."""

import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary_mdo.aerothermodynamics.model.tps_sizing import (
    AerothermoGroup,
    RadiativeEquilibrium,
    TPSMass,
    TPSThickness,
    TPS_MATERIALS,
    STEFAN_BOLTZMANN,
)


class TestRadiativeEquilibrium(unittest.TestCase):
    """Tests for the RadiativeEquilibrium component."""

    def _run_radiative_eq(self, q_stag, emissivity=0.85):
        prob = om.Problem()
        prob.model.add_subsystem('comp', RadiativeEquilibrium(), promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val('q_stag', q_stag, units='W/m**2')
        prob.set_val('emissivity', emissivity)
        prob.run_model()
        return prob

    def test_radiative_eq_temperature_reasonable(self):
        """Radiative equilibrium temp should be in 1000-4000 K for typical re-entry."""
        prob = self._run_radiative_eq(q_stag=5.0e6)
        T_wall = prob.get_val('T_wall_eq', units='K')[0]
        self.assertGreater(T_wall, 1000.0)
        self.assertLess(T_wall, 4000.0)

    def test_higher_q_gives_higher_T(self):
        """Higher heat flux must produce higher equilibrium temperature."""
        prob_low = self._run_radiative_eq(q_stag=0.5e6)
        prob_high = self._run_radiative_eq(q_stag=5.0e6)
        T_low = prob_low.get_val('T_wall_eq', units='K')[0]
        T_high = prob_high.get_val('T_wall_eq', units='K')[0]
        self.assertGreater(T_high, T_low)

    def test_radiative_eq_known_value(self):
        """Check against hand-computed value."""
        q = 5.0e6
        eps = 0.85
        expected = (q / (eps * STEFAN_BOLTZMANN)) ** 0.25
        prob = self._run_radiative_eq(q_stag=q, emissivity=eps)
        T_wall = prob.get_val('T_wall_eq', units='K')[0]
        assert_near_equal(T_wall, expected, tolerance=1e-10)

    def test_supersonic_missile_temperature(self):
        """For supersonic missile (q=0.5 MW/m^2), T_wall should be 1000-2000 K."""
        prob = self._run_radiative_eq(q_stag=0.5e6)
        T_wall = prob.get_val('T_wall_eq', units='K')[0]
        self.assertGreater(T_wall, 1000.0)
        self.assertLess(T_wall, 2000.0)

    def test_radiative_eq_partials(self):
        """Complex-step partials check."""
        prob = self._run_radiative_eq(q_stag=5.0e6)
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestTPSThickness(unittest.TestCase):
    """Tests for the TPSThickness component."""

    def _run_tps_thickness(self, tps_type, q_peak=5.0e6,
                           total_heat_load=1.5e9, soak_time=2000.0):
        prob = om.Problem()
        prob.model.add_subsystem('comp', TPSThickness(tps_type=tps_type),
                                 promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val('q_peak', q_peak, units='W/m**2')
        prob.set_val('total_heat_load_per_area', total_heat_load, units='J/m**2')
        prob.set_val('soak_time', soak_time, units='s')
        prob.run_model()
        return prob

    def test_ablative_thickness_positive_and_reasonable(self):
        """Ablative TPS thickness should be positive and on the order of cm."""
        # q=5 MW/m^2 for 300s gives Q_total = 1.5e9 J/m^2
        prob = self._run_tps_thickness('ablative', total_heat_load=1.5e9)
        thickness = prob.get_val('thickness', units='m')[0]
        self.assertGreater(thickness, 0.0)
        # Should be on order of cm (0.001 to 1 m)
        self.assertGreater(thickness, 0.001)
        self.assertLess(thickness, 1.0)

    def test_ceramic_thickness_positive(self):
        """Ceramic TPS thickness should be positive."""
        prob = self._run_tps_thickness('ceramic')
        thickness = prob.get_val('thickness', units='m')[0]
        self.assertGreater(thickness, 0.0)

    def test_metallic_thickness_positive(self):
        """Metallic TPS thickness should be positive."""
        prob = self._run_tps_thickness('metallic')
        thickness = prob.get_val('thickness', units='m')[0]
        self.assertGreater(thickness, 0.0)

    def test_higher_heat_load_gives_thicker_ablative(self):
        """More total heat load must produce thicker ablative TPS."""
        prob_low = self._run_tps_thickness('ablative', total_heat_load=1.0e9)
        prob_high = self._run_tps_thickness('ablative', total_heat_load=3.0e9)
        t_low = prob_low.get_val('thickness', units='m')[0]
        t_high = prob_high.get_val('thickness', units='m')[0]
        self.assertGreater(t_high, t_low)

    def test_longer_soak_gives_thicker_ceramic(self):
        """Longer soak time must produce thicker insulative TPS."""
        prob_low = self._run_tps_thickness('ceramic', soak_time=1000.0)
        prob_high = self._run_tps_thickness('ceramic', soak_time=3000.0)
        t_low = prob_low.get_val('thickness', units='m')[0]
        t_high = prob_high.get_val('thickness', units='m')[0]
        self.assertGreater(t_high, t_low)

    def test_ablative_mass_less_than_metallic(self):
        """Ablative should be lighter per unit area than metallic for same conditions."""
        prob_abl = self._run_tps_thickness('ablative', total_heat_load=1.5e9)
        prob_met = self._run_tps_thickness('metallic', soak_time=2000.0)
        m_abl = prob_abl.get_val('tps_mass_per_area', units='kg/m**2')[0]
        m_met = prob_met.get_val('tps_mass_per_area', units='kg/m**2')[0]
        self.assertLess(m_abl, m_met)

    def test_ablative_partials(self):
        """Complex-step partials check for ablative TPS."""
        prob = self._run_tps_thickness('ablative')
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_ceramic_partials(self):
        """Complex-step partials check for ceramic TPS."""
        prob = self._run_tps_thickness('ceramic')
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestTPSMass(unittest.TestCase):
    """Tests for the TPSMass component."""

    def test_mass_scales_with_wetted_area(self):
        """TPS mass should scale linearly with wetted area."""
        mass_per_area = 50.0  # kg/m^2

        prob1 = om.Problem()
        prob1.model.add_subsystem('comp', TPSMass(), promotes=['*'])
        prob1.setup(force_alloc_complex=True)
        prob1.set_val('tps_mass_per_area', mass_per_area, units='kg/m**2')
        prob1.set_val('wetted_area', 10.0, units='m**2')
        prob1.run_model()
        mass_10 = prob1.get_val('mass_tps', units='kg')[0]

        prob2 = om.Problem()
        prob2.model.add_subsystem('comp', TPSMass(), promotes=['*'])
        prob2.setup(force_alloc_complex=True)
        prob2.set_val('tps_mass_per_area', mass_per_area, units='kg/m**2')
        prob2.set_val('wetted_area', 20.0, units='m**2')
        prob2.run_model()
        mass_20 = prob2.get_val('mass_tps', units='kg')[0]

        assert_near_equal(mass_20 / mass_10, 2.0, tolerance=1e-10)

    def test_tps_mass_partials(self):
        """Complex-step partials check for TPSMass."""
        prob = om.Problem()
        prob.model.add_subsystem('comp', TPSMass(), promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val('tps_mass_per_area', 50.0, units='kg/m**2')
        prob.set_val('wetted_area', 10.0, units='m**2')
        prob.run_model()

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestAerothermoGroup(unittest.TestCase):
    """Tests for the assembled AerothermoGroup."""

    def test_reentry_capsule_ablative(self):
        """Re-entry capsule: q=5 MW/m^2, A=10 m^2, 300s flight time.

        Expected: ablative thickness few cm, mass 100-500 kg, T_wall 2000-3000 K.
        """
        prob = om.Problem()
        prob.model.add_subsystem('aero', AerothermoGroup(tps_type='ablative'),
                                 promotes=['*'])
        prob.setup(force_alloc_complex=True)

        q_stag = 5.0e6       # W/m^2
        flight_time = 300.0   # s
        Q_total = q_stag * flight_time  # 1.5e9 J/m^2

        prob.set_val('q_stag', q_stag, units='W/m**2')
        prob.set_val('q_peak', q_stag, units='W/m**2')
        prob.set_val('total_heat_load_per_area', Q_total, units='J/m**2')
        prob.set_val('wetted_area', 10.0, units='m**2')
        prob.run_model()

        T_wall = prob.get_val('T_wall_eq', units='K')[0]
        thickness = prob.get_val('thickness', units='m')[0]
        mass = prob.get_val('mass_tps', units='kg')[0]

        # Radiative eq temp should be 2000-4000 K
        self.assertGreater(T_wall, 2000.0)
        self.assertLess(T_wall, 4000.0)

        # Thickness should be a few cm
        self.assertGreater(thickness, 0.01)
        self.assertLess(thickness, 0.5)

        # Mass should be on the order of hundreds to ~1500 kg for 10 m^2 at 5 MW/m^2
        self.assertGreater(mass, 100.0)
        self.assertLess(mass, 1500.0)

    def test_group_partials(self):
        """Complex-step partials check for the assembled group."""
        prob = om.Problem()
        prob.model.add_subsystem('aero', AerothermoGroup(tps_type='ablative'),
                                 promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val('q_stag', 5.0e6, units='W/m**2')
        prob.set_val('q_peak', 5.0e6, units='W/m**2')
        prob.set_val('total_heat_load_per_area', 1.5e9, units='J/m**2')
        prob.set_val('wetted_area', 10.0, units='m**2')
        prob.run_model()

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
