"""
Tests for stagnation point heating models.

Tests cover Fay-Riddell and Sutton-Graves correlations against
expected physical behavior and cross-validation between models.
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.aerothermodynamics.model.stagnation_heating import (
    FayRiddellHeating,
    SuttonGravesHeating,
    _atmosphere_si,
)


class TestFayRiddellHeating(unittest.TestCase):
    """Tests for the Fay-Riddell stagnation heating component."""

    def _make_problem(self):
        prob = om.Problem()
        prob.model.add_subsystem('fr', FayRiddellHeating())
        prob.setup(force_alloc_complex=True)
        return prob

    def test_positive_heat_flux_supersonic(self):
        """Supersonic re-entry condition should produce positive heat flux."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 7000.0, units='m/s')
        prob.set_val('fr.altitude', 50000.0, units='m')
        prob.set_val('fr.nose_radius', 1.0, units='m')
        prob.set_val('fr.wall_temperature', 300.0, units='K')
        prob.run_model()

        q = prob.get_val('fr.q_stag', units='W/m**2')[0]
        self.assertGreater(q, 0.0, 'Heat flux must be positive for supersonic flow')
        # Re-entry at 7 km/s should be in MW/m^2 range
        self.assertGreater(q, 1.0e5, 'Heat flux should be significant at 7 km/s')

    def test_stagnation_temperature_above_freestream(self):
        """Stagnation temperature must exceed freestream temperature."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 3000.0, units='m/s')
        prob.set_val('fr.altitude', 30000.0, units='m')
        prob.run_model()

        T0 = prob.get_val('fr.stagnation_temperature', units='K')[0]
        # At 30 km, T_inf ~ 227 K; stagnation should be much higher
        self.assertGreater(T0, 300.0)

    def test_stagnation_pressure_above_freestream(self):
        """Stagnation pressure must exceed freestream static pressure."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 1000.0, units='m/s')
        prob.set_val('fr.altitude', 0.0, units='m')
        prob.run_model()

        p0 = prob.get_val('fr.stagnation_pressure', units='Pa')[0]
        self.assertGreater(p0, 101325.0, 'Stagnation pressure > freestream')

    def test_zero_velocity_zero_heat_flux(self):
        """Zero velocity must produce zero heat flux."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 0.0, units='m/s')
        prob.set_val('fr.altitude', 0.0, units='m')
        prob.set_val('fr.nose_radius', 0.1, units='m')
        prob.run_model()

        q = prob.get_val('fr.q_stag', units='W/m**2')[0]
        assert_near_equal(q, 0.0, tolerance=1e-10)

    def test_inverse_sqrt_radius_dependence(self):
        """Doubling nose radius should reduce heat flux by factor sqrt(2)."""
        prob = self._make_problem()
        V = 5000.0
        alt = 40000.0

        prob.set_val('fr.velocity', V, units='m/s')
        prob.set_val('fr.altitude', alt, units='m')
        prob.set_val('fr.wall_temperature', 300.0, units='K')

        prob.set_val('fr.nose_radius', 1.0, units='m')
        prob.run_model()
        q1 = prob.get_val('fr.q_stag', units='W/m**2')[0]

        prob.set_val('fr.nose_radius', 2.0, units='m')
        prob.run_model()
        q2 = prob.get_val('fr.q_stag', units='W/m**2')[0]

        ratio = q1 / q2
        assert_near_equal(ratio, np.sqrt(2.0), tolerance=1e-6)

    def test_velocity_cubed_dependence(self):
        """Heat flux should scale approximately as V^3."""
        prob = self._make_problem()
        alt = 40000.0
        prob.set_val('fr.altitude', alt, units='m')
        prob.set_val('fr.nose_radius', 1.0, units='m')
        prob.set_val('fr.wall_temperature', 300.0, units='K')

        prob.set_val('fr.velocity', 2000.0, units='m/s')
        prob.run_model()
        q1 = prob.get_val('fr.q_stag', units='W/m**2')[0]

        prob.set_val('fr.velocity', 4000.0, units='m/s')
        prob.run_model()
        q2 = prob.get_val('fr.q_stag', units='W/m**2')[0]

        # q ~ V^3 * (1 - h_w/h_s), so the ratio is not exactly 8
        # because h_s changes with velocity, but should be close
        ratio = q2 / q1
        # For V^3 pure scaling it would be 8.0; enthalpy correction
        # makes it slightly different but should be in 6-10 range
        self.assertGreater(ratio, 5.0,
                           f'Expected ratio > 5 for V^3 scaling, got {ratio}')
        self.assertLess(ratio, 12.0,
                        f'Expected ratio < 12 for V^3 scaling, got {ratio}')

    def test_higher_altitude_lower_heat_flux(self):
        """Higher altitude (lower density) should give lower heat flux."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 3000.0, units='m/s')
        prob.set_val('fr.nose_radius', 0.5, units='m')
        prob.set_val('fr.wall_temperature', 300.0, units='K')

        prob.set_val('fr.altitude', 10000.0, units='m')
        prob.run_model()
        q_low = prob.get_val('fr.q_stag', units='W/m**2')[0]

        prob.set_val('fr.altitude', 50000.0, units='m')
        prob.run_model()
        q_high = prob.get_val('fr.q_stag', units='W/m**2')[0]

        self.assertGreater(q_low, q_high,
                           'Lower altitude (higher density) should give more heating')

    def test_partials(self):
        """Complex-step derivative check for Fay-Riddell."""
        prob = self._make_problem()
        prob.set_val('fr.velocity', 3000.0, units='m/s')
        prob.set_val('fr.altitude', 30000.0, units='m')
        prob.set_val('fr.nose_radius', 0.5, units='m')
        prob.set_val('fr.wall_temperature', 500.0, units='K')
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True,
                                   out_stream=None)
        for comp_name, comp_data in data.items():
            for (out_name, in_name), deriv_data in comp_data.items():
                if deriv_data['magnitude'].forward > 1e-15:
                    assert_near_equal(
                        deriv_data['rel error'].forward, 0.0, tolerance=1e-6)


class TestSuttonGravesHeating(unittest.TestCase):
    """Tests for the Sutton-Graves stagnation heating component."""

    def _make_problem(self, gas_type='air'):
        prob = om.Problem()
        prob.model.add_subsystem('sg', SuttonGravesHeating(gas_type=gas_type))
        prob.setup(force_alloc_complex=True)
        return prob

    def test_positive_heat_flux(self):
        """Sutton-Graves should produce positive heat flux for typical conditions."""
        prob = self._make_problem()
        T_inf, rho, _ = _atmosphere_si(50000.0)
        prob.set_val('sg.velocity', 7000.0, units='m/s')
        prob.set_val('sg.density_inf', rho, units='kg/m**3')
        prob.set_val('sg.nose_radius', 1.0, units='m')
        prob.set_val('sg.wall_enthalpy', 3.0e5, units='J/kg')
        prob.set_val('sg.freestream_temperature', T_inf, units='K')
        prob.run_model()

        q = prob.get_val('sg.q_stag', units='W/m**2')[0]
        self.assertGreater(q, 0.0)
        self.assertGreater(q, 1.0e5, 'Re-entry heating should be significant')

    def test_zero_velocity_zero_heat_flux(self):
        """Zero velocity: stagnation enthalpy equals static enthalpy.
        If wall enthalpy equals freestream enthalpy, q should be zero."""
        prob = self._make_problem()
        T_inf = 288.15
        h_wall = 1005.0 * T_inf  # match freestream enthalpy
        prob.set_val('sg.velocity', 0.0, units='m/s')
        prob.set_val('sg.density_inf', 1.225, units='kg/m**3')
        prob.set_val('sg.nose_radius', 0.1, units='m')
        prob.set_val('sg.wall_enthalpy', h_wall, units='J/kg')
        prob.set_val('sg.freestream_temperature', T_inf, units='K')
        prob.run_model()

        q = prob.get_val('sg.q_stag', units='W/m**2')[0]
        assert_near_equal(q, 0.0, tolerance=1e-6)

    def test_inverse_sqrt_radius(self):
        """Doubling nose radius should reduce heat flux by factor sqrt(2)."""
        prob = self._make_problem()
        T_inf, rho, _ = _atmosphere_si(30000.0)
        prob.set_val('sg.velocity', 5000.0, units='m/s')
        prob.set_val('sg.density_inf', rho, units='kg/m**3')
        prob.set_val('sg.wall_enthalpy', 3.0e5, units='J/kg')
        prob.set_val('sg.freestream_temperature', T_inf, units='K')

        prob.set_val('sg.nose_radius', 1.0, units='m')
        prob.run_model()
        q1 = prob.get_val('sg.q_stag', units='W/m**2')[0]

        prob.set_val('sg.nose_radius', 2.0, units='m')
        prob.run_model()
        q2 = prob.get_val('sg.q_stag', units='W/m**2')[0]

        ratio = q1 / q2
        assert_near_equal(ratio, np.sqrt(2.0), tolerance=1e-6)

    def test_models_same_order_of_magnitude(self):
        """Fay-Riddell and Sutton-Graves should agree within an order of magnitude."""
        # Fay-Riddell
        prob_fr = om.Problem()
        prob_fr.model.add_subsystem('fr', FayRiddellHeating())
        prob_fr.setup(force_alloc_complex=True)
        prob_fr.set_val('fr.velocity', 5000.0, units='m/s')
        prob_fr.set_val('fr.altitude', 40000.0, units='m')
        prob_fr.set_val('fr.nose_radius', 1.0, units='m')
        prob_fr.set_val('fr.wall_temperature', 300.0, units='K')
        prob_fr.run_model()
        q_fr = prob_fr.get_val('fr.q_stag', units='W/m**2')[0]

        # Sutton-Graves at the same condition
        T_inf, rho, _ = _atmosphere_si(40000.0)
        prob_sg = om.Problem()
        prob_sg.model.add_subsystem('sg', SuttonGravesHeating())
        prob_sg.setup(force_alloc_complex=True)
        prob_sg.set_val('sg.velocity', 5000.0, units='m/s')
        prob_sg.set_val('sg.density_inf', rho, units='kg/m**3')
        prob_sg.set_val('sg.nose_radius', 1.0, units='m')
        prob_sg.set_val('sg.wall_enthalpy', 1005.0 * 300.0, units='J/kg')
        prob_sg.set_val('sg.freestream_temperature', T_inf, units='K')
        prob_sg.run_model()
        q_sg = prob_sg.get_val('sg.q_stag', units='W/m**2')[0]

        # Both should be positive and within an order of magnitude
        self.assertGreater(q_fr, 0.0)
        self.assertGreater(q_sg, 0.0)
        ratio = q_fr / q_sg if q_sg > q_fr else q_sg / q_fr
        self.assertGreater(ratio, 0.1,
                           f'Models differ by more than 10x: FR={q_fr:.2e}, SG={q_sg:.2e}')

    def test_gas_type_co2_higher_than_n2(self):
        """CO2 has higher K than N2, so should produce higher heat flux."""
        T_inf, rho, _ = _atmosphere_si(30000.0)

        def run_sg(gas):
            p = om.Problem()
            p.model.add_subsystem('sg', SuttonGravesHeating(gas_type=gas))
            p.setup(force_alloc_complex=True)
            p.set_val('sg.velocity', 5000.0, units='m/s')
            p.set_val('sg.density_inf', rho, units='kg/m**3')
            p.set_val('sg.nose_radius', 1.0, units='m')
            p.set_val('sg.wall_enthalpy', 3.0e5, units='J/kg')
            p.set_val('sg.freestream_temperature', T_inf, units='K')
            p.run_model()
            return p.get_val('sg.q_stag', units='W/m**2')[0]

        q_co2 = run_sg('CO2')
        q_n2 = run_sg('N2')
        self.assertGreater(q_co2, q_n2,
                           'CO2 K-factor is higher than N2, so q should be higher')

    def test_partials(self):
        """Complex-step derivative check for Sutton-Graves."""
        prob = self._make_problem()
        T_inf, rho, _ = _atmosphere_si(30000.0)
        prob.set_val('sg.velocity', 3000.0, units='m/s')
        prob.set_val('sg.density_inf', rho, units='kg/m**3')
        prob.set_val('sg.nose_radius', 0.5, units='m')
        prob.set_val('sg.wall_enthalpy', 3.0e5, units='J/kg')
        prob.set_val('sg.freestream_temperature', T_inf, units='K')
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True,
                                   out_stream=None)
        for comp_name, comp_data in data.items():
            for (out_name, in_name), deriv_data in comp_data.items():
                if deriv_data['magnitude'].forward > 1e-15:
                    assert_near_equal(
                        deriv_data['rel error'].forward, 0.0, tolerance=1e-6)


class TestAtmosphereSI(unittest.TestCase):
    """Sanity checks for the SI standard atmosphere function."""

    def test_sea_level(self):
        """Sea level standard conditions."""
        T, rho, p = _atmosphere_si(0.0)
        assert_near_equal(T, 288.15, tolerance=1e-4)
        assert_near_equal(p, 101325.0, tolerance=1e-4)
        assert_near_equal(rho, 1.225, tolerance=0.005)

    def test_11km(self):
        """At 11 km, temperature should be near 216.65 K."""
        T, rho, p = _atmosphere_si(11000.0)
        assert_near_equal(T, 216.65, tolerance=0.01)

    def test_density_decreases_with_altitude(self):
        """Density should decrease monotonically with altitude."""
        _, rho0, _ = _atmosphere_si(0.0)
        _, rho10, _ = _atmosphere_si(10000.0)
        _, rho50, _ = _atmosphere_si(50000.0)
        self.assertGreater(rho0, rho10)
        self.assertGreater(rho10, rho50)


if __name__ == '__main__':
    unittest.main()
