"""
Tests for the AerothermoBuilder and AerothermoPremission.

Tests cover builder instantiation, interface methods, and a full
pre-mission run at Mach 5 / 30 km altitude.
"""

import unittest

import openmdao.api as om

from aviary_mdo.aerothermodynamics.aerothermo_builder import AerothermoBuilder
from aviary_mdo.aerothermodynamics.aerothermo_variables import Aerothermo
from aviary_mdo.aerothermodynamics.model.aerothermo_premission import (
    AerothermoPremission,
)


class TestAerothermoBuilderInstantiation(unittest.TestCase):
    """Test basic builder instantiation and defaults."""

    def test_default_name(self):
        builder = AerothermoBuilder()
        self.assertEqual(builder.name, 'aerothermo')

    def test_custom_name(self):
        builder = AerothermoBuilder(name='my_aerothermo')
        self.assertEqual(builder.name, 'my_aerothermo')

    def test_default_tps_type(self):
        builder = AerothermoBuilder()
        self.assertEqual(builder.tps_type, 'ablative')

    def test_custom_tps_type(self):
        builder = AerothermoBuilder(tps_type='ceramic')
        self.assertEqual(builder.tps_type, 'ceramic')


class TestAerothermoBuilderInterface(unittest.TestCase):
    """Test builder interface methods."""

    def setUp(self):
        self.builder = AerothermoBuilder()

    def test_build_pre_mission_returns_group(self):
        pre_mission = self.builder.build_pre_mission(aviary_inputs=None)
        self.assertIsInstance(pre_mission, om.Group)
        self.assertIsInstance(pre_mission, AerothermoPremission)

    def test_build_mission_returns_none(self):
        result = self.builder.build_mission(num_nodes=1, aviary_inputs=None)
        self.assertIsNone(result)

    def test_get_mass_names_returns_tps_mass(self):
        mass_names = self.builder.get_mass_names()
        self.assertIn(Aerothermo.Output.TPS_MASS, mass_names)

    def test_get_design_vars_includes_nose_radius(self):
        dvs = self.builder.get_design_vars()
        self.assertIn(Aerothermo.Design.NOSE_RADIUS, dvs)
        self.assertEqual(dvs[Aerothermo.Design.NOSE_RADIUS]['units'], 'm')
        self.assertGreater(
            dvs[Aerothermo.Design.NOSE_RADIUS]['upper'],
            dvs[Aerothermo.Design.NOSE_RADIUS]['lower'],
        )

    def test_get_design_vars_includes_wall_temperature(self):
        dvs = self.builder.get_design_vars()
        self.assertIn(Aerothermo.Design.WALL_TEMPERATURE, dvs)

    def test_get_parameters(self):
        params = self.builder.get_parameters()
        self.assertIn(Aerothermo.Design.DESIGN_MACH, params)
        self.assertIn(Aerothermo.Design.DESIGN_ALTITUDE, params)


class TestAerothermoPremissionRun(unittest.TestCase):
    """Test a full pre-mission run at Mach 5 / 30 km altitude."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'aerothermo', AerothermoPremission(tps_type='ablative'),
            promotes=['*'],
        )
        self.prob.setup(force_alloc_complex=True)

        # Set Mach 5, 30 km altitude conditions
        self.prob.set_val('mach', 5.0)
        self.prob.set_val('altitude', 30000.0, units='m')
        self.prob.set_val('nose_radius', 0.5, units='m')
        self.prob.set_val('wall_temperature', 300.0, units='K')
        self.prob.set_val('body_length', 5.0, units='m')
        self.prob.set_val('wetted_area', 20.0, units='m**2')

        self.prob.run_model()

    def test_positive_stagnation_heat_flux(self):
        q_stag = self.prob.get_val('q_stag', units='W/m**2')
        self.assertGreater(q_stag[0], 0.0,
                           'Stagnation heat flux should be positive at Mach 5')

    def test_positive_tps_mass(self):
        mass_tps = self.prob.get_val('mass_tps', units='kg')
        self.assertGreater(mass_tps[0], 0.0,
                           'TPS mass should be positive')

    def test_reasonable_equilibrium_temperature(self):
        T_eq = self.prob.get_val('T_wall_eq', units='K')
        self.assertGreater(T_eq[0], 1000.0,
                           'Equilibrium temperature should be > 1000 K at Mach 5')
        self.assertLess(T_eq[0], 3000.0,
                        'Equilibrium temperature should be < 3000 K at Mach 5')

    def test_positive_total_heat_load(self):
        Q_total = self.prob.get_val('total_heat_load', units='W')
        self.assertGreater(Q_total[0], 0.0,
                           'Total heat load should be positive')

    def test_positive_tps_thickness(self):
        thickness = self.prob.get_val('thickness', units='m')
        self.assertGreater(thickness[0], 0.0,
                           'TPS thickness should be positive')

    def test_velocity_correct(self):
        V = self.prob.get_val('velocity', units='m/s')
        # At 30 km, speed of sound ~ 301.7 m/s, so V ~ 5 * 301.7 ~ 1508 m/s
        self.assertGreater(V[0], 1400.0)
        self.assertLess(V[0], 1600.0)


if __name__ == '__main__':
    unittest.main()
