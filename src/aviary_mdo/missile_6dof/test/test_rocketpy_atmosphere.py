"""
Tests for the RocketPy ISA atmosphere wrapper.

Validates thermodynamic properties against ISA standard values at several
altitudes: sea level, tropopause (11 km), and a typical missile engagement
altitude (6096 m / 20,000 ft).
"""

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile_6dof.missile_6dof_variables import Missile6DOF


class TestRocketPyAtmosphere(unittest.TestCase):
    """Test RocketPy atmosphere component against ISA standard values."""

    def setUp(self):
        from aviary_mdo.missile_6dof.model.rocketpy_atmosphere import (
            RocketPyAtmosphere,
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'atm', RocketPyAtmosphere(), promotes=['*']
        )
        self.prob.setup(force_alloc_complex=True)

    def test_sea_level(self):
        """Sea level ISA: T=288.15K, P=101325Pa, rho=1.225 kg/m^3."""
        self.prob.set_val(Missile6DOF.Atmosphere.ALTITUDE, 0.0, units='m')
        self.prob.run_model()

        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.TEMPERATURE, units='K'),
            288.15, tolerance=0.01,
        )
        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.PRESSURE, units='Pa'),
            101325.0, tolerance=0.01,
        )
        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.DENSITY, units='kg/m**3'),
            1.225, tolerance=0.01,
        )
        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.SPEED_OF_SOUND, units='m/s'),
            340.3, tolerance=0.01,
        )

    def test_11km_tropopause(self):
        """Tropopause at 11 km: T~216.65 K, P~22632 Pa."""
        self.prob.set_val(Missile6DOF.Atmosphere.ALTITUDE, 11000.0, units='m')
        self.prob.run_model()

        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.TEMPERATURE, units='K'),
            216.65, tolerance=0.02,
        )
        assert_near_equal(
            self.prob.get_val(Missile6DOF.Atmosphere.PRESSURE, units='Pa'),
            22632.0, tolerance=0.05,
        )

    def test_6096m(self):
        """20,000 ft (6096 m) -- typical missile launch altitude."""
        self.prob.set_val(Missile6DOF.Atmosphere.ALTITUDE, 6096.0, units='m')
        self.prob.run_model()

        T = self.prob.get_val(
            Missile6DOF.Atmosphere.TEMPERATURE, units='K'
        )[0]
        rho = self.prob.get_val(
            Missile6DOF.Atmosphere.DENSITY, units='kg/m**3'
        )[0]

        self.assertTrue(200 < T < 280, f"Temp {T}K out of range at 6096m")
        self.assertTrue(0.5 < rho < 1.0, f"Density {rho} out of range at 6096m")

    def test_dynamic_viscosity_positive(self):
        """Dynamic viscosity should be positive at any altitude."""
        for alt in [0.0, 5000.0, 11000.0]:
            self.prob.set_val(
                Missile6DOF.Atmosphere.ALTITUDE, alt, units='m'
            )
            self.prob.run_model()
            mu = self.prob.get_val(
                Missile6DOF.Atmosphere.DYNAMIC_VISCOSITY, units='Pa*s'
            )[0]
            self.assertGreater(mu, 0.0, f"Viscosity should be positive at {alt}m")


if __name__ == '__main__':
    unittest.main()
