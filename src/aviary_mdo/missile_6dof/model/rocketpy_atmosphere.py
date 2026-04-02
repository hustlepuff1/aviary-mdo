"""
RocketPy ISA atmosphere wrapper for the 6-DOF missile subsystem.

Evaluates the International Standard Atmosphere at a given altitude using
RocketPy's ``Environment`` class.  The component exposes five thermodynamic
properties (density, temperature, pressure, speed of sound, dynamic viscosity)
that downstream aero and propulsion components consume.

Because the RocketPy ``Function`` objects are opaque to complex-step
differentiation, partials are declared with finite differences.
"""

import numpy as np
import openmdao.api as om
from rocketpy import Environment

from aviary_mdo.missile_6dof.missile_6dof_variables import Missile6DOF


class RocketPyAtmosphere(om.ExplicitComponent):
    """
    OpenMDAO wrapper around RocketPy's ISA standard atmosphere model.

    Inputs
    ------
    missile_6dof:atm:altitude : float  (m)
        Geometric altitude above mean sea level.

    Outputs
    -------
    missile_6dof:atm:density           : float  (kg/m**3)
    missile_6dof:atm:temperature       : float  (K)
    missile_6dof:atm:pressure          : float  (Pa)
    missile_6dof:atm:speed_of_sound    : float  (m/s)
    missile_6dof:atm:dynamic_viscosity : float  (Pa*s)
    """

    def setup(self):
        # --- Build the RocketPy ISA environment once ---
        self._env = Environment()
        self._env.set_atmospheric_model(type="standard_atmosphere")

        # --- Inputs ---
        self.add_input(
            Missile6DOF.Atmosphere.ALTITUDE,
            val=0.0,
            units='m',
            desc='Geometric altitude above MSL',
        )

        # --- Outputs ---
        self.add_output(
            Missile6DOF.Atmosphere.DENSITY,
            val=1.225,
            units='kg/m**3',
            desc='Air density',
        )
        self.add_output(
            Missile6DOF.Atmosphere.TEMPERATURE,
            val=288.15,
            units='K',
            desc='Air static temperature',
        )
        self.add_output(
            Missile6DOF.Atmosphere.PRESSURE,
            val=101325.0,
            units='Pa',
            desc='Air static pressure',
        )
        self.add_output(
            Missile6DOF.Atmosphere.SPEED_OF_SOUND,
            val=340.3,
            units='m/s',
            desc='Speed of sound',
        )
        self.add_output(
            Missile6DOF.Atmosphere.DYNAMIC_VISCOSITY,
            val=1.789e-5,
            units='Pa*s',
            desc='Dynamic viscosity of air',
        )

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        alt = float(np.real(inputs[Missile6DOF.Atmosphere.ALTITUDE]))

        # Clamp to non-negative altitude (solver iterations may go negative)
        alt = max(alt, 0.0)

        env = self._env
        outputs[Missile6DOF.Atmosphere.DENSITY] = float(env.density(alt))
        outputs[Missile6DOF.Atmosphere.TEMPERATURE] = float(env.temperature(alt))
        outputs[Missile6DOF.Atmosphere.PRESSURE] = float(env.pressure(alt))
        outputs[Missile6DOF.Atmosphere.SPEED_OF_SOUND] = float(env.speed_of_sound(alt))
        outputs[Missile6DOF.Atmosphere.DYNAMIC_VISCOSITY] = float(
            env.dynamic_viscosity(alt)
        )
