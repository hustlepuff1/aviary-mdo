"""
Aerothermodynamics pre-mission system.

Assembles all aerothermal discipline components into a single OpenMDAO Group
that computes stagnation heating, surface heat flux distribution, and TPS
mass estimation in the pre-mission phase.

Flow: atmosphere -> flight conditions -> stagnation heating ->
      surface heat flux -> radiative equilibrium -> TPS sizing -> TPS mass
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.aerothermodynamics.model.stagnation_heating import (
    SuttonGravesHeating,
)
from aviary_mdo.aerothermodynamics.model.surface_heat_flux import (
    SurfaceHeatFlux,
)
from aviary_mdo.aerothermodynamics.model.tps_sizing import (
    RadiativeEquilibrium, TPSThickness, TPSMass,
)


class SimpleAtmosphere(om.ExplicitComponent):
    """
    Simplified 1976 US Standard Atmosphere lookup.

    Given altitude, returns temperature, density, and speed of sound.
    """

    def setup(self):
        self.add_input('altitude', val=30000.0, units='m',
                       desc='Altitude above sea level')

        self.add_output('temperature', val=226.5, units='K',
                        desc='Freestream temperature')
        self.add_output('density', val=0.018, units='kg/m**3',
                        desc='Freestream density')
        self.add_output('speed_of_sound', val=301.7, units='m/s',
                        desc='Speed of sound')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        alt = inputs['altitude']

        g0 = 9.80665
        R_air = 287.0528
        gamma = 1.4

        if np.real(alt) <= 11000.0:
            T = 288.15 - 0.0065 * alt
            p = 101325.0 * (T / 288.15) ** 5.2561
        elif np.real(alt) <= 25000.0:
            T = 216.65
            p11 = 101325.0 * (216.65 / 288.15) ** 5.2561
            p = p11 * np.exp(-g0 * (alt - 11000.0) / (R_air * 216.65))
        elif np.real(alt) <= 47000.0:
            T11 = 216.65
            p11 = 101325.0 * (T11 / 288.15) ** 5.2561
            p20 = p11 * np.exp(-g0 * 9000.0 / (R_air * T11))
            T20 = 216.65
            lapse = 0.001
            T = T20 + lapse * (alt - 20000.0)
            p = p20 * (T / T20) ** (-g0 / (lapse * R_air))
        else:
            T11 = 216.65
            p11 = 101325.0 * (T11 / 288.15) ** 5.2561
            p20 = p11 * np.exp(-g0 * 9000.0 / (R_air * T11))
            T20 = 216.65
            lapse_s = 0.001
            T47 = T20 + lapse_s * 27000.0
            p47 = p20 * (T47 / T20) ** (-g0 / (lapse_s * R_air))
            T = T47
            p = p47 * np.exp(-g0 * (alt - 47000.0) / (R_air * T))

        rho = p / (R_air * T)
        a = np.sqrt(gamma * R_air * T)

        outputs['temperature'] = T
        outputs['density'] = rho
        outputs['speed_of_sound'] = a


class FlightCondition(om.ExplicitComponent):
    """
    Compute freestream velocity and wall enthalpy from Mach number
    and atmospheric conditions.
    """

    def setup(self):
        self.add_input('mach', val=5.0,
                       desc='Design Mach number')
        self.add_input('speed_of_sound', val=301.7, units='m/s',
                       desc='Speed of sound')
        self.add_input('wall_temperature', val=300.0, units='K',
                       desc='Wall temperature')

        self.add_output('velocity', val=1500.0, units='m/s',
                        desc='Freestream velocity')
        self.add_output('wall_enthalpy', val=3.015e5, units='J/kg',
                        desc='Wall enthalpy = cp * T_wall')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['mach']
        a = inputs['speed_of_sound']
        T_w = inputs['wall_temperature']

        cp = 1005.0  # J/(kg*K)

        outputs['velocity'] = M * a
        outputs['wall_enthalpy'] = cp * T_w


class HeatLoadPerArea(om.ExplicitComponent):
    """
    Estimate total heat load per unit area from peak heat flux and
    an assumed exposure time for TPS sizing.
    """

    def setup(self):
        self.add_input('q_stag', val=1.0e6, units='W/m**2',
                       desc='Stagnation heat flux')
        self.add_input('exposure_time', val=60.0, units='s',
                       desc='Estimated heating exposure time')

        self.add_output('total_heat_load_per_area', val=6.0e7, units='J/m**2',
                        desc='Total heat load per unit area')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        # Approximate: Q = q * t * shape_factor
        # shape_factor ~ 0.5 accounts for non-constant heating profile
        shape_factor = 0.5
        outputs['total_heat_load_per_area'] = (
            inputs['q_stag'] * inputs['exposure_time'] * shape_factor
        )


class AerothermoPremission(om.Group):
    """
    Complete aerothermodynamics pre-mission sizing.

    Computes stagnation heating, surface heat flux distribution,
    radiative equilibrium temperature, and TPS mass from design
    Mach, altitude, geometry, and material selection.
    """

    def initialize(self):
        self.options.declare('tps_type', default='ablative',
                             values=['ablative', 'ceramic', 'metallic'],
                             desc='Type of TPS material')

    def setup(self):
        tps_type = self.options['tps_type']

        # 1. Atmosphere lookup
        self.add_subsystem('atmosphere', SimpleAtmosphere(),
                           promotes_inputs=['altitude'],
                           promotes_outputs=['temperature', 'density',
                                             'speed_of_sound'])

        # 2. Flight conditions
        self.add_subsystem('flight_condition', FlightCondition(),
                           promotes_inputs=['mach', 'speed_of_sound',
                                            'wall_temperature'],
                           promotes_outputs=['velocity', 'wall_enthalpy'])

        # 3. Stagnation heating (Sutton-Graves)
        self.add_subsystem('stagnation', SuttonGravesHeating(),
                           promotes_inputs=['velocity', 'density_inf',
                                            'nose_radius', 'wall_enthalpy',
                                            'freestream_temperature'],
                           promotes_outputs=['q_stag'])

        # Connect atmosphere to stagnation heating
        self.connect('density', 'density_inf')
        self.connect('temperature', 'freestream_temperature')

        # 4. Surface heat flux distribution
        self.add_subsystem('surface_heating', SurfaceHeatFlux(),
                           promotes_inputs=['q_stag', 'velocity', 'altitude',
                                            'nose_radius', 'wall_temperature',
                                            'body_length'],
                           promotes_outputs=['total_heat_load'])

        # 5a. Radiative equilibrium temperature
        self.add_subsystem('radiative_eq', RadiativeEquilibrium(),
                           promotes_inputs=['q_stag'],
                           promotes_outputs=['T_wall_eq'])

        # 5b. Heat load per area for TPS sizing
        self.add_subsystem('heat_load', HeatLoadPerArea(),
                           promotes_inputs=['q_stag'],
                           promotes_outputs=['total_heat_load_per_area'])

        # 5c. TPS thickness sizing
        self.add_subsystem('tps_thickness', TPSThickness(tps_type=tps_type),
                           promotes_inputs=['total_heat_load_per_area'],
                           promotes_outputs=['thickness', 'tps_mass_per_area'])

        # Connect peak heat flux for TPS sizing
        self.connect('q_stag', 'tps_thickness.q_peak')

        # 5d. TPS mass
        self.add_subsystem('tps_mass', TPSMass(),
                           promotes_inputs=['tps_mass_per_area', 'wetted_area'],
                           promotes_outputs=['mass_tps'])

        # Set default input values
        self.set_input_defaults('altitude', val=30000.0, units='m')
        self.set_input_defaults('mach', val=5.0)
        self.set_input_defaults('nose_radius', val=0.5, units='m')
        self.set_input_defaults('wall_temperature', val=300.0, units='K')
        self.set_input_defaults('body_length', val=5.0, units='m')
        self.set_input_defaults('wetted_area', val=20.0, units='m**2')
