"""
Thermal Protection System (TPS) mass estimation components.

Provides OpenMDAO components for radiative equilibrium wall temperature,
TPS thickness sizing, and total TPS mass estimation for high-speed vehicles.
"""

import numpy as np
import openmdao.api as om


# Material properties database for supported TPS types
TPS_MATERIALS = {
    'ablative': {'density': 545.0, 'h_ablation': 11.7e6, 'k': 0.2, 'cp': 1260.0},
    'ceramic': {'density': 144.0, 'h_ablation': None, 'k': 0.04, 'cp': 628.0},
    'metallic': {'density': 8440.0, 'h_ablation': None, 'k': 11.4, 'cp': 444.0},
}

# Stefan-Boltzmann constant (W / m^2 / K^4)
STEFAN_BOLTZMANN = 5.67e-8


class RadiativeEquilibrium(om.ExplicitComponent):
    """
    Compute the radiative equilibrium wall temperature.

    T_wall_eq = (q_stag / (epsilon * sigma))^0.25
    """

    def setup(self):
        self.add_input('q_stag', val=1.0e6, units='W/m**2',
                        desc='Stagnation-point heat flux')
        self.add_input('emissivity', val=0.85,
                        desc='Surface emissivity')

        self.add_output('T_wall_eq', val=2000.0, units='K',
                         desc='Radiative equilibrium wall temperature')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        q_stag = inputs['q_stag']
        epsilon = inputs['emissivity']

        outputs['T_wall_eq'] = (q_stag / (epsilon * STEFAN_BOLTZMANN)) ** 0.25


class TPSThickness(om.ExplicitComponent):
    """
    Compute required TPS thickness and mass per unit area.

    For ablative TPS:
        thickness = Q_total / (rho_tps * h_ablation)

    For insulative TPS (ceramic, metallic):
        thickness = sqrt(2 * alpha_tps * t_soak)
        where alpha_tps = k / (rho * cp)
    """

    def initialize(self):
        self.options.declare('tps_type', default='ablative',
                             values=['ablative', 'ceramic', 'metallic'],
                             desc='Type of TPS material')

    def setup(self):
        self.add_input('q_peak', val=1.0e6, units='W/m**2',
                        desc='Peak heat flux')
        self.add_input('total_heat_load_per_area', val=1.0e9, units='J/m**2',
                        desc='Total heat load per unit area')
        self.add_input('soak_time', val=2000.0, units='s',
                        desc='Soak-through time for insulative TPS')

        self.add_output('thickness', val=0.01, units='m',
                         desc='Required TPS thickness', lower=1.0e-6)
        self.add_output('tps_mass_per_area', val=1.0, units='kg/m**2',
                         desc='TPS mass per unit wetted area')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        tps_type = self.options['tps_type']
        mat = TPS_MATERIALS[tps_type]
        rho = mat['density']

        if tps_type == 'ablative':
            Q_total = inputs['total_heat_load_per_area']
            h_ablation = mat['h_ablation']
            thickness = Q_total / (rho * h_ablation)
        else:
            # Insulative: ceramic or metallic
            k = mat['k']
            cp = mat['cp']
            t_soak = inputs['soak_time']
            alpha = k / (rho * cp)
            thickness = np.sqrt(2.0 * alpha * t_soak)

        outputs['thickness'] = thickness
        outputs['tps_mass_per_area'] = rho * thickness


class TPSMass(om.ExplicitComponent):
    """
    Compute total TPS mass for the vehicle.

    mass_tps = tps_mass_per_area * wetted_area
    """

    def setup(self):
        self.add_input('tps_mass_per_area', val=1.0, units='kg/m**2',
                        desc='TPS mass per unit wetted area')
        self.add_input('wetted_area', val=10.0, units='m**2',
                        desc='Vehicle wetted area exposed to heating')

        self.add_output('mass_tps', val=10.0, units='kg',
                         desc='Total TPS mass')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        outputs['mass_tps'] = inputs['tps_mass_per_area'] * inputs['wetted_area']


class AerothermoGroup(om.Group):
    """
    Assembled group wiring RadiativeEquilibrium, TPSThickness, and TPSMass.
    """

    def initialize(self):
        self.options.declare('tps_type', default='ablative',
                             values=['ablative', 'ceramic', 'metallic'],
                             desc='Type of TPS material')

    def setup(self):
        tps_type = self.options['tps_type']

        self.add_subsystem('radiative_eq', RadiativeEquilibrium(),
                           promotes_inputs=['q_stag', 'emissivity'],
                           promotes_outputs=['T_wall_eq'])

        self.add_subsystem('tps_thickness', TPSThickness(tps_type=tps_type),
                           promotes_inputs=['q_peak', 'total_heat_load_per_area',
                                            'soak_time'],
                           promotes_outputs=['thickness', 'tps_mass_per_area'])

        self.add_subsystem('tps_mass', TPSMass(),
                           promotes_inputs=['tps_mass_per_area', 'wetted_area'],
                           promotes_outputs=['mass_tps'])
