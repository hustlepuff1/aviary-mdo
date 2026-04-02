"""
Tactical Missile Design warhead discipline components.

Implements blast-fragmentation, penetration, and lethality models
from "Tactical Missile Design" (AIAA Education Series), Sheet 8.

Fragmentation:
  Gurney equation for cylindrical warhead fragment velocity.

Blast Overpressure:
  Hopkinson-Cranz scaled distance overpressure.

Penetration:
  Empirical penetrator depth correlation.

Lethality:
  Single-shot probability of kill (Pk) from fragment spray.

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001.
"""

import numpy as np
import openmdao.api as om


class WarheadFragmentation(om.ExplicitComponent):
    """
    Warhead fragmentation using the Gurney equation for a cylindrical warhead.

    Computes charge mass, metal mass, explosive weight, and initial
    fragment velocity from warhead parameters.
    """

    def setup(self):
        self.add_input('warhead_mass', val=2.4, units='slug',
                       desc='Total warhead mass')
        self.add_input('charge_to_metal_ratio', val=1.0,
                       desc='Charge-to-metal mass ratio (C/M)')
        self.add_input('charge_energy_sqrt', val=10230.0, units='ft/s',
                       desc='Gurney characteristic velocity sqrt(2E)')
        self.add_input('fragment_weight', val=0.00022, units='slug',
                       desc='Individual fragment weight')
        self.add_input('fragment_density', val=15.2, units='slug/ft**3',
                       desc='Fragment material density')

        self.add_output('charge_mass', val=0.0, units='slug',
                        desc='Mass of explosive charge (Mc)')
        self.add_output('metal_mass', val=0.0, units='slug',
                        desc='Mass of metal casing (Mm)')
        self.add_output('explosive_weight', val=0.0, units='lbm',
                        desc='Explosive weight in lbm')
        self.add_output('fragment_velocity', val=0.0, units='ft/s',
                        desc='Initial fragment velocity (Gurney)')
        self.add_output('num_fragments', val=0.0,
                        desc='Estimated number of fragments')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Mwh = inputs['warhead_mass']
        C_M = inputs['charge_to_metal_ratio']
        Ec = inputs['charge_energy_sqrt']
        Wf = inputs['fragment_weight']

        # Charge and metal masses
        Mc = Mwh * C_M / (1.0 + C_M)
        Mm = Mwh - Mc

        outputs['charge_mass'] = Mc
        outputs['metal_mass'] = Mm

        # Explosive weight in lbm (slug * g = lbm)
        W_expl = Mc * 32.174
        outputs['explosive_weight'] = W_expl

        # Gurney equation for cylindrical warhead:
        #   Vf = sqrt(2E) * sqrt(C/M / (1 + 0.5 * C/M))
        Vf = Ec * np.sqrt(C_M / (1.0 + 0.5 * C_M))
        outputs['fragment_velocity'] = Vf

        # Number of fragments = metal mass / fragment weight
        if Wf > 0.0:
            outputs['num_fragments'] = Mm / Wf
        else:
            outputs['num_fragments'] = 0.0


class WarheadBlastOverpressure(om.ExplicitComponent):
    """
    Blast overpressure from Hopkinson-Cranz scaling.

    Computes peak overpressure at a given standoff range
    using scaled distance Z = R / W^(1/3).
    """

    def setup(self):
        self.add_input('explosive_weight', val=38.64, units='lbm',
                       desc='Explosive weight')
        self.add_input('target_range', val=8.0, units='ft',
                       desc='Standoff distance to target')
        self.add_input('atmospheric_pressure', val=14.696, units='psi',
                       desc='Ambient atmospheric pressure')

        self.add_output('scaled_distance', val=0.0,
                        desc='Hopkinson-Cranz scaled distance Z (ft/lbm^1/3)')
        self.add_output('peak_overpressure', val=0.0, units='psi',
                        desc='Peak overpressure at target range')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        W = inputs['explosive_weight']
        R = inputs['target_range']
        P0 = inputs['atmospheric_pressure']

        # Scaled distance (Hopkinson-Cranz)
        W_cuberoot = W ** (1.0 / 3.0)
        Z = R / W_cuberoot
        outputs['scaled_distance'] = Z

        # Kingery-Bulmash empirical overpressure (simplified fit)
        # Peak overpressure Ps/P0 = 808 * (1 + (Z/4.5)^2) /
        #   sqrt(1 + (Z/0.048)^2) / sqrt(1 + (Z/0.32)^2) / sqrt(1 + (Z/1.35)^2)
        # Simplified: Ps = P0 * K / Z^3 for moderate Z
        if Z > 0.0:
            Ps = P0 * (808.0 / (Z ** 3)) * np.sqrt(
                (1.0 + (Z / 4.5) ** 2)
                / ((1.0 + (Z / 0.048) ** 2)
                   * (1.0 + (Z / 0.32) ** 2)
                   * (1.0 + (Z / 1.35) ** 2))
            )
        else:
            Ps = 0.0

        outputs['peak_overpressure'] = Ps


class WarheadPenetration(om.ExplicitComponent):
    """
    Penetration depth for a kinetic energy penetrator.

    Uses a simplified empirical correlation relating penetrator geometry,
    impact velocity, and material properties to depth of penetration.
    """

    def setup(self):
        self.add_input('penetrator_length', val=14.0, units='inch',
                       desc='Penetrator length')
        self.add_input('penetrator_diameter', val=2.0, units='inch',
                       desc='Penetrator diameter')
        self.add_input('impact_velocity', val=2000.0, units='ft/s',
                       desc='Impact velocity')
        self.add_input('penetrator_density', val=15.2, units='slug/ft**3',
                       desc='Penetrator material density')
        self.add_input('target_density', val=15.2, units='slug/ft**3',
                       desc='Target material density')

        self.add_output('length_to_diameter', val=0.0,
                        desc='Penetrator L/D ratio')
        self.add_output('penetration_depth', val=0.0, units='inch',
                        desc='Depth of penetration')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        L = inputs['penetrator_length']
        D = inputs['penetrator_diameter']
        V = inputs['impact_velocity']
        rho_p = inputs['penetrator_density']
        rho_t = inputs['target_density']

        L_D = L / D
        outputs['length_to_diameter'] = L_D

        # Tate-Alekseevskii long rod penetration model (simplified):
        #   P = L * (rho_p / rho_t)^0.5
        # With velocity correction for hydrodynamic regime:
        #   P = L * sqrt(rho_p / rho_t) * (1 + 0.001 * (V - 1000))  for V > 1000
        density_ratio = np.sqrt(rho_p / rho_t)
        vel_factor = 1.0 + 0.001 * (V - 1000.0)

        P = L * density_ratio * vel_factor
        outputs['penetration_depth'] = P


class WarheadLethality(om.ExplicitComponent):
    """
    Single-shot probability of kill (Pk) from fragment spray.

    Assumes uniform fragment distribution over a spherical shell,
    computes expected number of hits on target vulnerable area,
    then Pk from a Poisson distribution.
    """

    def setup(self):
        self.add_input('fragment_velocity', val=8352.76, units='ft/s',
                       desc='Initial fragment velocity')
        self.add_input('target_range', val=8.0, units='ft',
                       desc='Distance to target at detonation')
        self.add_input('target_presented_area', val=20.0, units='ft**2',
                       desc='Target presented area')
        self.add_input('target_vulnerable_area', val=2.0, units='ft**2',
                       desc='Target vulnerable area')
        self.add_input('num_fragments', val=5454.5,
                       desc='Total number of fragments')

        self.add_output('fragment_density_at_target', val=0.0,
                        desc='Fragments per ft^2 at target range')
        self.add_output('expected_hits', val=0.0,
                        desc='Expected number of hits on vulnerable area')
        self.add_output('Pk', val=0.0,
                        desc='Single-shot probability of kill')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Vf = inputs['fragment_velocity']
        R = inputs['target_range']
        Ap = inputs['target_presented_area']
        Av = inputs['target_vulnerable_area']
        N = inputs['num_fragments']

        # Fragment spray covers a spherical shell at range R
        # Total sphere area = 4 * pi * R^2
        sphere_area = 4.0 * np.pi * R ** 2

        # Fragment spatial density (fragments / ft^2)
        if sphere_area > 0.0:
            frag_density = N / sphere_area
        else:
            frag_density = 0.0

        outputs['fragment_density_at_target'] = frag_density

        # Expected hits on the vulnerable area
        # Hits = frag_density * Av
        expected_hits = frag_density * Av
        outputs['expected_hits'] = expected_hits

        # Pk from Poisson: Pk = 1 - exp(-expected_hits)
        Pk = 1.0 - np.exp(-expected_hits)
        outputs['Pk'] = Pk


class WarheadGroup(om.Group):
    """
    Complete warhead analysis group.

    Connects fragmentation outputs to lethality inputs.
    """

    def setup(self):
        self.add_subsystem('fragmentation', WarheadFragmentation(),
                           promotes_inputs=['warhead_mass',
                                            'charge_to_metal_ratio',
                                            'charge_energy_sqrt',
                                            'fragment_weight',
                                            'fragment_density'],
                           promotes_outputs=['charge_mass', 'metal_mass',
                                             'explosive_weight',
                                             'fragment_velocity',
                                             'num_fragments'])

        self.add_subsystem('blast', WarheadBlastOverpressure(),
                           promotes_inputs=['target_range',
                                            'atmospheric_pressure'],
                           promotes_outputs=['scaled_distance',
                                             'peak_overpressure'])

        self.add_subsystem('penetration', WarheadPenetration(),
                           promotes_inputs=['penetrator_length',
                                            'penetrator_diameter',
                                            'impact_velocity'],
                           promotes_outputs=['length_to_diameter',
                                             'penetration_depth'])

        self.add_subsystem('lethality', WarheadLethality(),
                           promotes_inputs=['target_range',
                                            'target_presented_area',
                                            'target_vulnerable_area'],
                           promotes_outputs=['fragment_density_at_target',
                                             'expected_hits', 'Pk'])

        # Connect fragmentation outputs to downstream components
        self.connect('explosive_weight', 'blast.explosive_weight')
        self.connect('fragment_velocity', 'lethality.fragment_velocity')
        self.connect('num_fragments', 'lethality.num_fragments')
