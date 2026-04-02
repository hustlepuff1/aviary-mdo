"""Standard atmosphere component for missile trajectory ODE.

Vectorized over num_nodes. Pure math — supports complex-step partials.
US Standard Atmosphere 1976, valid 0–86 km.
"""

import numpy as np
import openmdao.api as om


class MissileAtmosComp(om.ExplicitComponent):
    """Compute atmospheric properties from altitude (SI units).

    Inputs: altitude (m)
    Outputs: temperature (K), pressure (Pa), density (kg/m^3), speed_of_sound (m/s)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('altitude', val=np.zeros(nn), units='m',
                        desc='Geometric altitude')
        self.add_output('temperature', val=288.15 * np.ones(nn), units='K')
        self.add_output('pressure', val=101325.0 * np.ones(nn), units='Pa')
        self.add_output('density', val=1.225 * np.ones(nn), units='kg/m**3')
        self.add_output('speed_of_sound', val=340.3 * np.ones(nn), units='m/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.declare_partials('temperature', 'altitude', rows=ar, cols=ar)
        self.declare_partials('pressure', 'altitude', rows=ar, cols=ar)
        self.declare_partials('density', 'altitude', rows=ar, cols=ar)
        self.declare_partials('speed_of_sound', 'altitude', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        h = inputs['altitude']
        gamma_air = 1.4
        R = 287.058

        # Troposphere: 0-11 km
        # Stratosphere: 11-20 km (isothermal at 216.65 K)
        # Upper strato: 20-32 km
        T0 = 288.15
        P0 = 101325.0
        rho0 = 1.225

        # Lapse rate (troposphere)
        L = 0.0065  # K/m
        h_trop = 11000.0
        T_trop = T0 - L * h_trop  # 216.65 K

        # Clamp altitude for numerical safety
        h_safe = np.clip(h, 0, 80000.0)

        # Piecewise atmosphere
        in_trop = h_safe <= h_trop

        # Troposphere
        T_t = T0 - L * h_safe
        P_t = P0 * (T_t / T0) ** (9.80665 / (L * R))
        rho_t = P_t / (R * T_t)

        # Stratosphere (isothermal)
        g_over_RT = 9.80665 / (R * T_trop)
        P_11 = P0 * (T_trop / T0) ** (9.80665 / (L * R))
        P_s = P_11 * np.exp(-g_over_RT * (h_safe - h_trop))
        rho_s = P_s / (R * T_trop)

        T = np.where(in_trop, T_t, T_trop)
        P = np.where(in_trop, P_t, P_s)
        rho = np.where(in_trop, rho_t, rho_s)
        a = np.sqrt(gamma_air * R * T)

        outputs['temperature'] = T
        outputs['pressure'] = P
        outputs['density'] = rho
        outputs['speed_of_sound'] = a

    def compute_partials(self, inputs, J):
        h = inputs['altitude']
        gamma_air = 1.4
        R = 287.058
        g = 9.80665
        T0 = 288.15
        P0 = 101325.0
        L = 0.0065
        h_trop = 11000.0
        T_trop = T0 - L * h_trop
        exponent = g / (L * R)

        h_safe = np.clip(h, 0, 80000.0)
        in_trop = h_safe <= h_trop

        # Troposphere partials
        T_t = T0 - L * h_safe
        dT_dh_t = -L
        P_t = P0 * (T_t / T0) ** exponent
        dP_dh_t = P_t * exponent * (-L) / T_t
        rho_t = P_t / (R * T_t)
        drho_dh_t = (dP_dh_t * T_t - P_t * dT_dh_t) / (R * T_t ** 2)

        # Stratosphere partials
        g_over_RT = g / (R * T_trop)
        P_11 = P0 * (T_trop / T0) ** exponent
        P_s = P_11 * np.exp(-g_over_RT * (h_safe - h_trop))
        dP_dh_s = -g_over_RT * P_s
        drho_dh_s = dP_dh_s / (R * T_trop)

        T = np.where(in_trop, T_t, T_trop)
        a = np.sqrt(gamma_air * R * T)

        J['temperature', 'altitude'] = np.where(in_trop, dT_dh_t, 0.0)
        J['pressure', 'altitude'] = np.where(in_trop, dP_dh_t, dP_dh_s)
        J['density', 'altitude'] = np.where(in_trop, drho_dh_t, drho_dh_s)
        # da/dh = 0.5 * (gamma*R/T)^0.5 * dT/dh  (only in troposphere)
        da_dh = np.where(in_trop, 0.5 * np.sqrt(gamma_air * R / T) * dT_dh_t, 0.0)
        J['speed_of_sound', 'altitude'] = da_dh
