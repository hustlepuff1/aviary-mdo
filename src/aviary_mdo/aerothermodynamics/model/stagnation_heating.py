"""
Stagnation point heating models for hypersonic vehicle design.

Implements two classical correlations:

1. Fay-Riddell (1958) — equilibrium stagnation point heat transfer
   Reference: Fay & Riddell, "Theory of Stagnation Point Heat Transfer
   in Dissociated Air," J. Aeronautical Sciences, Vol 25, No 2, 1958.

2. Sutton-Graves (1971) — engineering correlation for convective heating
   Reference: Sutton & Graves, "A General Stagnation-Point Convective-
   Heating Equation for Arbitrary Gas Mixtures," NASA TR R-376, 1971.
"""

import numpy as np
import openmdao.api as om


# ===== SI Standard Atmosphere =====

def _atmosphere_si(alt_m):
    """
    Simplified 1976 US Standard Atmosphere (SI units).

    Compatible with complex-step differentiation: branching uses the
    real part of altitude, but arithmetic preserves the complex part.

    Parameters
    ----------
    alt_m : float or complex or ndarray
        Geometric altitude in meters.

    Returns
    -------
    T : float or complex
        Temperature in Kelvin.
    rho : float or complex
        Density in kg/m^3.
    p : float or complex
        Pressure in Pa.
    """
    # Extract scalar; use real part for branching, keep full value for math
    h = np.asarray(alt_m).flat[0]
    h_real = np.real(h)
    if h_real < 0.0:
        h = h - h_real  # clamp to zero while keeping imaginary part

    # Constants
    g0 = 9.80665       # m/s^2
    R_air = 287.0528    # J/(kg*K)

    if h_real <= 11000.0:
        # Troposphere: linear lapse rate
        T0 = 288.15     # K at sea level
        lapse = -0.0065  # K/m
        T = T0 + lapse * h
        p0 = 101325.0    # Pa
        p = p0 * (T / T0) ** (-g0 / (lapse * R_air))
    elif h_real <= 25000.0:
        # Lower stratosphere: isothermal
        T11 = 216.65
        p11 = 101325.0 * (T11 / 288.15) ** (-g0 / (-0.0065 * R_air))
        T = T11 + 0.0 * h  # constant T but keep complex dependence
        p = p11 * np.exp(-g0 * (h - 11000.0) / (R_air * T11))
    elif h_real <= 47000.0:
        # Upper stratosphere: positive lapse rate
        T11 = 216.65
        p11 = 101325.0 * (T11 / 288.15) ** (-g0 / (-0.0065 * R_air))
        p20 = p11 * np.exp(-g0 * (20000.0 - 11000.0) / (R_air * T11))
        T20 = 216.65
        lapse = 0.001  # K/m from 20 km
        T = T20 + lapse * (h - 20000.0)
        p = p20 * (T / T20) ** (-g0 / (lapse * R_air))
    else:
        # Above 47 km: simplified isothermal approximation
        T11 = 216.65
        p11 = 101325.0 * (T11 / 288.15) ** (-g0 / (-0.0065 * R_air))
        p20 = p11 * np.exp(-g0 * (20000.0 - 11000.0) / (R_air * T11))
        T20 = 216.65
        lapse_strato = 0.001
        T47 = T20 + lapse_strato * (47000.0 - 20000.0)
        p47 = p20 * (T47 / T20) ** (-g0 / (lapse_strato * R_air))
        T = T47 + 0.0 * h  # constant T but keep complex dependence
        p = p47 * np.exp(-g0 * (h - 47000.0) / (R_air * T47))

    rho = p / (R_air * T)
    return T, rho, p


# ===== Thermodynamic helpers =====

def _sutherland_viscosity(T):
    """
    Dynamic viscosity of air via Sutherland's law.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    mu : float
        Dynamic viscosity in Pa*s.
    """
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def _specific_enthalpy(T):
    """
    Approximate specific enthalpy of air (calorically perfect gas).

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    h : float
        Specific enthalpy in J/kg.
    """
    cp = 1005.0  # J/(kg*K) for air
    return cp * T


def _stagnation_temperature(T_inf, V_inf):
    """
    Stagnation (total) temperature.

    Parameters
    ----------
    T_inf : float
        Freestream temperature in K.
    V_inf : float
        Freestream velocity in m/s.

    Returns
    -------
    T0 : float
        Stagnation temperature in K.
    """
    cp = 1005.0
    return T_inf + V_inf ** 2 / (2.0 * cp)


def _stagnation_enthalpy(T_inf, V_inf):
    """
    Stagnation enthalpy.

    Parameters
    ----------
    T_inf : float
        Freestream temperature in K.
    V_inf : float
        Freestream velocity in m/s.

    Returns
    -------
    h0 : float
        Stagnation enthalpy in J/kg.
    """
    return _specific_enthalpy(T_inf) + 0.5 * V_inf ** 2


# ===== OpenMDAO Components =====

class FayRiddellHeating(om.ExplicitComponent):
    """
    Stagnation point convective heat flux using the Fay-Riddell (1958)
    correlation.

    From Fay & Riddell (1958), the equilibrium stagnation point heat
    transfer for a sphere (Eq. 63 simplified):

        q = 0.94 * (rho_s * mu_s)^0.4 * (rho_w * mu_w)^0.1
            * sqrt(du_e/dx) * (h_s - h_w)

    where the velocity gradient at the stagnation point for a sphere:

        du_e/dx = (1/R) * sqrt(2 * (p_s - p_inf) / rho_s)

    Post-shock stagnation conditions are computed using normal shock
    and isentropic relations.

    Also outputs stagnation temperature and stagnation pressure.
    """

    def setup(self):
        self.add_input('velocity', val=1000.0, units='m/s',
                        desc='Freestream velocity')
        self.add_input('altitude', val=0.0, units='m',
                        desc='Altitude above sea level')
        self.add_input('nose_radius', val=0.1, units='m',
                        desc='Nose radius of curvature')
        self.add_input('wall_temperature', val=300.0, units='K',
                        desc='Wall temperature')

        self.add_output('q_stag', val=0.0, units='W/m**2',
                         desc='Stagnation point heat flux')
        self.add_output('stagnation_temperature', val=0.0, units='K',
                         desc='Stagnation (total) temperature')
        self.add_output('stagnation_pressure', val=0.0, units='Pa',
                         desc='Stagnation (total) pressure')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        V = inputs['velocity']
        alt = inputs['altitude']
        R_n = inputs['nose_radius']
        T_w = inputs['wall_temperature']

        gamma = 1.4
        R_gas = 287.0528  # J/(kg*K)

        # Atmosphere
        T_inf, rho_inf, p_inf = _atmosphere_si(alt)

        # Stagnation temperature
        T0 = _stagnation_temperature(T_inf, V)
        outputs['stagnation_temperature'] = T0

        # Stagnation pressure: Newtonian approximation (p0 = p_inf + q_inf)
        # This is appropriate for all speed regimes and avoids normal shock
        # branching. At hypersonic conditions it matches modified Newtonian
        # theory used in Fay-Riddell's original paper for du_e/dx.
        q_inf = 0.5 * rho_inf * V ** 2  # dynamic pressure
        p_s = p_inf + q_inf
        outputs['stagnation_pressure'] = p_s

        # Post-shock stagnation conditions
        rho_s = p_s / (R_gas * T0)
        mu_s = _sutherland_viscosity(T0)

        # Wall conditions (at stagnation pressure, wall temperature)
        rho_w = p_s / (R_gas * T_w)
        mu_w = _sutherland_viscosity(T_w)

        # Velocity gradient at stagnation point for a sphere (Eq. 64)
        # du_e/dx = (1/R) * sqrt(2*(p_s - p_inf) / rho_s)
        #         = (1/R) * sqrt(2*q_inf / rho_s)
        dudx = (1.0 / R_n) * np.sqrt(2.0 * q_inf / rho_s)

        # Enthalpies
        h_s = _stagnation_enthalpy(T_inf, V)
        h_w = _specific_enthalpy(T_w)

        # Fay-Riddell Eq. 63 (equilibrium, Le=1 simplification)
        # q = 0.94 * (rho_s*mu_s)^0.4 * (rho_w*mu_w)^0.1
        #     * sqrt(du_e/dx) * (h_s - h_w)
        q = (0.94
             * (rho_s * mu_s) ** 0.4
             * (rho_w * mu_w) ** 0.1
             * np.sqrt(dudx)
             * (h_s - h_w))
        outputs['q_stag'] = q


class SuttonGravesHeating(om.ExplicitComponent):
    """
    Stagnation point convective heat flux using the Sutton-Graves (1971)
    correlation.

    From NASA TR R-376, the general form is:

        q_stag [W/cm^2] = K * sqrt(rho_inf / R_nose) * (h_s - h_w)

    where K depends on gas composition (Table II of the paper):
        Air:  K = 1.7415e-4  kg^0.5/m
        CO2:  K = 1.9027e-4
        N2:   K = 1.6746e-4

    This component converts the result to W/m^2.
    """

    def initialize(self):
        self.options.declare('gas_type', default='air',
                             values=['air', 'CO2', 'N2'],
                             desc='Gas composition for K coefficient')

    def setup(self):
        self.add_input('velocity', val=1000.0, units='m/s',
                        desc='Freestream velocity')
        self.add_input('density_inf', val=1.225, units='kg/m**3',
                        desc='Freestream density')
        self.add_input('nose_radius', val=0.1, units='m',
                        desc='Nose radius of curvature')
        self.add_input('wall_enthalpy', val=3.0e5, units='J/kg',
                        desc='Wall enthalpy')
        self.add_input('freestream_temperature', val=288.15, units='K',
                        desc='Freestream temperature for stagnation enthalpy')

        self.add_output('q_stag', val=0.0, units='W/m**2',
                         desc='Stagnation point heat flux')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        V = inputs['velocity']
        rho = inputs['density_inf']
        R_n = inputs['nose_radius']
        h_w = inputs['wall_enthalpy']
        T_inf = inputs['freestream_temperature']

        # K values from Sutton-Graves Table II (kg^0.5 / m)
        K_values = {
            'air': 1.7415e-4,
            'CO2': 1.9027e-4,
            'N2': 1.6746e-4,
        }
        K = K_values[self.options['gas_type']]

        # Stagnation enthalpy
        h_s = _stagnation_enthalpy(T_inf, V)

        # Enthalpy difference
        dh = h_s - h_w

        # q in W/cm^2, then convert to W/m^2
        q_cm2 = K * np.sqrt(rho / R_n) * dh
        q = q_cm2 * 1.0e4  # W/cm^2 -> W/m^2
        outputs['q_stag'] = q
