"""
Surface heat flux distribution over a blunt-body vehicle.

Given the stagnation-point heat flux q_stag, computes the heat flux at
downstream body stations using:

1. **Nose region** (x <= R_nose): stagnation-region analogy
       q(x) / q_stag = cos(theta),  theta = arcsin(x / R_nose)

2. **Downstream flat-plate region** (x > R_nose): Eckert reference-temperature
   method with laminar or turbulent correlations depending on local Re_x vs
   a user-specified transition Reynolds number.

   Laminar:
       q_lam = 0.332 * k_ref / x * sqrt(Re_x_ref) * Pr_ref^(1/3) * (T_aw - T_w)

   Turbulent:
       q_turb = 0.0296 * k_ref / x * Re_x_ref^(4/5) * Pr_ref^(1/3) * (T_aw - T_w)

   Reference temperature (Eckert):
       T_ref = T_e * (1 + 0.032*M_e^2 + 0.58*(T_w/T_e - 1))

   Adiabatic wall temperature:
       T_aw = T_e * (1 + r*(gamma-1)/2 * M_e^2),  r = sqrt(Pr) for laminar

References
----------
- Fay & Riddell, J. Aero. Sci., 1958.
- Eckert, E.R.G., "Engineering Relations for Friction and Heat Transfer
  to Surfaces in High Velocity Flow," J. Aero. Sci., 1955.
"""

import numpy as np
import openmdao.api as om


class SurfaceHeatFlux(om.ExplicitComponent):
    """
    Heat flux distribution along a blunt-body surface.

    Options
    -------
    num_stations : int
        Number of equally-spaced body stations (default 10).
    """

    def initialize(self):
        self.options.declare('num_stations', default=10, types=int,
                             desc='Number of body stations')

    def setup(self):
        nn = self.options['num_stations']

        # Scalar inputs
        self.add_input('q_stag', val=1e6, units='W/m**2',
                       desc='Stagnation point heat flux')
        self.add_input('velocity', val=2000.0, units='m/s',
                       desc='Freestream velocity')
        self.add_input('altitude', val=30000.0, units='m',
                       desc='Flight altitude')
        self.add_input('nose_radius', val=0.1, units='m',
                       desc='Nose radius')
        self.add_input('wall_temperature', val=400.0, units='K',
                       desc='Wall temperature')
        self.add_input('body_length', val=5.0, units='m',
                       desc='Total body length')
        self.add_input('body_diameter', val=0.5, units='m',
                       desc='Body diameter for area integration')
        self.add_input('Re_transition', val=5e5,
                       desc='Transition Reynolds number')

        # Array outputs
        self.add_output('q_distribution', val=np.zeros(nn), units='W/m**2',
                        desc='Heat flux at each body station')
        self.add_output('x_stations', val=np.zeros(nn), units='m',
                        desc='Body station locations')
        self.add_output('q_max', val=0.0, units='W/m**2',
                        desc='Maximum heat flux along the body')
        self.add_output('total_heat_load', val=0.0, units='W',
                        desc='Integrated heat load over body surface')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_stations']

        # Extract scalars (OpenMDAO inputs are shape-(1,) arrays)
        q_stag = inputs['q_stag'].item()
        V = inputs['velocity'].item()
        alt = inputs['altitude'].item()
        R_nose = inputs['nose_radius'].item()
        T_w = inputs['wall_temperature'].item()
        L = inputs['body_length'].item()
        D = inputs['body_diameter'].item()
        Re_trans = inputs['Re_transition'].item()

        # Body stations (avoid x=0 singularity by starting at small offset)
        x_start = L / (2.0 * nn)
        x_stations = np.linspace(x_start, L, nn)
        outputs['x_stations'] = x_stations

        # Handle zero-velocity edge case
        if np.real(V) <= 0.0:
            outputs['q_distribution'] = np.zeros(nn)
            outputs['q_max'] = 0.0
            outputs['total_heat_load'] = 0.0
            return

        # Atmospheric properties (SI)
        T_e, rho, mu, k_air = _atmosphere_si(alt)

        # Freestream derived quantities
        gamma = 1.4
        R_gas = 287.058  # J/(kg*K)
        a = np.sqrt(gamma * R_gas * T_e)
        M_e = V / a
        Pr = 0.71  # Prandtl number for air

        # Adiabatic wall temperature
        r_lam = np.sqrt(Pr)  # recovery factor (laminar)
        T_aw = T_e * (1.0 + r_lam * (gamma - 1.0) / 2.0 * M_e**2)

        # Reference temperature (Eckert)
        T_ref = T_e * (1.0 + 0.032 * M_e**2 + 0.58 * (T_w / T_e - 1.0))
        # Ensure T_ref stays physical
        if np.real(T_ref) < 50.0:
            T_ref = 50.0

        # Reference properties via Sutherland's law
        mu_ref = 1.458e-6 * T_ref**1.5 / (T_ref + 110.4)
        # Thermal conductivity: k = mu * cp / Pr, cp ~ 1005 J/(kg*K)
        k_ref = mu_ref * 1005.0 / Pr
        # Reference density from ideal gas at reference temperature
        p_inf = rho * R_gas * T_e
        rho_ref = p_inf / (R_gas * T_ref)

        # Compute heat flux at each station
        use_complex = np.iscomplexobj(q_stag) or np.iscomplexobj(V)
        q = np.zeros(nn, dtype=complex if use_complex else float)

        for i in range(nn):
            x = x_stations[i]
            if np.real(x) <= np.real(R_nose):
                # Nose region: stagnation analogy
                ratio = x / R_nose
                if np.real(ratio) > 1.0:
                    ratio = 1.0
                theta = np.arcsin(ratio)
                q[i] = q_stag * np.cos(theta)
            else:
                # Downstream flat-plate region
                Re_x = rho_ref * V * x / mu_ref

                dT = T_aw - T_w
                if np.real(dT) < 0.0:
                    dT = 0.0

                if np.real(Re_x) < np.real(Re_trans):
                    # Laminar
                    q[i] = (0.332 * k_ref / x
                            * np.sqrt(Re_x)
                            * Pr**(1.0 / 3.0)
                            * dT)
                else:
                    # Turbulent
                    q[i] = (0.0296 * k_ref / x
                            * Re_x**0.8
                            * Pr**(1.0 / 3.0)
                            * dT)

        outputs['q_distribution'] = q
        idx_max = np.argmax(np.real(q))
        outputs['q_max'] = q[idx_max]

        # Total heat load: trapezoidal integration over cylindrical body
        # dA = pi * D * dx  (circumferential strip)
        dx = np.diff(x_stations)
        q_avg = 0.5 * (q[:-1] + q[1:])
        outputs['total_heat_load'] = np.sum(q_avg * np.pi * D * dx)


def _atmosphere_si(alt_m):
    """
    Simplified 1976 US Standard Atmosphere (SI units).

    Parameters
    ----------
    alt_m : float
        Altitude in metres.

    Returns
    -------
    T : float
        Temperature (K).
    rho : float
        Density (kg/m^3).
    mu : float
        Dynamic viscosity (Pa*s).
    k : float
        Thermal conductivity (W/(m*K)).
    """
    if np.real(alt_m) < 0.0:
        alt_m = 0.0

    if np.real(alt_m) <= 11000.0:
        # Troposphere
        T = 288.15 - 0.0065 * alt_m
        p = 101325.0 * (T / 288.15)**5.2561
    else:
        # Lower stratosphere (isothermal up to ~20 km)
        T = 216.65
        p = 101325.0 * 0.22336 * np.exp(-0.00015769 * (alt_m - 11000.0))

    rho = p / (287.058 * T)

    # Sutherland's law
    mu = 1.458e-6 * T**1.5 / (T + 110.4)

    # Thermal conductivity of air (approximate)
    k = mu * 1005.0 / 0.71  # k = mu * cp / Pr

    return T, rho, mu, k
