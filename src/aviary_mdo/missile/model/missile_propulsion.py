"""
Tactical Missile Design propulsion component.

Implements solid rocket motor analysis and ramjet cycle analysis
from Fleeman's TMD methodology.

Rocket: Given Pc, expansion ratio, fuel type -> Cf, c*, Isp, thrust, mass flow
Ramjet: Given fuel-air ratio, T4_max, flight condition -> thrust, Isp

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001.
"""

import numpy as np
import openmdao.api as om


# Fuel property database from TMD spreadsheet (Propulsion sheet, rows 4-14)
FUEL_DATABASE = {
    # type_id: (name, density_lbm_in3, gamma, c_star_ft_s)
    1: ('HMX', 0.0686, 1.217, 5298.0),
    2: ('RDX', 0.0656, 1.230, 5258.0),
    3: ('CL20', 0.0712, 1.210, 5350.0),
    4: ('High Smoke Composite', 0.0619, 1.217, 5298.0),
    5: ('Reduced Smoke Composite', 0.0609, 1.220, 5200.0),
}

# Air-breathing fuel database
AIR_FUEL_DATABASE = {
    # type_id: (name, density_lbm_in3, heating_value_ft_lbf_per_lbm, cp_btu_per_lbm)
    6: ('JP-n', 0.028, 15535588.25, 0.3),
    7: ('RJ-n', 0.040, 11302904.725, 0.3),
    8: ('Carbon Slurry', 0.049, 12720681.0, 0.3),
    9: ('Solid Hydrocarbon', 0.075, 11745164.107, 0.3),
    10: ('Aluminum Slurry', 0.072, 9359643.806, 0.3),
}

# Engine type lookup Isp tables (Mach-dependent for non-rocket types)
ENGINE_ISP_LOOKUP = {
    # type: {mach: Isp} (simplified from TMD spreadsheet)
    5: {0.5: 3500, 1.0: 3200, 1.5: 2943, 2.0: 2700, 2.5: 2500},  # turbofan high
    6: {0.5: 2800, 1.0: 2500, 1.5: 2246, 2.0: 2000, 2.5: 1800},  # turbofan mid
    7: {0.5: 2200, 1.0: 1900, 1.5: 1653, 2.0: 1400, 2.5: 1200},  # turbofan low
}


class RocketMotor(om.ExplicitComponent):
    """
    Solid rocket motor performance calculation.

    Computes thrust coefficient, specific impulse, thrust, mass flow,
    throat area, and exit area from chamber pressure, expansion ratio,
    and fuel properties.
    """

    def setup(self):
        self.add_input('Pc', val=1769.0, units='lbf/inch**2',
                       desc='Chamber pressure')
        self.add_input('expansion_ratio', val=6.0,
                       desc='Nozzle area expansion ratio Ae/At')
        self.add_input('fuel_type', val=4.0,
                       desc='Fuel type designation (1-5)')
        self.add_input('propellant_weight', val=84.8, units='lbm',
                       desc='Propellant weight')
        self.add_input('burn_time', val=3.26, units='s',
                       desc='Motor burn time')
        self.add_input('Pa', val=6.76, units='lbf/inch**2',
                       desc='Ambient pressure')

        self.add_output('gamma', val=1.217,
                        desc='Specific heat ratio of combustion products')
        self.add_output('Cf', val=1.0,
                        desc='Thrust coefficient')
        self.add_output('c_star', val=5298.0, units='ft/s',
                        desc='Characteristic exhaust velocity')
        self.add_output('Isp', val=250.0, units='s',
                        desc='Specific impulse')
        self.add_output('thrust', val=7000.0, units='lbf',
                        desc='Motor thrust')
        self.add_output('mass_flow', val=26.0, units='lbm/s',
                        desc='Propellant mass flow rate')
        self.add_output('throat_area', val=2.42, units='inch**2',
                        desc='Nozzle throat area')
        self.add_output('exit_area', val=14.52, units='inch**2',
                        desc='Nozzle exit area')
        self.add_output('exit_pressure', val=6.0, units='lbf/inch**2',
                        desc='Nozzle exit pressure')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Pc = inputs['Pc']
        eps = inputs['expansion_ratio']
        fuel_id = int(np.round(inputs['fuel_type']))
        W_prop = inputs['propellant_weight']
        t_burn = inputs['burn_time']
        Pa = inputs['Pa']

        # Look up fuel properties
        if fuel_id in FUEL_DATABASE:
            _, _, gamma, c_star = FUEL_DATABASE[fuel_id]
        else:
            gamma = 1.217
            c_star = 5298.0

        outputs['gamma'] = gamma
        outputs['c_star'] = c_star

        g = gamma

        # Exit pressure ratio Pe/Pc from isentropic expansion
        # Solve A/A* = (1/M)*((2/(g+1))*(1+(g-1)/2*M^2))^((g+1)/(2*(g-1)))
        # For a given expansion ratio, find exit Mach, then Pe/Pc
        Me = _exit_mach_from_area_ratio(eps, gamma)
        Pe_Pc = (1.0 + (g - 1.0) / 2.0 * Me**2)**(- g / (g - 1.0))
        Pe = Pe_Pc * Pc
        outputs['exit_pressure'] = Pe

        # Thrust coefficient
        # Cf = sqrt(2*g^2/(g-1) * (2/(g+1))^((g+1)/(g-1)) * (1-(Pe/Pc)^((g-1)/g)))
        #      + (Pe - Pa)/Pc * eps
        term1 = (2.0 * g**2 / (g - 1.0))
        term2 = (2.0 / (g + 1.0))**((g + 1.0) / (g - 1.0))
        term3 = 1.0 - Pe_Pc**((g - 1.0) / g)
        Cf = np.sqrt(term1 * term2 * term3) + (Pe - Pa) / Pc * eps

        outputs['Cf'] = Cf

        # Isp = Cf * c_star / g0
        g0 = 32.174  # ft/s^2
        Isp = Cf * c_star / g0
        outputs['Isp'] = Isp

        # Mass flow rate
        mdot = W_prop / t_burn
        outputs['mass_flow'] = mdot

        # Thrust = mdot * Isp * g0 / g0 = mdot * c_star * Cf / g0
        # Or more simply: F = Cf * Pc * At
        # From mdot = Pc * At / c_star -> At = mdot * c_star / Pc
        # Need consistent units: c_star in ft/s, Pc in lbf/in^2
        # At = mdot * c_star / (Pc * g0)  [in^2]
        # Actually: mdot (lbm/s), c_star (ft/s), Pc (psi)
        # At = mdot * c_star / (Pc * g0) [ft^2] ... need to convert
        # Use: At = W_prop / (t_burn * Pc) * c_star / g0 ... in slug units
        # Simpler: thrust = Isp * mdot (in lbf when Isp in sec, mdot in lbm/s)
        thrust = Isp * mdot
        outputs['thrust'] = thrust

        # Throat area: At = mdot * c_star / (Pc * g0)
        # Units: (lbm/s * ft/s) / (lbf/in^2 * ft/s^2) needs careful handling
        # F = Cf * Pc * At -> At = F / (Cf * Pc)
        if Cf * Pc > 0:
            At = thrust / (Cf * Pc)  # in^2
        else:
            At = 1.0
        outputs['throat_area'] = At

        # Exit area
        Ae = At * eps
        outputs['exit_area'] = Ae


class RamjetMotor(om.ExplicitComponent):
    """
    Ramjet cycle analysis for cruise/sustain phase.

    Simple cycle: inlet ram compression -> combustor heat addition -> nozzle expansion.
    Computes thrust and Isp from flight condition and fuel properties.
    """

    def setup(self):
        self.add_input('mach', val=2.5, desc='Flight Mach number')
        self.add_input('alt', val=40000.0, units='ft', desc='Altitude')
        self.add_input('fuel_air_ratio', val=0.06, desc='Fuel-to-air ratio')
        self.add_input('T4_max', val=4500.0, units='degR',
                       desc='Maximum combustor temperature')
        self.add_input('fuel_type', val=6.0, desc='Air-breathing fuel type (6-10)')
        self.add_input('expansion_ratio', val=6.2, desc='Nozzle expansion ratio')
        self.add_input('ref_area', val=326.0, units='inch**2',
                       desc='Inlet capture area / reference area')

        self.add_output('thrust', val=0.0, units='lbf', desc='Ramjet thrust')
        self.add_output('Isp', val=0.0, units='s', desc='Specific impulse')
        self.add_output('mass_flow_fuel', val=0.0, units='lbm/s',
                        desc='Fuel mass flow rate')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['mach']
        alt = inputs['alt']
        f = inputs['fuel_air_ratio']
        T4_max = inputs['T4_max']
        fuel_id = int(np.round(inputs['fuel_type']))
        eps = inputs['expansion_ratio']

        from .missile_aero import _atmosphere
        T_atm, rho, mu = _atmosphere(alt)

        gamma = 1.4  # air
        g0 = 32.174
        gc = 32.174

        a_sound = np.sqrt(gamma * 1716.49 * T_atm)  # ft/s
        V = M * a_sound
        P_atm = rho * 1716.49 * T_atm  # psf

        # Stagnation conditions at inlet
        T0 = T_atm * (1.0 + (gamma - 1.0) / 2.0 * M**2)

        # Inlet pressure recovery (MIL-E-5007 approximation)
        if M <= 1.0:
            eta_inlet = 1.0
        else:
            eta_inlet = 1.0 - 0.075 * (M - 1.0)**1.35

        P0 = P_atm * (1.0 + (gamma - 1.0) / 2.0 * M**2)**(gamma / (gamma - 1.0))
        P02 = eta_inlet * P0

        # Combustor: heat addition to T4
        T4 = min(T4_max, T4_max)

        # Combustion products gamma (simplified)
        gamma_c = 1.3  # hot gas gamma

        # Nozzle exit conditions (isentropic from P02 to P_atm)
        # P02/Pe = (1 + (gc-1)/2 * Me^2)^(gc/(gc-1))
        # For fully expanded: Pe = P_atm
        P_ratio = P02 / P_atm
        if P_ratio > 1.0:
            Me = np.sqrt(2.0 / (gamma_c - 1.0) * (P_ratio**((gamma_c - 1.0) / gamma_c) - 1.0))
        else:
            Me = 0.0

        Te = T4 / (1.0 + (gamma_c - 1.0) / 2.0 * Me**2)
        Ve = Me * np.sqrt(gamma_c * 1716.49 * Te)  # exit velocity, ft/s

        # Air mass flow (through inlet capture)
        # Simplified: mdot_air = rho * V * A_capture
        A_capture_ft2 = inputs['ref_area'] / 144.0  # in^2 to ft^2
        mdot_air = rho * V * A_capture_ft2  # slugs/s
        mdot_air_lbm = mdot_air * gc  # lbm/s

        # Fuel mass flow
        mdot_fuel = f * mdot_air_lbm  # lbm/s
        outputs['mass_flow_fuel'] = mdot_fuel

        # Thrust = (mdot_air + mdot_fuel) * Ve - mdot_air * V + (Pe - Pa)*Ae
        # For fully expanded nozzle, Pe = Pa, so pressure term is ~0
        mdot_total_slug = mdot_air + mdot_fuel / gc
        thrust = mdot_total_slug * Ve - mdot_air * V  # lbf
        outputs['thrust'] = max(thrust, 0.0)

        # Isp = F / (mdot_fuel * g0)
        if mdot_fuel > 1e-10:
            Isp = max(thrust, 0.0) / mdot_fuel
        else:
            Isp = 0.0
        outputs['Isp'] = Isp


def _exit_mach_from_area_ratio(eps, gamma):
    """
    Compute supersonic exit Mach number from area ratio using Newton iteration.

    Parameters
    ----------
    eps : float
        Nozzle area ratio Ae/At (>= 1.0).
    gamma : float
        Specific heat ratio.

    Returns
    -------
    Me : float
        Exit Mach number (supersonic branch).
    """
    g = gamma
    gp1 = g + 1.0
    gm1 = g - 1.0
    exp = gp1 / (2.0 * gm1)

    # Initial guess (approximation for supersonic branch)
    Me = 1.0 + 0.27 * (eps - 1.0)
    if Me < 1.01:
        Me = 1.01

    for _ in range(100):
        t = 1.0 + gm1 / 2.0 * Me**2
        A_ratio = (1.0 / Me) * ((2.0 / gp1) * t)**exp

        residual = A_ratio - eps
        if abs(residual) < 1e-12:
            break

        # dA/dM analytically:
        # A/A* = (1/M) * f(M), so dA/dM = -A/(M^2) * M + (1/M)*df/dM
        # Simpler: use logarithmic derivative
        # ln(A/A*) = -ln(M) + exp * ln(2/(g+1) * (1 + (g-1)/2*M^2))
        # d/dM [ln(A/A*)] = -1/M + exp * (g-1)*M / (1 + (g-1)/2*M^2)
        dlogA_dM = -1.0 / Me + exp * gm1 * Me / t
        dA_dM = A_ratio * dlogA_dM

        if abs(dA_dM) < 1e-15:
            break
        Me = Me - residual / dA_dM
        Me = max(Me, 1.001)

    return Me
