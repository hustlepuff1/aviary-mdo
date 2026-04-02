"""
Tactical Missile Design trajectory component.

Three-phase co-altitude flight model:
1. Boost: Rocket equation with drag (Fleeman Eq. 1)
2. Sustain: Breguet range (air-breather) or rocket equation
3. Coast: 1-DOF deceleration to terminal Mach

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001.
"""

import numpy as np
import openmdao.api as om

from .missile_aero import _atmosphere


class BoostPhase(om.ExplicitComponent):
    """
    Boost phase trajectory using the rocket equation with average drag.

    delta_V = -g * Isp * (1 - D_avg/T) * ln(1 - W_prop/W_launch)
    """

    def setup(self):
        # Inputs
        self.add_input('mach_launch', val=0.8, desc='Launch Mach number')
        self.add_input('alt', val=20000.0, units='ft', desc='Flight altitude')
        self.add_input('W_launch', val=500.0, units='lbm', desc='Launch weight')
        self.add_input('W_prop_boost', val=84.8, units='lbm',
                       desc='Boost propellant weight')
        self.add_input('thrust_boost', val=7036.0, units='lbf',
                       desc='Boost thrust')
        self.add_input('Isp_boost', val=271.0, units='s',
                       desc='Boost specific impulse')
        self.add_input('burn_time', val=3.26, units='s',
                       desc='Boost burn time')
        self.add_input('CD0_launch', val=0.31,
                       desc='Zero-lift drag at launch condition')
        self.add_input('ref_area', val=50.27, units='inch**2',
                       desc='Reference area')

        # Outputs
        self.add_output('V_launch', val=830.0, units='ft/s',
                        desc='Launch velocity')
        self.add_output('V_boost_end', val=2400.0, units='ft/s',
                        desc='Velocity at end of boost')
        self.add_output('mach_boost_end', val=2.3,
                        desc='Mach at end of boost')
        self.add_output('range_boost', val=0.87, units='nmi',
                        desc='Range during boost phase')
        self.add_output('W_boost_end', val=415.0, units='lbm',
                        desc='Weight at end of boost')
        self.add_output('CD0_boost_avg', val=0.35,
                        desc='Average CD0 during boost')
        self.add_output('time_boost', val=3.26, units='s',
                        desc='Boost phase duration (pass-through of burn_time)')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M0 = inputs['mach_launch']
        alt = inputs['alt']
        W0 = inputs['W_launch']
        Wp = inputs['W_prop_boost']
        T = inputs['thrust_boost']
        Isp = inputs['Isp_boost']
        t_burn = inputs['burn_time']
        CD0_launch = inputs['CD0_launch']
        S_ref = inputs['ref_area'] / 144.0  # convert to ft^2

        g = 32.174  # ft/s^2

        # Atmospheric conditions
        T_atm, rho, mu = _atmosphere(alt)
        a = np.sqrt(1.4 * 1716.49 * T_atm)

        V0 = M0 * a
        outputs['V_launch'] = V0

        # No-drag delta-V
        dV_no_drag = -g * Isp * np.log(1.0 - Wp / W0)

        # Estimate end-of-boost velocity (no drag first pass)
        V1_est = V0 + dV_no_drag
        M1_est = V1_est / a

        # Average drag during boost
        # Drag at launch
        q0 = 0.5 * rho * V0**2
        D0 = q0 * CD0_launch * S_ref

        # Estimate CD0 at boost end Mach (use launch CD0 as approximation
        # since we don't have the full aero model connected yet in this component)
        # In a connected system, this would come from the aero group
        CD0_end = CD0_launch * 1.1  # slight increase at higher Mach (placeholder)
        q1 = 0.5 * rho * V1_est**2
        D1 = q1 * CD0_end * S_ref

        D_avg = 0.5 * (D0 + D1)
        CD0_avg = 0.5 * (CD0_launch + CD0_end)
        outputs['CD0_boost_avg'] = CD0_avg

        # Rocket equation with drag: dV = -g*Isp*(1-D/T)*ln(1-Wp/W0)
        D_over_T = D_avg / T if T > 0 else 0.0
        dV = -g * Isp * (1.0 - D_over_T) * np.log(1.0 - Wp / W0)

        V1 = V0 + dV
        M1 = V1 / a
        outputs['V_boost_end'] = V1
        outputs['mach_boost_end'] = M1

        # Weight at end of boost
        W1 = W0 - Wp
        outputs['W_boost_end'] = W1

        # Range during boost (average velocity * time)
        V_avg = 0.5 * (V0 + V1)
        range_ft = V_avg * t_burn
        range_nmi = range_ft / 6076.12
        outputs['range_boost'] = range_nmi
        outputs['time_boost'] = t_burn


class SustainPhase(om.ExplicitComponent):
    """
    Sustain phase trajectory.

    For rocket: uses rocket equation (same as boost).
    For air-breather: uses Breguet range equation at constant velocity.
    """

    def initialize(self):
        self.options.declare('engine_type', default='rocket',
                            values=['rocket', 'ramjet'],
                            desc='Sustain engine type')

    def setup(self):
        self.add_input('mach_start', val=2.3, desc='Mach at start of sustain')
        self.add_input('alt', val=20000.0, units='ft', desc='Altitude')
        self.add_input('W_start', val=415.0, units='lbm',
                       desc='Weight at start of sustain')
        self.add_input('W_ejectables', val=0.0, units='lbm',
                       desc='Weight of ejectables discarded')
        self.add_input('W_prop_sustain', val=48.2, units='lbm',
                       desc='Sustain propellant weight')
        self.add_input('thrust_sustain', val=1119.0, units='lbf',
                       desc='Sustain thrust')
        self.add_input('Isp_sustain', val=252.0, units='s',
                       desc='Sustain specific impulse')
        self.add_input('burn_time', val=10.86, units='s',
                       desc='Sustain burn time')
        self.add_input('CD0', val=0.42, desc='Average CD0 during sustain')
        self.add_input('ref_area', val=50.27, units='inch**2',
                       desc='Reference area')

        self.add_output('V_sustain_end', val=2500.0, units='ft/s')
        self.add_output('mach_sustain_end', val=2.4)
        self.add_output('range_sustain', val=4.4, units='nmi')
        self.add_output('W_sustain_end', val=367.0, units='lbm')
        self.add_output('time_sustain', val=10.86, units='s')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M0 = inputs['mach_start']
        alt = inputs['alt']
        W0 = inputs['W_start'] - inputs['W_ejectables']
        Wp = inputs['W_prop_sustain']
        T = inputs['thrust_sustain']
        Isp = inputs['Isp_sustain']
        t_burn = inputs['burn_time']
        CD0 = inputs['CD0']
        S_ref = inputs['ref_area'] / 144.0

        g = 32.174
        T_atm, rho, mu = _atmosphere(alt)
        a = np.sqrt(1.4 * 1716.49 * T_atm)
        V0 = M0 * a

        engine_type = self.options['engine_type']

        if engine_type == 'rocket':
            # Rocket equation with drag (same as boost)
            q0 = 0.5 * rho * V0**2
            D0 = q0 * CD0 * S_ref

            dV_no_drag = -g * Isp * np.log(1.0 - Wp / W0)
            V1_est = V0 + dV_no_drag
            q1 = 0.5 * rho * V1_est**2
            D1 = q1 * CD0 * S_ref
            D_avg = 0.5 * (D0 + D1)

            D_over_T = D_avg / T if T > 0 else 0.0
            dV = -g * Isp * (1.0 - D_over_T) * np.log(1.0 - Wp / W0)
            V1 = V0 + dV

            V_avg = 0.5 * (V0 + V1)
            range_ft = V_avg * t_burn

        else:
            # Ramjet / air-breather: Breguet range equation
            # Constant velocity cruise
            V1 = V0  # constant velocity for air-breather

            # Need L/D for Breguet
            q = 0.5 * rho * V0**2
            D = q * CD0 * S_ref
            # For co-altitude level flight: L = W, so L/D = W/D
            W_avg = W0 - Wp / 2.0
            L_over_D = W_avg / D if D > 0 else 10.0

            # Breguet: R = (L/D) * Isp * V * ln(W0/W1) / g
            W1_temp = W0 - Wp
            range_ft = L_over_D * Isp * V0 * np.log(W0 / W1_temp)
            V1 = V0

        W1 = W0 - Wp
        M1 = V1 / a

        outputs['V_sustain_end'] = V1
        outputs['mach_sustain_end'] = M1
        outputs['range_sustain'] = range_ft / 6076.12  # nmi
        outputs['W_sustain_end'] = W1
        outputs['time_sustain'] = t_burn


class CoastPhase(om.ExplicitComponent):
    """
    Coast phase: unpowered deceleration to terminal Mach.

    Uses 1-DOF model with average drag to estimate coast time and distance.
    Stops when Mach reaches the terminal Mach criterion.
    """

    def setup(self):
        self.add_input('mach_start', val=2.4, desc='Mach at start of coast')
        self.add_input('mach_terminal', val=1.5, desc='Terminal Mach criterion')
        self.add_input('alt', val=20000.0, units='ft', desc='Altitude')
        self.add_input('W_coast', val=367.0, units='lbm',
                       desc='Weight during coast (constant)')
        self.add_input('CD0', val=0.46, desc='Average CD0 during coast')
        self.add_input('ref_area', val=50.27, units='inch**2',
                       desc='Reference area')

        self.add_output('range_coast', val=5.4, units='nmi',
                        desc='Coast range')
        self.add_output('time_coast', val=17.0, units='s',
                        desc='Coast time')
        self.add_output('V_coast_end', val=1555.0, units='ft/s',
                        desc='Velocity at end of coast')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M0 = inputs['mach_start']
        M_term = inputs['mach_terminal']
        alt = inputs['alt']
        W = inputs['W_coast']
        CD0 = inputs['CD0']
        S_ref = inputs['ref_area'] / 144.0

        g = 32.174
        T_atm, rho, mu = _atmosphere(alt)
        a = np.sqrt(1.4 * 1716.49 * T_atm)

        V0 = M0 * a
        V_end = M_term * a
        outputs['V_coast_end'] = V_end

        # 1-DOF coast: m * dV/dt = -D = -0.5 * rho * V^2 * CD0 * S
        # Analytical solution: V(t) = V0 / (1 + t * rho*V0*CD0*S / (2*W/g))
        # Time to reach V_end:
        # V_end = V0 / (1 + t_coast * rho*V0*CD0*S*g / (2*W))
        # t_coast = (V0/V_end - 1) * 2*W / (rho*V0*CD0*S*g)
        mass_slug = W / g
        k = 0.5 * rho * CD0 * S_ref / mass_slug  # 1/ft

        if k > 0 and V0 > V_end:
            t_coast = (1.0 / V_end - 1.0 / V0) / k
        else:
            t_coast = 0.0

        outputs['time_coast'] = t_coast

        # Range: integral of V(t) dt from 0 to t_coast
        # R = (1/k) * ln(1 + k*V0*t_coast)
        if k > 0 and t_coast > 0:
            range_ft = (1.0 / k) * np.log(1.0 + k * V0 * t_coast)
        else:
            range_ft = 0.0

        outputs['range_coast'] = range_ft / 6076.12


class MissileTrajectoryGroup(om.Group):
    """
    Complete three-phase missile trajectory.

    Assembles boost, sustain, and coast into a sequential trajectory.
    Outputs total range and time-to-target.
    """

    def initialize(self):
        self.options.declare('sustain_engine', default='rocket',
                            values=['rocket', 'ramjet'])

    def setup(self):
        self.add_subsystem('boost', BoostPhase(),
                           promotes_inputs=['alt', 'ref_area'])

        self.add_subsystem('sustain',
                           SustainPhase(engine_type=self.options['sustain_engine']),
                           promotes_inputs=['alt', 'ref_area'])

        self.add_subsystem('coast', CoastPhase(),
                           promotes_inputs=['alt', 'ref_area'])

        # Connect phases
        self.connect('boost.mach_boost_end', 'sustain.mach_start')
        self.connect('boost.W_boost_end', 'sustain.W_start')
        self.connect('sustain.mach_sustain_end', 'coast.mach_start')
        self.connect('sustain.W_sustain_end', 'coast.W_coast')

        # Total range and time
        self.add_subsystem('totals',
                           om.ExecComp([
                               'range_total = range_boost + range_sustain + range_coast',
                               'time_total = time_boost + time_sustain + time_coast',
                           ],
                               range_total={'units': 'nmi'},
                               range_boost={'units': 'nmi'},
                               range_sustain={'units': 'nmi'},
                               range_coast={'units': 'nmi'},
                               time_total={'units': 's'},
                               time_boost={'units': 's'},
                               time_sustain={'units': 's'},
                               time_coast={'units': 's'},
                           ),
                           promotes_outputs=['range_total', 'time_total'])

        self.connect('boost.range_boost', 'totals.range_boost')
        self.connect('sustain.range_sustain', 'totals.range_sustain')
        self.connect('coast.range_coast', 'totals.range_coast')
        # burn_time is an input, not output — pass it through an IndepVarComp
        # or just use direct connections from sustain outputs
        self.connect('boost.time_boost', 'totals.time_boost')
        self.connect('sustain.time_sustain', 'totals.time_sustain')
        self.connect('coast.time_coast', 'totals.time_coast')
