"""Missile ODE Group for Dymos trajectory optimization.

Assembles atmosphere, aero, and EOM components into a single ODE group
that Dymos can use as the ode_class for a Phase.

Usage with Dymos:
    import dymos as dm
    from aviary_mdo.missile_6dof.model.dymos_ode import MissileODE

    phase = dm.Phase(
        ode_class=MissileODE,
        ode_init_kwargs={'S_ref': 0.0324, 'CD0': 0.3},
        transcription=dm.GaussLobatto(num_segments=20, order=3),
    )
    phase.set_time_options(fix_initial=True, duration_bounds=(1, 300))
    phase.add_state('velocity', rate_source='eom.velocity_rate', ...)
    phase.add_state('flight_path_angle', rate_source='eom.flight_path_angle_rate', ...)
    phase.add_state('altitude', rate_source='eom.altitude_rate', ...)
    phase.add_state('range', rate_source='eom.range_rate', ...)
    phase.add_state('mass', rate_source='eom.mass_rate', ...)
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_6dof.model.dymos_ode.missile_atmos_comp import MissileAtmosComp
from aviary_mdo.missile_6dof.model.dymos_ode.missile_aero_comp import MissileAeroComp
from aviary_mdo.missile_6dof.model.dymos_ode.missile_eom import MissileEOM


class MissileThrustComp(om.ExplicitComponent):
    """Simple thrust model: constant thrust during burn, zero after.

    Uses a smooth tanh transition to maintain differentiability.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('thrust_magnitude', val=10000.0, units='N',
                        desc='Thrust level during burn')
        self.add_input('burn_time', val=3.0, units='s',
                        desc='Motor burn duration')
        self.add_input('mass_flow_rate_burn', val=5.0, units='kg/s',
                        desc='Mass flow rate during burn')

        self.add_output('thrust', val=np.zeros(nn), units='N')
        self.add_output('mass_flow_rate', val=np.zeros(nn), units='kg/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.declare_partials('thrust', ['time'], rows=ar, cols=ar)
        self.declare_partials('thrust', ['thrust_magnitude', 'burn_time'])
        self.declare_partials('mass_flow_rate', ['time'], rows=ar, cols=ar)
        self.declare_partials('mass_flow_rate', ['mass_flow_rate_burn', 'burn_time'])

    def compute(self, inputs, outputs):
        t = inputs['time']
        T_mag = inputs['thrust_magnitude']
        t_burn = inputs['burn_time']
        mdot_burn = inputs['mass_flow_rate_burn']

        # Smooth step: sigma(k*(t_burn - t)), k controls sharpness
        k = 20.0  # sharpness of transition
        sigma = 0.5 * (1.0 + np.tanh(k * (t_burn - t)))

        outputs['thrust'] = T_mag * sigma
        outputs['mass_flow_rate'] = mdot_burn * sigma

    def compute_partials(self, inputs, J):
        t = inputs['time']
        T_mag = inputs['thrust_magnitude']
        t_burn = inputs['burn_time']
        mdot_burn = inputs['mass_flow_rate_burn']

        k = 20.0
        arg = k * (t_burn - t)
        sigma = 0.5 * (1.0 + np.tanh(arg))
        dsigma_darg = 0.5 * (1.0 - np.tanh(arg) ** 2)
        dsigma_dt = dsigma_darg * (-k)
        dsigma_dtburn = dsigma_darg * k

        J['thrust', 'time'] = T_mag * dsigma_dt
        J['thrust', 'thrust_magnitude'] = sigma
        J['thrust', 'burn_time'] = T_mag * dsigma_dtburn

        J['mass_flow_rate', 'time'] = mdot_burn * dsigma_dt
        J['mass_flow_rate', 'burn_time'] = mdot_burn * dsigma_dtburn
        J['mass_flow_rate', 'mass_flow_rate_burn'] = sigma


class MissileODE(om.Group):
    """Complete missile ODE for Dymos.

    Assembles: atmosphere -> aero -> thrust -> EOM

    All components have analytic partials. No finite differences.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('S_ref', default=0.0324, types=float,
                             desc='Reference area (m^2)')
        self.options.declare('CD0', default=0.3, types=float,
                             desc='Zero-lift drag coefficient')
        self.options.declare('CN_alpha', default=2.0, types=float,
                             desc='Normal force slope (1/rad)')

    def setup(self):
        nn = self.options['num_nodes']

        # 1. Atmosphere
        self.add_subsystem('atmos', MissileAtmosComp(num_nodes=nn),
                           promotes_inputs=['altitude'],
                           promotes_outputs=['density', 'speed_of_sound',
                                             'temperature'])

        # 2. Aero
        self.add_subsystem('aero', MissileAeroComp(num_nodes=nn),
                           promotes_inputs=['velocity', 'density',
                                            'speed_of_sound', 'alpha'],
                           promotes_outputs=['drag', 'lift', 'mach', 'q_inf'])

        # 3. Thrust
        self.add_subsystem('prop', MissileThrustComp(num_nodes=nn),
                           promotes_inputs=[('time', 'time'),
                                            'thrust_magnitude',
                                            'burn_time',
                                            'mass_flow_rate_burn'],
                           promotes_outputs=['thrust', 'mass_flow_rate'])

        # 4. Equations of motion
        self.add_subsystem('eom', MissileEOM(num_nodes=nn),
                           promotes_inputs=['velocity', 'flight_path_angle',
                                            'mass', 'thrust', 'drag', 'lift',
                                            'alpha', 'mass_flow_rate'],
                           promotes_outputs=['*'])

        # Set aero parameters as defaults from options
        self.set_input_defaults('aero.S_ref', self.options['S_ref'], units='m**2')
        self.set_input_defaults('aero.CD0', self.options['CD0'])
        self.set_input_defaults('aero.CN_alpha', self.options['CN_alpha'],
                                units='1/rad')

        # Resolve promoted input ambiguities (multiple subsystems share these)
        self.set_input_defaults('velocity', np.ones(nn) * 300.0, units='m/s')
        self.set_input_defaults('alpha', np.zeros(nn), units='rad')
        self.set_input_defaults('density', np.ones(nn) * 1.225, units='kg/m**3')
        self.set_input_defaults('speed_of_sound', np.ones(nn) * 340.3, units='m/s')
