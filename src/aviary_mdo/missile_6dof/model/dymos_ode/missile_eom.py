"""Missile equations of motion for Dymos trajectory optimization.

3-DOF point-mass EOM in a vertical plane (range-altitude):
    States: velocity (V), flight_path_angle (gamma), altitude (h),
            range (x), mass (m)
    Control: angle of attack (alpha) — enters via aero component

Equations:
    dV/dt     = (T*cos(alpha) - D) / m - g*sin(gamma)
    dgamma/dt = (T*sin(alpha) + L) / (m*V) - (g/V)*cos(gamma)
    dh/dt     = V * sin(gamma)
    dx/dt     = V * cos(gamma)
    dm/dt     = -mdot  (if t < burn_time, else 0)

All inputs/outputs vectorized over num_nodes.
Hand-coded analytic partials for full gradient compatibility with Dymos.
"""

import numpy as np
import openmdao.api as om


class MissileEOM(om.ExplicitComponent):
    """3-DOF missile equations of motion.

    Follows the Aviary FlightPathEOM pattern: vectorized, analytic partials,
    Dymos state rate tags for automatic state wiring.
    """

    G0 = 9.80665  # m/s^2

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # ----- State inputs -----
        self.add_input('velocity', val=300.0 * np.ones(nn), units='m/s',
                        desc='Missile speed')
        self.add_input('flight_path_angle', val=np.zeros(nn), units='rad',
                        desc='Flight path angle')
        self.add_input('mass', val=200.0 * np.ones(nn), units='kg',
                        desc='Current missile mass')

        # ----- Force inputs (from aero + propulsion) -----
        self.add_input('thrust', val=np.zeros(nn), units='N',
                        desc='Thrust magnitude')
        self.add_input('drag', val=np.zeros(nn), units='N',
                        desc='Drag force')
        self.add_input('lift', val=np.zeros(nn), units='N',
                        desc='Lift force (from alpha)')
        self.add_input('alpha', val=np.zeros(nn), units='rad',
                        desc='Angle of attack')
        self.add_input('mass_flow_rate', val=np.zeros(nn), units='kg/s',
                        desc='Propellant mass flow rate (positive = burning)')

        # ----- State rate outputs (tagged for Dymos) -----
        self.add_output('velocity_rate', val=np.zeros(nn), units='m/s**2',
                         tags=['dymos.state_rate_source:velocity',
                               'dymos.state_units:m/s'])
        self.add_output('flight_path_angle_rate', val=np.zeros(nn), units='rad/s',
                         tags=['dymos.state_rate_source:flight_path_angle',
                               'dymos.state_units:rad'])
        self.add_output('altitude_rate', val=np.zeros(nn), units='m/s',
                         tags=['dymos.state_rate_source:altitude',
                               'dymos.state_units:m'])
        self.add_output('range_rate', val=np.zeros(nn), units='m/s',
                         tags=['dymos.state_rate_source:range',
                               'dymos.state_units:m'])
        self.add_output('mass_rate', val=np.zeros(nn), units='kg/s',
                         tags=['dymos.state_rate_source:mass',
                               'dymos.state_units:kg'])

        # ----- Auxiliary outputs -----
        self.add_output('load_factor', val=np.ones(nn),
                         desc='Normal load factor (g-units)')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        # velocity_rate depends on: thrust, drag, alpha, mass, flight_path_angle
        self.declare_partials('velocity_rate',
            ['thrust', 'drag', 'alpha', 'mass', 'flight_path_angle'],
            rows=ar, cols=ar)

        # flight_path_angle_rate depends on: thrust, lift, alpha, mass, velocity, flight_path_angle
        self.declare_partials('flight_path_angle_rate',
            ['thrust', 'lift', 'alpha', 'mass', 'velocity', 'flight_path_angle'],
            rows=ar, cols=ar)

        # altitude_rate depends on: velocity, flight_path_angle
        self.declare_partials('altitude_rate',
            ['velocity', 'flight_path_angle'], rows=ar, cols=ar)

        # range_rate depends on: velocity, flight_path_angle
        self.declare_partials('range_rate',
            ['velocity', 'flight_path_angle'], rows=ar, cols=ar)

        # mass_rate depends on: mass_flow_rate
        self.declare_partials('mass_rate', 'mass_flow_rate', rows=ar, cols=ar,
                              val=-np.ones(nn))

        # load_factor depends on: lift, thrust, alpha, mass, flight_path_angle
        self.declare_partials('load_factor',
            ['lift', 'thrust', 'alpha', 'mass', 'flight_path_angle'],
            rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        g = self.G0
        V = inputs['velocity']
        gamma = inputs['flight_path_angle']
        m = inputs['mass']
        T = inputs['thrust']
        D = inputs['drag']
        L = inputs['lift']
        alpha = inputs['alpha']
        mdot = inputs['mass_flow_rate']

        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        # Clamp velocity to avoid division by zero
        V_safe = np.maximum(V, 1.0)

        # dV/dt = (T*cos(alpha) - D)/m - g*sin(gamma)
        outputs['velocity_rate'] = (T * cos_a - D) / m - g * sin_g

        # dgamma/dt = (T*sin(alpha) + L)/(m*V) - (g/V)*cos(gamma)
        outputs['flight_path_angle_rate'] = (
            (T * sin_a + L) / (m * V_safe) - (g / V_safe) * cos_g
        )

        # dh/dt = V*sin(gamma)
        outputs['altitude_rate'] = V * sin_g

        # dx/dt = V*cos(gamma)
        outputs['range_rate'] = V * cos_g

        # dm/dt = -mdot
        outputs['mass_rate'] = -mdot

        # Load factor: n = sqrt(L^2 + (T*sin(alpha))^2) / (m*g)
        # Simplified: n = (L + T*sin(alpha)) / (m*g*cos(gamma))
        weight_cos_g = m * g * np.maximum(cos_g, 0.01)
        outputs['load_factor'] = (L + T * sin_a) / weight_cos_g

    def compute_partials(self, inputs, J):
        g = self.G0
        V = inputs['velocity']
        gamma = inputs['flight_path_angle']
        m = inputs['mass']
        T = inputs['thrust']
        D = inputs['drag']
        L = inputs['lift']
        alpha = inputs['alpha']

        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        V_safe = np.maximum(V, 1.0)

        # --- velocity_rate partials ---
        # dV/dt = (T*cos_a - D)/m - g*sin_g
        J['velocity_rate', 'thrust'] = cos_a / m
        J['velocity_rate', 'drag'] = -1.0 / m
        J['velocity_rate', 'alpha'] = -T * sin_a / m
        J['velocity_rate', 'mass'] = -(T * cos_a - D) / m ** 2
        J['velocity_rate', 'flight_path_angle'] = -g * cos_g

        # --- flight_path_angle_rate partials ---
        # dgamma/dt = (T*sin_a + L)/(m*V) - g*cos_g/V
        mV = m * V_safe
        num = T * sin_a + L
        J['flight_path_angle_rate', 'thrust'] = sin_a / mV
        J['flight_path_angle_rate', 'lift'] = 1.0 / mV
        J['flight_path_angle_rate', 'alpha'] = T * cos_a / mV
        J['flight_path_angle_rate', 'mass'] = -num / (m ** 2 * V_safe)
        J['flight_path_angle_rate', 'velocity'] = -num / (m * V_safe ** 2) + g * cos_g / V_safe ** 2
        J['flight_path_angle_rate', 'flight_path_angle'] = g * sin_g / V_safe

        # --- altitude_rate partials ---
        J['altitude_rate', 'velocity'] = sin_g
        J['altitude_rate', 'flight_path_angle'] = V * cos_g

        # --- range_rate partials ---
        J['range_rate', 'velocity'] = cos_g
        J['range_rate', 'flight_path_angle'] = -V * sin_g

        # --- load_factor partials ---
        weight_cos_g = m * g * np.maximum(cos_g, 0.01)
        J['load_factor', 'lift'] = 1.0 / weight_cos_g
        J['load_factor', 'thrust'] = sin_a / weight_cos_g
        J['load_factor', 'alpha'] = T * cos_a / weight_cos_g
        J['load_factor', 'mass'] = -(L + T * sin_a) / (m ** 2 * g * np.maximum(cos_g, 0.01))
        J['load_factor', 'flight_path_angle'] = (
            (L + T * sin_a) * sin_g / (m * g * np.maximum(cos_g, 0.01) ** 2)
        )
