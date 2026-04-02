"""
Proportional Navigation (PN) guidance analysis for missile sizing.

Implements analytical PN and Augmented PN (APN) computations for
use in multidisciplinary design optimization loops.  These are *not*
time-stepping simulations; they provide single-evaluation estimates
of required acceleration, miss distance, load factor, and divert
budget from the initial engagement geometry.

Two guidance law modes:
  PN  -- basic proportional navigation: a_cmd = N' * Vc * lambda_dot
  APN -- augmented proportional navigation: adds (N'/2) * n_T term
         to compensate for target maneuver acceleration.

Miss distance estimation uses the Fleeman single-lag adjoint model,
which captures the degradation caused by autopilot time constant.

References
----------
- Zarchan, P., "Tactical and Strategic Missile Guidance," 7th ed.,
  AIAA, 2019, Chapters 2-4.
- Fleeman, E., "Tactical Missile Design," 2nd ed., AIAA, 2006,
  Chapter 8 (Guidance).
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)

# Standard gravity (m/s^2)
_G0 = 9.80665


class ProportionalNavigation(om.ExplicitComponent):
    """
    Analytical Proportional Navigation guidance sizing.

    Computes the commanded lateral acceleration, zero-effort miss,
    adjoint miss distance (Fleeman single-lag model), required load
    factor, and total divert requirement from engagement geometry
    inputs.  Supports both basic PN and Augmented PN.

    Inputs
    ------
    guidance:nav:ratio : float
        Navigation constant N' (dimensionless, typically 3-5).
    guidance:nav:law : str (discrete)
        Guidance law selection: 'PN' or 'APN'.
    guidance:engagement:closing_velocity : float (m/s)
        Closing velocity (positive when closing).
    guidance:engagement:los_rate : float (rad/s)
        Line-of-sight rate.
    guidance:engagement:initial_range : float (m)
        Missile-to-target range.
    guidance:engagement:time_to_go : float (s)
        Estimated time to intercept.
    guidance:autopilot:time_constant : float (s)
        Combined autopilot + actuator time constant.
    guidance:target:acceleration_max : float (m/s^2)
        Maximum target maneuver acceleration.
    guidance:missile:max_acceleration : float (m/s^2)
        Missile maximum lateral acceleration capability.

    Outputs
    -------
    dynamic:guidance:max_commanded_accel : float (m/s^2)
        Required commanded acceleration from the guidance law.
    dynamic:guidance:zero_effort_miss : float (m)
        Zero-effort miss distance (miss if no further guidance).
    dynamic:guidance:miss_distance : float (m)
        Adjoint miss distance estimate (Fleeman single-lag model).
    dynamic:guidance:required_load_factor : float (g)
        Peak load factor required.
    dynamic:guidance:divert_requirement : float (m/s)
        Total delta-V budget from guidance commands.
    """

    def setup(self):
        # ----- Navigation law parameters -----
        self.add_input(Guidance.Navigation.RATIO, val=4.0,
                       desc='Navigation ratio N-prime (3-5 typical)')
        self.add_discrete_input(Guidance.Navigation.LAW, val='PN',
                                desc='Guidance law: PN or APN')

        # ----- Engagement geometry (from EngagementGeometry) -----
        self.add_input(Guidance.Engagement.CLOSING_VELOCITY, val=1000.0,
                       units='m/s',
                       desc='Closing velocity (positive when closing)')
        self.add_input(Guidance.Engagement.LOS_RATE, val=0.01,
                       units='rad/s',
                       desc='Line-of-sight rate')
        self.add_input(Guidance.Engagement.INITIAL_RANGE, val=15000.0,
                       units='m',
                       desc='Missile-to-target range')
        self.add_input(Guidance.Engagement.TIME_TO_GO, val=15.0, units='s',
                       desc='Estimated time to intercept')

        # ----- Autopilot -----
        self.add_input(Guidance.Autopilot.TIME_CONSTANT, val=0.2, units='s',
                       desc='Autopilot + actuator time constant')

        # ----- Target -----
        self.add_input(Guidance.Target.ACCELERATION_MAX, val=50.0,
                       units='m/s**2',
                       desc='Maximum target maneuver acceleration')

        # ----- Missile -----
        self.add_input(Guidance.Missile.MAX_ACCELERATION, val=300.0,
                       units='m/s**2',
                       desc='Missile maximum lateral acceleration')

        # ----- Outputs -----
        self.add_output(DynamicGuidance.MAX_COMMANDED_ACCEL, val=0.0,
                        units='m/s**2',
                        desc='Required commanded acceleration')
        self.add_output(DynamicGuidance.ZERO_EFFORT_MISS, val=0.0,
                        units='m',
                        desc='Zero effort miss distance')
        self.add_output(DynamicGuidance.MISS_DISTANCE, val=0.0,
                        units='m',
                        desc='Adjoint miss distance (Fleeman single-lag)')
        self.add_output(DynamicGuidance.REQUIRED_LOAD_FACTOR, val=0.0,
                        desc='Peak required load factor (g-units)')
        self.add_output(DynamicGuidance.DIVERT_REQUIREMENT, val=0.0,
                        units='m/s',
                        desc='Total delta-V from guidance commands')

    def setup_partials(self):
        # Partials for all continuous inputs via complex step.
        # The discrete input (guidance law string) has no partials.
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None,
                discrete_outputs=None):
        N = inputs[Guidance.Navigation.RATIO]
        Vc = inputs[Guidance.Engagement.CLOSING_VELOCITY]
        lam_dot = inputs[Guidance.Engagement.LOS_RATE]
        R = inputs[Guidance.Engagement.INITIAL_RANGE]
        tgo = inputs[Guidance.Engagement.TIME_TO_GO]
        tau = inputs[Guidance.Autopilot.TIME_CONSTANT]
        n_T = inputs[Guidance.Target.ACCELERATION_MAX]
        a_max = inputs[Guidance.Missile.MAX_ACCELERATION]

        # Retrieve guidance law (discrete, not differentiable)
        if discrete_inputs is not None:
            law = discrete_inputs[Guidance.Navigation.LAW]
        else:
            law = 'PN'

        # ----- Basic PN commanded acceleration -----
        # a_cmd = N' * Vc * lambda_dot  (Zarchan Eq. 2.1)
        a_cmd = N * Vc * lam_dot

        # ----- Zero Effort Miss -----
        # ZEM = R * lambda_dot * tgo  (Zarchan Ch. 4)
        ZEM = R * lam_dot * tgo

        # ----- Adjoint miss distance (Fleeman single-lag model) -----
        # For PN with a single autopilot time constant tau, the miss
        # due to a step target maneuver of magnitude n_T is:
        #   miss = n_T * tau^2 * 0.5 * (1 + 4*tau / tgo)
        # This captures the degradation as tgo approaches tau (endgame
        # timing). The factor of 0.5 applies for N'=3; for other N'
        # values the constant changes, but this is the standard Fleeman
        # approximation used for sizing.
        #
        # Guard against tgo < tau to prevent blow-up.
        # Smooth lower bound for complex-step compatibility:
        # tgo_safe ~ max(tgo, tau + 0.01)
        _floor = tau + 0.01
        tgo_safe = 0.5 * (tgo + _floor + np.sqrt((tgo - _floor)**2 + 1.0e-6))
        miss_adjoint = n_T * tau**2 * 0.5 * (1.0 + 4.0 * tau / tgo_safe)

        # ----- Augmented PN adjustments -----
        if law == 'APN':
            # APN adds a target-acceleration compensation term:
            #   a_cmd_APN = N' * Vc * lambda_dot + (N'/2) * n_T
            # The (N'/2)*n_T term drives ZEM to zero for a constant
            # target maneuver, roughly halving the adjoint miss
            # (Zarchan Ch. 3).
            a_cmd = N * Vc * lam_dot + (N / 2.0) * n_T
            miss_adjoint = miss_adjoint * 0.5

        # ----- Required load factor (in g-units) -----
        # Use smooth absolute value: sqrt(x^2 + eps) for cs compatibility
        abs_a_cmd = np.sqrt(a_cmd**2 + 1.0e-10)
        n_req = abs_a_cmd / _G0

        # ----- Divert requirement (delta-V budget) -----
        # Simplified: integral of |a_lat| over flight time ~ |a_cmd|*tgo
        divert = abs_a_cmd * tgo

        # ----- Pack outputs -----
        outputs[DynamicGuidance.MAX_COMMANDED_ACCEL] = a_cmd
        outputs[DynamicGuidance.ZERO_EFFORT_MISS] = ZEM
        outputs[DynamicGuidance.MISS_DISTANCE] = miss_adjoint
        outputs[DynamicGuidance.REQUIRED_LOAD_FACTOR] = n_req
        outputs[DynamicGuidance.DIVERT_REQUIREMENT] = divert


class ProNavGuidance(om.ExplicitComponent):
    """
    Simplified PN guidance component with short variable names.

    Uses local naming for standalone proportional navigation analysis
    and unit testing.  Computes commanded acceleration and adjoint miss
    distance from the PN guidance law.

    Inputs
    ------
    los_rate : float (rad/s)
        Line-of-sight rate.
    closing_velocity : float (m/s)
        Engagement closing speed.
    nav_ratio : float
        Navigation constant N'.
    time_constant : float (s)
        Autopilot time constant (for adjoint miss model).
    target_accel : float (m/s**2)
        Target maneuver acceleration magnitude.
    time_to_go : float (s)
        Estimated time to intercept.

    Outputs
    -------
    commanded_accel : float (m/s**2)
        PN commanded lateral acceleration (N' * Vc * lambda_dot).
    adjoint_miss_distance : float (m)
        Fleeman single-lag adjoint miss estimate.
    """

    def setup(self):
        self.add_input('los_rate', val=0.01, units='rad/s',
                       desc='Line-of-sight rate')
        self.add_input('closing_velocity', val=1000.0, units='m/s',
                       desc='Closing velocity (positive when closing)')
        self.add_input('nav_ratio', val=4.0,
                       desc='Navigation ratio N-prime')
        self.add_input('time_constant', val=0.2, units='s',
                       desc='Autopilot + actuator time constant')
        self.add_input('target_accel', val=50.0, units='m/s**2',
                       desc='Target maneuver acceleration magnitude')
        self.add_input('time_to_go', val=10.0, units='s',
                       desc='Estimated time to intercept')

        self.add_output('commanded_accel', val=0.0, units='m/s**2',
                        desc='PN commanded lateral acceleration')
        self.add_output('adjoint_miss_distance', val=0.0, units='m',
                        desc='Adjoint miss distance (Fleeman single-lag)')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        lam_dot = inputs['los_rate']
        Vc = inputs['closing_velocity']
        N = inputs['nav_ratio']
        tau = inputs['time_constant']
        n_T = inputs['target_accel']
        tgo = inputs['time_to_go']

        # PN commanded acceleration: a_cmd = N' * Vc * lambda_dot
        a_cmd = N * Vc * lam_dot
        outputs['commanded_accel'] = a_cmd

        # Adjoint miss distance (Fleeman single-lag model):
        #   miss = n_T * tau^2 * 0.5 * (1 + 4*tau / tgo)
        # Smooth lower bound for complex-step compatibility
        _floor = tau + 0.01
        tgo_safe = 0.5 * (tgo + _floor + np.sqrt((tgo - _floor)**2 + 1.0e-6))
        miss = n_T * tau**2 * 0.5 * (1.0 + 4.0 * tau / tgo_safe)
        outputs['adjoint_miss_distance'] = miss
