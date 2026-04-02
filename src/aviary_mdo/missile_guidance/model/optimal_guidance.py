"""
Optimal Guidance Law (OGL) and Zero Effort Miss analysis.

Implements the Optimal Guidance Law for a first-order-lag autopilot
missile against a bounded-evasion target. Translated from the Technion
MATLAB Projects 2 and 3 (Engelsman, "Advanced Topics in Missile
Guidance").

Key results:
    - Zero Effort Miss: ZEM = R * los_rate * tgo
    - OGL commanded accel: a_cmd = N_eff * ZEM / tgo^2
      with N_eff = N' * (1 + tau/tgo) correcting for autopilot lag
    - Predicted miss under OGL with bounded target evasion

The critical time-to-go (tgo_cr) determines whether the missile has
sufficient maneuverability advantage to guarantee intercept:
    tgo_cr = 2 * tau * sqrt(n_T / delta_rho + 1)
    where delta_rho = n_M - n_T

References:
    Zarchan, P., "Tactical and Strategic Missile Guidance," AIAA, 6th ed.
    Engelsman, "Advanced Topics in Missile Guidance," Technion Projects 2-3.
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)


class OptimalGuidance(om.ExplicitComponent):
    """
    Optimal Guidance Law (OGL) with Zero Effort Miss computation.

    Computes the OGL commanded acceleration for a linear engagement
    model with first-order autopilot lag. Predicts the miss distance
    under OGL considering bounded target evasion and autopilot dynamics.

    All computations are pure arithmetic suitable for complex-step
    differentiation. No branching on computed values.
    """

    def setup(self):
        # --- Navigation parameters ---
        self.add_input('ratio', val=4.0,
                       desc='Navigation ratio N-prime (3-5 typical)')

        # --- Engagement geometry ---
        self.add_input('initial_range', val=15000.0, units='m',
                       desc='Initial missile-to-target range')
        self.add_input('closing_velocity', val=1000.0, units='m/s',
                       desc='Engagement closing speed')
        self.add_input('los_rate', val=0.01, units='rad/s',
                       desc='Line-of-sight rate')
        self.add_input('time_to_go', val=10.0, units='s',
                       desc='Estimated time remaining to intercept')

        # --- Target capability ---
        self.add_input('acceleration_max', val=50.0, units='m/s**2',
                       desc='Maximum target maneuver acceleration')

        # --- Missile capability ---
        self.add_input('max_acceleration', val=300.0, units='m/s**2',
                       desc='Missile maximum lateral acceleration')

        # --- Autopilot ---
        self.add_input('time_constant', val=0.2, units='s',
                       desc='Autopilot + actuator combined time constant')

        # --- Outputs ---
        self.add_output('zero_effort_miss', val=0.0, units='m',
                        desc='Zero Effort Miss distance')
        self.add_output('ogl_commanded_accel', val=0.0, units='m/s**2',
                        desc='OGL commanded lateral acceleration')
        self.add_output('ogl_miss_distance', val=0.0, units='m',
                        desc='Predicted miss distance under OGL')
        self.add_output('required_load_factor', val=0.0,
                        desc='Peak required load factor (g-units)')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        N_prime = inputs['ratio']
        R0 = inputs['initial_range']
        Vc = inputs['closing_velocity']
        los_rt = inputs['los_rate']
        tgo = inputs['time_to_go']
        n_T = inputs['acceleration_max']
        n_M = inputs['max_acceleration']
        tau = inputs['time_constant']

        g0 = 9.81  # m/s^2

        # ---- Zero Effort Miss ----
        # ZEM = R * los_rate * tgo
        # Use smooth tgo to avoid division-by-zero downstream.
        tgo_safe = np.sqrt(tgo**2 + 1e-6)
        ZEM = R0 * los_rt * tgo_safe

        outputs['zero_effort_miss'] = ZEM

        # ---- OGL commanded acceleration ----
        # Effective navigation ratio corrected for first-order lag:
        #   N_eff = N' * (1 + tau / tgo)
        # This increases the gain at short time-to-go to compensate
        # for autopilot lag (Technion Project 3 result).
        N_eff = N_prime * (1.0 + tau / tgo_safe)

        # a_cmd = N_eff * ZEM / tgo^2
        tgo2_safe = tgo_safe**2 + 1e-3  # prevent divide-by-zero
        a_ogl = N_eff * ZEM / tgo2_safe

        outputs['ogl_commanded_accel'] = a_ogl

        # ---- Required load factor ----
        # n_req = |a_ogl| / g
        abs_a_ogl = np.sqrt(a_ogl**2 + 1e-12)
        n_req = abs_a_ogl / g0

        outputs['required_load_factor'] = n_req

        # ---- Predicted miss distance under OGL ----
        # Maneuverability advantage: delta_rho = n_M - n_T
        # Both n_M and n_T are magnitudes (positive).
        abs_n_M = np.sqrt(n_M**2 + 1e-12)
        abs_n_T = np.sqrt(n_T**2 + 1e-12)
        delta_rho = abs_n_M - abs_n_T

        # Critical time-to-go from Technion Project 3:
        #   tgo_cr = 2 * tau * sqrt(n_T / delta_rho + 1)
        # When delta_rho <= 0 (target more agile), tgo_cr -> large.
        # Use smooth formulation to avoid branching.

        # Smooth max(delta_rho, epsilon) for the ratio
        delta_safe = np.sqrt(delta_rho**2 + 1.0) * 0.5 + delta_rho * 0.5 + 0.5
        # delta_safe is approximately max(delta_rho, ~0.5)

        tgo_cr = 2.0 * tau * np.sqrt(abs_n_T / delta_safe + 1.0)

        # Miss distance model:
        # When tgo > tgo_cr and delta_rho > 0: OGL drives miss to ~0
        # When tgo < tgo_cr or delta_rho < 0: residual miss ~ n_T * tau^2
        # Use smooth blending via tanh.

        # Smooth indicator for "missile has the advantage and enough time"
        # advantage_factor in [0, 1]: 1 = favorable, 0 = unfavorable
        advantage = 0.5 * (1.0 + np.tanh(2.0 * delta_rho))
        time_margin = 0.5 * (1.0 + np.tanh(5.0 * (tgo_safe - tgo_cr)))

        # Favorable case: miss -> 0
        # Unfavorable case: miss ~ n_T * tau^2 (from target evasion through lag)
        miss_residual = abs_n_T * tau**2
        # Worst case (no advantage at all): miss ~ |ZEM|
        abs_ZEM = np.sqrt(ZEM**2 + 1e-12)

        # Blend:
        # good conditions (advantage=1, time_margin=1) -> miss ~ 0
        # some advantage but not enough time -> miss ~ n_T * tau^2
        # no advantage -> miss grows toward |ZEM|
        favorability = advantage * time_margin
        miss_ogl = miss_residual * (1.0 - favorability) \
            + abs_ZEM * (1.0 - advantage) * (1.0 - time_margin) * 0.1

        outputs['ogl_miss_distance'] = miss_ogl
