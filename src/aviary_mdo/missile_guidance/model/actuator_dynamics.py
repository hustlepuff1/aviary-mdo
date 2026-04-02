"""
Autopilot and actuator dynamics for missile guidance sizing.

Models the combined effect of autopilot time constant, fin rate limiting,
and acceleration saturation on achievable lateral acceleration. Used in
the guidance sizing loop to evaluate control authority margin.

The effective time constant accounts for rate-limiting degradation:
    tau_eff = tau + 0.5 * (deflection_limit / rate_limit)

Saturation clips commanded acceleration to the missile's maximum lateral
acceleration capability, preserving sign.

Reference: Zarchan, P., "Tactical and Strategic Missile Guidance,"
           AIAA, 6th ed., Chapter 8 (Autopilot Effects).
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)


class ActuatorDynamics(om.ExplicitComponent):
    """
    Combined autopilot + actuator dynamics for guidance sizing.

    Computes the achieved lateral acceleration after accounting for
    time-constant lag and acceleration saturation. Outputs include the
    control authority ratio (achieved / commanded) and an effective time
    constant that incorporates fin-rate-limit degradation.

    All computations are pure arithmetic suitable for complex-step
    differentiation.
    """

    def setup(self):
        # --- Autopilot / actuator parameters ---
        self.add_input('time_constant', val=0.2, units='s',
                       desc='Combined autopilot + servo time constant')
        self.add_input('rate_limit', val=200.0, units='deg/s',
                       desc='Maximum fin deflection rate')
        self.add_input('deflection_limit', val=25.0, units='deg',
                       desc='Maximum fin deflection angle')

        # --- Acceleration envelope ---
        self.add_input('max_acceleration', val=300.0, units='m/s**2',
                       desc='Missile maximum lateral acceleration capability')
        self.add_input('commanded_accel', val=0.0, units='m/s**2',
                       desc='Guidance-law commanded lateral acceleration')

        # --- Outputs ---
        self.add_output('achieved_accel', val=0.0, units='m/s**2',
                        desc='Actual achieved acceleration after lag and saturation')
        self.add_output('accel_ratio', val=1.0,
                        desc='Achieved / commanded ratio (0-1 control authority margin)')
        self.add_output('effective_time_constant', val=0.2, units='s',
                        desc='Time constant accounting for rate-limit degradation')
        self.add_output('is_saturated', val=0.0,
                        desc='1.0 if commanded exceeds available, 0.0 otherwise')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        tau = inputs['time_constant']
        rate_lim = inputs['rate_limit']
        defl_lim = inputs['deflection_limit']
        a_max = inputs['max_acceleration']
        a_cmd = inputs['commanded_accel']

        # ---- Saturation check ----
        # Use smooth min/max to keep complex-step friendly.
        # abs(a_cmd) vs a_max
        abs_cmd = np.sqrt(a_cmd**2 + 1e-12)  # smooth abs
        abs_max = np.sqrt(a_max**2 + 1e-12)

        # Smooth saturation indicator: approaches 1 when abs_cmd > abs_max
        # tanh ramp centered at abs_max with narrow transition
        is_sat = 0.5 * (1.0 + np.tanh(10.0 * (abs_cmd - abs_max)))
        outputs['is_saturated'] = is_sat

        # ---- Achieved acceleration ----
        # Clip magnitude to a_max while preserving sign.
        # achieved = a_cmd * min(1, a_max / abs_cmd)
        # Smooth version: use a_max / sqrt(a_cmd^2 + a_max^2) * a_cmd
        # when |a_cmd| >> a_max this approaches sign(a_cmd)*a_max
        clamp_factor = abs_max / np.sqrt(a_cmd**2 + a_max**2 + 1e-12)
        a_achieved = a_cmd * clamp_factor

        outputs['achieved_accel'] = a_achieved

        # ---- Accel ratio (control authority margin) ----
        # ratio = |a_achieved| / max(|a_cmd|, epsilon)
        abs_achieved = np.sqrt(a_achieved**2 + 1e-12)
        ratio = abs_achieved / (abs_cmd + 1e-6)
        # Clamp to [0, 1] range smoothly
        ratio = ratio * 0.5 * (1.0 + np.tanh(100.0 * (1.0 - ratio))) \
            + 1.0 * 0.5 * (1.0 + np.tanh(100.0 * (ratio - 1.0)))

        outputs['accel_ratio'] = ratio

        # ---- Effective time constant with rate-limit degradation ----
        # tau_rate = deflection_limit / max(rate_limit, 1.0)
        # tau_eff = tau + 0.5 * tau_rate
        rate_lim_safe = np.sqrt(rate_lim**2 + 1.0)  # smooth max(rate_lim, ~1)
        tau_rate = defl_lim / rate_lim_safe
        tau_eff = tau + 0.5 * tau_rate

        outputs['effective_time_constant'] = tau_eff
