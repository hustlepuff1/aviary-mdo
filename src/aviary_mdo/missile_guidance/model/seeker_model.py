"""
Seeker measurement effects on guidance accuracy.

Models the noise-induced miss distance and gimbal-limit constraints for
a semi-active or active homing seeker. The noise-induced miss is based
on the classical result for proportional navigation against white-noise
LOS rate measurements (Zarchan, Chapter 8).

For PN with navigation ratio N ~ 3-4:
    sigma_miss = K(N) * sigma_noise * Vc * tgo^(3/2) / sqrt(f_s)
    3-sigma miss = 3 * sigma_miss

The gimbal-limit check determines whether the current line-of-sight
angle is within the seeker's field of regard.

References:
    Zarchan, P., "Tactical and Strategic Missile Guidance," AIAA, 6th ed.
    Fleeman, E., "Tactical Missile Design," AIAA, Ch. 8.
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)


class SeekerModel(om.ExplicitComponent):
    """
    Seeker noise and gimbal-limit model for guidance sizing.

    Computes the noise-induced miss distance (3-sigma) from LOS rate
    measurement noise, seeker update rate, closing velocity, and
    time-to-go. Also evaluates whether the LOS angle is within the
    gimbal field of regard.

    All computations are pure arithmetic suitable for complex-step
    differentiation.
    """

    def setup(self):
        # --- Seeker parameters ---
        self.add_input('noise_sigma', val=1e-4, units='rad/s',
                       desc='LOS rate measurement noise, 1-sigma')
        self.add_input('sample_rate', val=50.0, units='Hz',
                       desc='Seeker measurement update rate')
        self.add_input('gimbal_limit', val=60.0, units='deg',
                       desc='Maximum seeker look angle')

        # --- Engagement state ---
        self.add_input('los_angle', val=0.0, units='rad',
                       desc='Current line-of-sight angle')
        self.add_input('los_rate', val=0.0, units='rad/s',
                       desc='True LOS rate')
        self.add_input('closing_velocity', val=1000.0, units='m/s',
                       desc='Engagement closing speed (positive when closing)')
        self.add_input('time_to_go', val=10.0, units='s',
                       desc='Estimated time remaining to intercept')

        # --- Outputs ---
        self.add_output('noise_induced_miss', val=0.0, units='m',
                        desc='3-sigma miss distance from seeker noise alone')
        self.add_output('within_gimbal', val=1.0,
                        desc='1.0 if LOS within gimbal limits, 0.0 if not')
        self.add_output('effective_los_rate', val=0.0, units='rad/s',
                        desc='Measured LOS rate (pass-through for sizing)')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        sigma_n = inputs['noise_sigma']
        f_s = inputs['sample_rate']
        gim_lim_deg = inputs['gimbal_limit']
        los_ang = inputs['los_angle']
        los_rt = inputs['los_rate']
        Vc = inputs['closing_velocity']
        tgo = inputs['time_to_go']

        # ---- Noise-induced miss (Zarchan, Ch. 8) ----
        # sigma_miss = K(N) * sigma_noise * Vc * tgo^(3/2) / sqrt(f_s)
        # K(N) ~ 0.5 for nav ratios 3-4.
        # Report 3-sigma miss as the design value.
        K_noise = 0.5

        # Protect against negative tgo and zero sample rate with smooth ops.
        # Use sqrt(tgo^2 + eps) for smooth |tgo|, then (smooth_tgo)^1.5
        tgo_safe = np.sqrt(tgo**2 + 1e-12)
        fs_safe = np.sqrt(f_s**2 + 1.0)   # smooth max(f_s, ~1)

        sigma_miss = K_noise * sigma_n * Vc * tgo_safe**1.5 / np.sqrt(fs_safe)
        noise_miss = 3.0 * sigma_miss

        outputs['noise_induced_miss'] = noise_miss

        # ---- Gimbal check ----
        # Convert gimbal limit to radians, then smooth step function.
        gim_lim_rad = gim_lim_deg * np.pi / 180.0

        # Smooth indicator: 1 when |los_angle| < gimbal_limit, 0 otherwise.
        # Use tanh with a moderate steepness for CS compatibility.
        abs_los = np.sqrt(los_ang**2 + 1e-14)
        margin = gim_lim_rad - abs_los
        within = 0.5 * (1.0 + np.tanh(50.0 * margin))

        outputs['within_gimbal'] = within

        # ---- Effective LOS rate ----
        # In the sizing context the noise adds miss distance but does not
        # bias the mean LOS rate. Pass through the true rate.
        outputs['effective_los_rate'] = los_rt
