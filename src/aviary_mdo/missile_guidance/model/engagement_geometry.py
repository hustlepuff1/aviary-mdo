"""
2-D engagement geometry for missile guidance sizing.

Computes initial engagement parameters from missile and target states:
range, closing velocity, line-of-sight angle and rate, time-to-go
estimate, and heading error.  All quantities are planar (2-D) with
the standard math convention (x-East, y-North, angles CCW from East).

Equations follow:
- Zarchan, P., "Tactical and Strategic Missile Guidance," 7th ed.,
  AIAA, 2019, Chapter 2.
- Fleeman, E., "Tactical Missile Design," 2nd ed., AIAA, 2006,
  Chapter 8.
"""

import numpy as np
import openmdao.api as om
from openmdao.utils.cs_safe import arctan2 as cs_arctan2

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)


class EngagementGeometry(om.ExplicitComponent):
    """
    Planar engagement geometry from initial missile and target states.

    Given the 2-D position, speed, and heading of both the missile and
    the target, this component computes the engagement geometry
    parameters needed by proportional navigation guidance laws.

    Inputs
    ------
    guidance:missile:position_x : float (m)
        Missile initial X position.
    guidance:missile:position_y : float (m)
        Missile initial Y position.
    guidance:missile:velocity : float (m/s)
        Missile speed.
    guidance:missile:heading : float (deg)
        Missile heading, 0 = East, 90 = North.
    guidance:target:position_x : float (m)
        Target initial X position.
    guidance:target:position_y : float (m)
        Target initial Y position.
    guidance:target:velocity : float (m/s)
        Target speed.
    guidance:target:heading : float (deg)
        Target heading, 0 = East, 90 = North.

    Outputs
    -------
    guidance:engagement:initial_range : float (m)
        Distance between missile and target.
    guidance:engagement:closing_velocity : float (m/s)
        Rate at which range is decreasing (positive when closing).
    guidance:engagement:los_angle : float (rad)
        Line-of-sight angle from missile to target.
    guidance:engagement:los_rate : float (rad/s)
        Rate of change of LOS angle.
    guidance:engagement:time_to_go : float (s)
        Estimated time to intercept (range / Vc).
    guidance:engagement:heading_error : float (deg)
        Angle between missile heading and LOS direction.
    """

    def setup(self):
        # ----- Missile state -----
        self.add_input(Guidance.Missile.POSITION_X, val=0.0, units='m',
                       desc='Missile initial X position')
        self.add_input(Guidance.Missile.POSITION_Y, val=0.0, units='m',
                       desc='Missile initial Y position')
        self.add_input(Guidance.Missile.VELOCITY, val=1000.0, units='m/s',
                       desc='Missile speed')
        self.add_input(Guidance.Missile.HEADING, val=45.0, units='deg',
                       desc='Missile heading (0=East, 90=North)')

        # ----- Target state -----
        self.add_input(Guidance.Target.POSITION_X, val=15000.0, units='m',
                       desc='Target initial X position')
        self.add_input(Guidance.Target.POSITION_Y, val=15000.0, units='m',
                       desc='Target initial Y position')
        self.add_input(Guidance.Target.VELOCITY, val=300.0, units='m/s',
                       desc='Target speed')
        self.add_input(Guidance.Target.HEADING, val=180.0, units='deg',
                       desc='Target heading (0=East, 90=North)')

        # ----- Engagement geometry outputs -----
        self.add_output(Guidance.Engagement.INITIAL_RANGE, val=1.0, units='m',
                        desc='Initial missile-to-target range',
                        lower=0.0)
        self.add_output(Guidance.Engagement.CLOSING_VELOCITY, val=0.0,
                        units='m/s',
                        desc='Closing velocity (positive when closing)')
        self.add_output(Guidance.Engagement.LOS_ANGLE, val=0.0, units='rad',
                        desc='Line-of-sight angle from missile to target')
        self.add_output(Guidance.Engagement.LOS_RATE, val=0.0, units='rad/s',
                        desc='Rate of change of LOS angle')
        self.add_output(Guidance.Engagement.TIME_TO_GO, val=1.0, units='s',
                        desc='Estimated time to intercept',
                        lower=0.0)
        self.add_output(Guidance.Engagement.HEADING_ERROR, val=0.0,
                        units='deg',
                        desc='Heading error to collision course')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        # Unpack missile state
        x_m = inputs[Guidance.Missile.POSITION_X]
        y_m = inputs[Guidance.Missile.POSITION_Y]
        V_m = inputs[Guidance.Missile.VELOCITY]
        hdg_m_deg = inputs[Guidance.Missile.HEADING]

        # Unpack target state
        x_t = inputs[Guidance.Target.POSITION_X]
        y_t = inputs[Guidance.Target.POSITION_Y]
        V_t = inputs[Guidance.Target.VELOCITY]
        hdg_t_deg = inputs[Guidance.Target.HEADING]

        # Convert headings to radians (manual multiply for complex-step)
        hdg_m = hdg_m_deg * np.pi / 180.0
        hdg_t = hdg_t_deg * np.pi / 180.0

        # ----- Relative position -----
        dx = x_t - x_m
        dy = y_t - y_m
        R = np.sqrt(dx**2 + dy**2)

        # Guard against zero range (would cause division by zero).
        # Add a small constant under the sqrt for smooth, complex-step
        # compatible protection instead of np.maximum.
        R_safe = np.sqrt(dx**2 + dy**2 + 1.0e-10)

        # ----- LOS angle (missile -> target) -----
        lam = cs_arctan2(dy, dx)

        # ----- Velocity components -----
        Vmx = V_m * np.cos(hdg_m)
        Vmy = V_m * np.sin(hdg_m)
        Vtx = V_t * np.cos(hdg_t)
        Vty = V_t * np.sin(hdg_t)

        # ----- Relative velocity (target minus missile) -----
        Vrx = Vtx - Vmx
        Vry = Vty - Vmy

        # ----- Range rate (dR/dt, negative when closing) -----
        R_dot = (dx * Vrx + dy * Vry) / R_safe

        # Closing velocity (positive when closing)
        Vc = -R_dot

        # ----- LOS rate (d_lambda/dt) -----
        # From the cross product of (relative position) and (relative velocity)
        lambda_dot = (dx * Vry - dy * Vrx) / R_safe**2

        # ----- Time-to-go estimate -----
        # Smooth lower bound on Vc for cs compatibility:
        # Vc_safe ~ max(Vc, 1.0) via softplus-like construction
        _VC_FLOOR = 1.0
        Vc_safe = 0.5 * (Vc + _VC_FLOOR + np.sqrt((Vc - _VC_FLOOR)**2 + 1.0e-6))
        tgo = R_safe / Vc_safe

        # ----- Heading error -----
        # Angle from missile heading to LOS, wrapped to [-pi, pi]
        HE_rad = lam - hdg_m
        # Wrap to [-pi, pi] using arctan2 identity
        HE_rad = cs_arctan2(np.sin(HE_rad), np.cos(HE_rad))
        HE_deg = HE_rad * 180.0 / np.pi

        # ----- Pack outputs -----
        outputs[Guidance.Engagement.INITIAL_RANGE] = R
        outputs[Guidance.Engagement.CLOSING_VELOCITY] = Vc
        outputs[Guidance.Engagement.LOS_ANGLE] = lam
        outputs[Guidance.Engagement.LOS_RATE] = lambda_dot
        outputs[Guidance.Engagement.TIME_TO_GO] = tgo
        outputs[Guidance.Engagement.HEADING_ERROR] = HE_deg
