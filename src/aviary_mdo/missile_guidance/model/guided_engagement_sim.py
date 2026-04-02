"""
3-DOF point-mass guided engagement simulation.

Runs a full 2-D planar missile-vs-target engagement using scipy's
solve_ivp integrator.  Supports proportional navigation (PN),
augmented PN (APN), and optimal guidance law (OGL) with a first-order
autopilot lag model.  Replaces the MATLAB/Simulink models from the
Technion "Advanced Topics in Missile Guidance" course with a clean
Python implementation suitable for MDO trade studies.

State vector (9 states)::

    y = [x_m, y_m, V_m, theta_m, x_t, y_t, V_t, theta_t, a_achieved]
         0     1    2     3        4    5    6     7         8

Coordinate convention: x-East, y-North, angles CCW from East (standard
math convention, consistent with EngagementGeometry).

References
----------
- Zarchan, P., "Tactical and Strategic Missile Guidance," 7th ed.,
  AIAA, 2019, Chapters 2-8.
- Fleeman, E., "Tactical Missile Design," 2nd ed., AIAA, 2006, Ch. 8.
- Engelsman, Y., "Advanced Topics in Missile Guidance," Technion, 2024.
"""

import numpy as np
import openmdao.api as om
from scipy.integrate import solve_ivp

from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G = 9.80665            # standard gravity (m/s^2)
_R_LETHAL = 10.0        # warhead lethal radius, SM-2 class (m)
_MAX_SIM_TIME = 200.0   # maximum simulation duration (s)
_MIN_RANGE_GUARD = 0.1  # prevent division by zero in range (m)
_MIN_RANGE_SQ_GUARD = 1.0  # prevent division by zero in range^2 (m^2)
_MIN_SPEED_GUARD = 10.0  # clamp speed for heading-rate calc (m/s)
_MIN_VC_GUARD = 1.0     # minimum closing velocity for tgo (m/s)
_MIN_TGO_SQ_GUARD = 0.001  # minimum tgo^2 for OGL (s^2)
_MISS_SENTINEL = 1.0e6  # fallback miss distance when solver fails (m)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------
def _engagement_ode(t, y, params):
    """
    Equations of motion for the 2-D point-mass engagement.

    Parameters
    ----------
    t : float
        Current time (s).
    y : array_like, shape (9,)
        State vector [x_m, y_m, V_m, theta_m, x_t, y_t, V_t, theta_t, a_m].
    params : dict
        Engagement parameters (nav ratio, guidance law, limits, etc.).

    Returns
    -------
    dydt : list of float, length 9
        State derivatives.
    """
    x_m, y_m, V_m, theta_m, x_t, y_t, V_t, theta_t, a_m = y

    N = params['N']
    law = params['law']
    tau = params['tau']
    max_accel = params['max_accel']
    rho = params['rho']
    CD = params['CD']
    S_ref = params['S_ref']
    mass = params['mass']
    thrust = params['thrust']
    burn_time = params['burn_time']
    a_t_max = params['a_t_max']
    maneuver_time = params['maneuver_time']

    # ---- Engagement geometry ----
    dx = x_t - x_m
    dy = y_t - y_m
    R = np.sqrt(dx * dx + dy * dy)
    R_safe = max(R, _MIN_RANGE_GUARD)

    # Velocity components
    cos_theta_m = np.cos(theta_m)
    sin_theta_m = np.sin(theta_m)
    cos_theta_t = np.cos(theta_t)
    sin_theta_t = np.sin(theta_t)

    Vmx = V_m * cos_theta_m
    Vmy = V_m * sin_theta_m
    Vtx = V_t * cos_theta_t
    Vty = V_t * sin_theta_t

    # Relative velocity (target minus missile)
    Vrx = Vtx - Vmx
    Vry = Vty - Vmy

    # Range rate and closing velocity
    R_dot = (dx * Vrx + dy * Vry) / R_safe
    Vc = -R_dot  # positive when closing

    # LOS angle and rate
    R_sq_safe = max(R_safe * R_safe, _MIN_RANGE_SQ_GUARD)
    lam_dot = (dx * Vry - dy * Vrx) / R_sq_safe

    # Time to go estimate
    Vc_safe = max(Vc, _MIN_VC_GUARD)
    tgo = R_safe / Vc_safe

    # ---- Target maneuver (step onset) ----
    a_t = a_t_max if t >= maneuver_time else 0.0

    # ---- Guidance law ----
    if law == 'APN':
        # Augmented proportional navigation: compensates target accel
        a_cmd = N * Vc * lam_dot + 0.5 * N * a_t
    elif law == 'OGL':
        # Optimal guidance law (linear-quadratic)
        ZEM = R_safe * lam_dot * tgo
        tgo_sq = max(tgo * tgo, _MIN_TGO_SQ_GUARD)
        a_cmd = N * ZEM / tgo_sq
    else:
        # Pure proportional navigation (default / 'PN')
        a_cmd = N * Vc * lam_dot

    # Saturate commanded acceleration
    a_cmd = np.clip(a_cmd, -max_accel, max_accel)

    # Track maximum commanded accel via extra bookkeeping in params
    params['_a_cmd_history'].append(abs(a_cmd))

    # ---- Autopilot first-order lag ----
    tau_safe = max(tau, 1.0e-6)
    da_m = (a_cmd - a_m) / tau_safe

    # ---- Missile equations of motion ----
    # Drag
    D = 0.5 * rho * V_m * V_m * CD * S_ref

    # Thrust (only during burn)
    T = thrust if t <= burn_time else 0.0

    # Along-velocity: dV/dt = (T - D) / m
    dVm = (T - D) / mass

    # Normal to velocity: dtheta/dt = a_lateral / V
    V_m_safe = max(V_m, _MIN_SPEED_GUARD)
    dtheta_m = a_m / V_m_safe

    # Position derivatives
    dx_m = V_m * cos_theta_m
    dy_m = V_m * sin_theta_m

    # ---- Target equations of motion ----
    # Target maneuver is perpendicular to its velocity vector
    V_t_safe = max(V_t, _MIN_SPEED_GUARD)
    dtheta_t = a_t / V_t_safe
    dVt = 0.0  # constant speed target

    dx_t = V_t * cos_theta_t
    dy_t = V_t * sin_theta_t

    return [dx_m, dy_m, dVm, dtheta_m, dx_t, dy_t, dVt, dtheta_t, da_m]


# ---------------------------------------------------------------------------
# Terminal event: range starts increasing (closest approach passed)
# ---------------------------------------------------------------------------
def _make_range_increasing_event(params):
    """
    Factory that returns a terminal event function for solve_ivp.

    The event fires when R_dot crosses zero from negative to positive,
    meaning the missile has passed closest approach.
    """
    def range_increasing(t, y):
        x_m, y_m, V_m, theta_m, x_t, y_t, V_t, theta_t, _ = y
        dx = x_t - x_m
        dy = y_t - y_m
        R = np.sqrt(dx * dx + dy * dy)
        R_safe = max(R, _MIN_RANGE_GUARD)

        Vmx = V_m * np.cos(theta_m)
        Vmy = V_m * np.sin(theta_m)
        Vtx = V_t * np.cos(theta_t)
        Vty = V_t * np.sin(theta_t)

        Vrx = Vtx - Vmx
        Vry = Vty - Vmy

        R_dot = (dx * Vrx + dy * Vry) / R_safe
        return R_dot

    range_increasing.terminal = True
    range_increasing.direction = 1.0  # only trigger on rising zero crossing
    return range_increasing


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------
class GuidedEngagementSim(om.ExplicitComponent):
    """
    2-D point-mass guided engagement simulation.

    Integrates a 9-state ODE (missile + target kinematics with guidance
    law and first-order autopilot lag) using scipy.integrate.solve_ivp
    to determine miss distance, required acceleration, Pk, and other
    engagement performance metrics.

    The component is a "black box" from OpenMDAO's perspective: partial
    derivatives are computed via finite differencing because the ODE
    solver is not analytically differentiable.

    Inputs
    ------
    guidance:missile:position_x : float (m)
    guidance:missile:position_y : float (m)
    guidance:missile:velocity : float (m/s)
    guidance:missile:heading : float (deg)
    guidance:missile:max_acceleration : float (m/s**2)
    guidance:missile:drag_coefficient : float (unitless)
    guidance:missile:reference_area : float (m**2)
    guidance:missile:mass : float (kg)
    guidance:missile:thrust : float (N)
    guidance:missile:burn_time : float (s)
    guidance:target:position_x : float (m)
    guidance:target:position_y : float (m)
    guidance:target:velocity : float (m/s)
    guidance:target:heading : float (deg)
    guidance:target:acceleration_max : float (m/s**2)
    guidance:target:maneuver_time : float (s)
    guidance:nav:ratio : float (unitless)
    guidance:nav:law : str, discrete ('PN', 'APN', 'OGL')
    guidance:autopilot:time_constant : float (s)
    guidance:env:altitude : float (m)
    guidance:env:density : float (kg/m**3)

    Outputs
    -------
    dynamic:guidance:miss_distance : float (m)
    dynamic:guidance:max_lateral_accel : float (m/s**2)
    dynamic:guidance:max_commanded_accel : float (m/s**2)
    dynamic:guidance:time_to_intercept : float (s)
    dynamic:guidance:zero_effort_miss : float (m)
    dynamic:guidance:divert_requirement : float (m/s)
    dynamic:guidance:closest_approach : float (m)
    dynamic:guidance:required_load_factor : float (g-units)
    dynamic:guidance:single_shot_pk : float (0-1)
    dynamic:guidance:intercept_achieved : float (1 or 0)
    dynamic:guidance:final_range : float (m)
    dynamic:guidance:flight_time : float (s)
    """

    def setup(self):
        # ---- Missile state inputs ----
        self.add_input(Guidance.Missile.POSITION_X, val=0.0, units='m',
                       desc='Missile initial X position')
        self.add_input(Guidance.Missile.POSITION_Y, val=0.0, units='m',
                       desc='Missile initial Y position')
        self.add_input(Guidance.Missile.VELOCITY, val=1000.0, units='m/s',
                       desc='Missile initial velocity')
        self.add_input(Guidance.Missile.HEADING, val=45.0, units='deg',
                       desc='Missile initial heading (0=East, 90=North)')
        self.add_input(Guidance.Missile.MAX_ACCELERATION, val=300.0,
                       units='m/s**2',
                       desc='Maximum lateral acceleration capability')
        self.add_input(Guidance.Missile.DRAG_COEFFICIENT, val=0.4,
                       desc='Missile zero-lift drag coefficient')
        self.add_input(Guidance.Missile.REFERENCE_AREA, val=0.0324,
                       units='m**2', desc='Missile reference area')
        self.add_input(Guidance.Missile.MASS, val=700.0, units='kg',
                       desc='Missile mass')
        self.add_input(Guidance.Missile.THRUST, val=0.0, units='N',
                       desc='Missile thrust (0 if coasting)')
        self.add_input(Guidance.Missile.BURN_TIME, val=0.0, units='s',
                       desc='Remaining burn time')

        # ---- Target state inputs ----
        self.add_input(Guidance.Target.POSITION_X, val=15000.0, units='m',
                       desc='Target initial X position')
        self.add_input(Guidance.Target.POSITION_Y, val=15000.0, units='m',
                       desc='Target initial Y position')
        self.add_input(Guidance.Target.VELOCITY, val=300.0, units='m/s',
                       desc='Target velocity')
        self.add_input(Guidance.Target.HEADING, val=180.0, units='deg',
                       desc='Target heading (0=East, 90=North)')
        self.add_input(Guidance.Target.ACCELERATION_MAX, val=50.0,
                       units='m/s**2',
                       desc='Maximum target maneuver acceleration')
        self.add_input(Guidance.Target.MANEUVER_TIME, val=5.0, units='s',
                       desc='Time when target begins maneuvering')

        # ---- Navigation inputs ----
        self.add_input(Guidance.Navigation.RATIO, val=4.0,
                       desc="Navigation ratio N' (3-5 typical)")
        self.add_discrete_input(Guidance.Navigation.LAW, val='PN',
                                desc='Guidance law: PN, APN, OGL')

        # ---- Autopilot inputs ----
        self.add_input(Guidance.Autopilot.TIME_CONSTANT, val=0.2, units='s',
                       desc='Autopilot + actuator time constant')

        # ---- Environment inputs ----
        self.add_input(Guidance.Environment.ALTITUDE, val=6096.0, units='m',
                       desc='Engagement altitude')
        self.add_input(Guidance.Environment.DENSITY, val=0.66,
                       units='kg/m**3',
                       desc='Air density at engagement altitude')

        # ---- Engagement performance outputs ----
        self.add_output(DynamicGuidance.MISS_DISTANCE, val=0.0, units='m',
                        desc='Closest approach distance', lower=0.0)
        self.add_output(DynamicGuidance.MAX_LATERAL_ACCEL, val=0.0,
                        units='m/s**2',
                        desc='Maximum achieved lateral acceleration',
                        lower=0.0)
        self.add_output(DynamicGuidance.MAX_COMMANDED_ACCEL, val=0.0,
                        units='m/s**2',
                        desc='Maximum commanded acceleration', lower=0.0)
        self.add_output(DynamicGuidance.TIME_TO_INTERCEPT, val=0.0,
                        units='s',
                        desc='Time to closest approach', lower=0.0)
        self.add_output(DynamicGuidance.ZERO_EFFORT_MISS, val=0.0,
                        units='m', desc='Zero effort miss at start')
        self.add_output(DynamicGuidance.DIVERT_REQUIREMENT, val=0.0,
                        units='m/s',
                        desc='Total delta-V from guidance', lower=0.0)
        self.add_output(DynamicGuidance.CLOSEST_APPROACH, val=0.0,
                        units='m', desc='Minimum range', lower=0.0)
        self.add_output(DynamicGuidance.REQUIRED_LOAD_FACTOR, val=0.0,
                        desc='Peak load factor (g-units)', lower=0.0)
        self.add_output(DynamicGuidance.SINGLE_SHOT_PK, val=0.0,
                        desc='Single-shot probability of kill (0-1)',
                        lower=0.0, upper=1.0)
        self.add_output(DynamicGuidance.INTERCEPT_ACHIEVED, val=0.0,
                        desc='Intercept achieved (1=yes, 0=no)',
                        lower=0.0, upper=1.0)
        self.add_output(DynamicGuidance.FINAL_RANGE, val=0.0, units='m',
                        desc='Range at simulation end', lower=0.0)
        self.add_output(DynamicGuidance.FLIGHT_TIME, val=0.0, units='s',
                        desc='Total flight time', lower=0.0)

    def setup_partials(self):
        # ODE solver is a black box -- finite differences only.
        # Exclude the discrete guidance law input from the partial
        # declaration since OpenMDAO handles discrete variables separately.
        continuous_inputs = [
            Guidance.Missile.POSITION_X,
            Guidance.Missile.POSITION_Y,
            Guidance.Missile.VELOCITY,
            Guidance.Missile.HEADING,
            Guidance.Missile.MAX_ACCELERATION,
            Guidance.Missile.DRAG_COEFFICIENT,
            Guidance.Missile.REFERENCE_AREA,
            Guidance.Missile.MASS,
            Guidance.Missile.THRUST,
            Guidance.Missile.BURN_TIME,
            Guidance.Target.POSITION_X,
            Guidance.Target.POSITION_Y,
            Guidance.Target.VELOCITY,
            Guidance.Target.HEADING,
            Guidance.Target.ACCELERATION_MAX,
            Guidance.Target.MANEUVER_TIME,
            Guidance.Navigation.RATIO,
            Guidance.Autopilot.TIME_CONSTANT,
            Guidance.Environment.ALTITUDE,
            Guidance.Environment.DENSITY,
        ]
        self.declare_partials('*', continuous_inputs, method='fd', step=1e-3)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # ==================================================================
        # 1. Unpack inputs
        # ==================================================================
        x_m0 = float(inputs[Guidance.Missile.POSITION_X])
        y_m0 = float(inputs[Guidance.Missile.POSITION_Y])
        V_m0 = float(inputs[Guidance.Missile.VELOCITY])
        hdg_m_deg = float(inputs[Guidance.Missile.HEADING])
        max_accel = float(inputs[Guidance.Missile.MAX_ACCELERATION])
        CD = float(inputs[Guidance.Missile.DRAG_COEFFICIENT])
        S_ref = float(inputs[Guidance.Missile.REFERENCE_AREA])
        mass = float(inputs[Guidance.Missile.MASS])
        thrust = float(inputs[Guidance.Missile.THRUST])
        burn_time = float(inputs[Guidance.Missile.BURN_TIME])

        x_t0 = float(inputs[Guidance.Target.POSITION_X])
        y_t0 = float(inputs[Guidance.Target.POSITION_Y])
        V_t0 = float(inputs[Guidance.Target.VELOCITY])
        hdg_t_deg = float(inputs[Guidance.Target.HEADING])
        a_t_max = float(inputs[Guidance.Target.ACCELERATION_MAX])
        maneuver_time = float(inputs[Guidance.Target.MANEUVER_TIME])

        N = float(inputs[Guidance.Navigation.RATIO])
        law = discrete_inputs[Guidance.Navigation.LAW]

        tau = float(inputs[Guidance.Autopilot.TIME_CONSTANT])
        rho = float(inputs[Guidance.Environment.DENSITY])

        # Convert headings from degrees to radians
        theta_m0 = np.radians(hdg_m_deg)
        theta_t0 = np.radians(hdg_t_deg)

        # Sanitise to prevent degenerate solver conditions
        V_m0 = max(V_m0, _MIN_SPEED_GUARD)
        V_t0 = max(V_t0, 0.0)
        mass = max(mass, 1.0)
        S_ref = max(S_ref, 1.0e-6)
        rho = max(rho, 0.001)
        max_accel = max(max_accel, 1.0)

        # ==================================================================
        # 2. Compute initial engagement geometry for ZEM
        # ==================================================================
        dx0 = x_t0 - x_m0
        dy0 = y_t0 - y_m0
        R0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        R0_safe = max(R0, _MIN_RANGE_GUARD)

        Vmx0 = V_m0 * np.cos(theta_m0)
        Vmy0 = V_m0 * np.sin(theta_m0)
        Vtx0 = V_t0 * np.cos(theta_t0)
        Vty0 = V_t0 * np.sin(theta_t0)

        Vrx0 = Vtx0 - Vmx0
        Vry0 = Vty0 - Vmy0

        R_dot0 = (dx0 * Vrx0 + dy0 * Vry0) / R0_safe
        Vc0 = max(-R_dot0, _MIN_VC_GUARD)
        tgo0 = R0_safe / Vc0

        R0_sq_safe = max(R0_safe * R0_safe, _MIN_RANGE_SQ_GUARD)
        lam_dot0 = (dx0 * Vry0 - dy0 * Vrx0) / R0_sq_safe
        ZEM_0 = R0_safe * lam_dot0 * tgo0

        # ==================================================================
        # 3. Build params dict and initial state
        # ==================================================================
        params = {
            'N': N,
            'law': law,
            'tau': tau,
            'max_accel': max_accel,
            'rho': rho,
            'CD': CD,
            'S_ref': S_ref,
            'mass': mass,
            'thrust': thrust,
            'burn_time': burn_time,
            'a_t_max': a_t_max,
            'maneuver_time': maneuver_time,
            '_a_cmd_history': [],  # populated during integration
        }

        y0 = [
            x_m0,       # 0: x_m
            y_m0,       # 1: y_m
            V_m0,       # 2: V_m
            theta_m0,   # 3: theta_m
            x_t0,       # 4: x_t
            y_t0,       # 5: y_t
            V_t0,       # 6: V_t
            theta_t0,   # 7: theta_t
            0.0,        # 8: a_achieved (starts at zero)
        ]

        # ==================================================================
        # 4. Integrate
        # ==================================================================
        try:
            event_fn = _make_range_increasing_event(params)

            sol = solve_ivp(
                fun=lambda t, y: _engagement_ode(t, y, params),
                t_span=(0.0, _MAX_SIM_TIME),
                y0=y0,
                method='RK45',
                events=[event_fn],
                max_step=0.01,
                rtol=1e-8,
                atol=1e-10,
                dense_output=False,
            )

            if not sol.success and sol.t.size < 2:
                raise RuntimeError(sol.message)

        except Exception:
            # Solver failed -- return safe fallback values
            outputs[DynamicGuidance.MISS_DISTANCE] = _MISS_SENTINEL
            outputs[DynamicGuidance.MAX_LATERAL_ACCEL] = 0.0
            outputs[DynamicGuidance.MAX_COMMANDED_ACCEL] = 0.0
            outputs[DynamicGuidance.TIME_TO_INTERCEPT] = _MAX_SIM_TIME
            outputs[DynamicGuidance.ZERO_EFFORT_MISS] = abs(ZEM_0)
            outputs[DynamicGuidance.DIVERT_REQUIREMENT] = 0.0
            outputs[DynamicGuidance.CLOSEST_APPROACH] = _MISS_SENTINEL
            outputs[DynamicGuidance.REQUIRED_LOAD_FACTOR] = 0.0
            outputs[DynamicGuidance.SINGLE_SHOT_PK] = 0.0
            outputs[DynamicGuidance.INTERCEPT_ACHIEVED] = 0.0
            outputs[DynamicGuidance.FINAL_RANGE] = _MISS_SENTINEL
            outputs[DynamicGuidance.FLIGHT_TIME] = _MAX_SIM_TIME
            return

        # ==================================================================
        # 5. Post-process trajectory
        # ==================================================================
        x_m_traj = sol.y[0]
        y_m_traj = sol.y[1]
        x_t_traj = sol.y[4]
        y_t_traj = sol.y[5]

        # Range history
        R_traj = np.sqrt(
            (x_t_traj - x_m_traj) ** 2 + (y_t_traj - y_m_traj) ** 2
        )

        idx_min = int(np.argmin(R_traj))
        miss_distance = float(R_traj[idx_min])
        time_to_intercept = float(sol.t[idx_min])

        # Final range (last integration point)
        final_range = float(R_traj[-1])
        flight_time = float(sol.t[-1])

        # Achieved lateral acceleration history
        a_achieved_traj = sol.y[8]
        max_lateral = float(np.max(np.abs(a_achieved_traj)))
        required_g = max_lateral / _G

        # Maximum commanded acceleration (tracked during integration)
        if params['_a_cmd_history']:
            max_commanded = float(max(params['_a_cmd_history']))
        else:
            max_commanded = 0.0

        # Divert requirement: integral of |a_achieved| dt
        if sol.t.size > 1:
            dt_traj = np.diff(sol.t)
            divert = float(np.sum(np.abs(a_achieved_traj[:-1]) * dt_traj))
        else:
            divert = 0.0

        # Closest approach (same as miss distance from range history)
        closest_approach = miss_distance

        # Single-shot Pk (Gaussian miss model with lethal radius)
        if miss_distance > 1.0e-12:
            Pk = 1.0 - np.exp(-(_R_LETHAL / miss_distance) ** 2)
        else:
            Pk = 1.0
        Pk = float(np.clip(Pk, 0.0, 1.0))

        # Intercept achieved
        intercept = 1.0 if miss_distance < _R_LETHAL else 0.0

        # ==================================================================
        # 6. Pack outputs
        # ==================================================================
        outputs[DynamicGuidance.MISS_DISTANCE] = miss_distance
        outputs[DynamicGuidance.MAX_LATERAL_ACCEL] = max_lateral
        outputs[DynamicGuidance.MAX_COMMANDED_ACCEL] = max_commanded
        outputs[DynamicGuidance.TIME_TO_INTERCEPT] = time_to_intercept
        outputs[DynamicGuidance.ZERO_EFFORT_MISS] = abs(ZEM_0)
        outputs[DynamicGuidance.DIVERT_REQUIREMENT] = divert
        outputs[DynamicGuidance.CLOSEST_APPROACH] = closest_approach
        outputs[DynamicGuidance.REQUIRED_LOAD_FACTOR] = required_g
        outputs[DynamicGuidance.SINGLE_SHOT_PK] = Pk
        outputs[DynamicGuidance.INTERCEPT_ACHIEVED] = intercept
        outputs[DynamicGuidance.FINAL_RANGE] = final_range
        outputs[DynamicGuidance.FLIGHT_TIME] = flight_time
