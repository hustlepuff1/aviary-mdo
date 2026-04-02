"""
RocketPy 6-DOF Flight wrapper -- OpenMDAO ExplicitComponent.

Builds a RocketPy Environment, SolidMotor, Rocket, and Flight from
OpenMDAO inputs, runs the full 6-DOF trajectory simulation, and
extracts scalar performance outputs for the optimizer.

Finite-difference partials are used because the RocketPy simulation
is a black-box time-marching solver with no analytic derivatives.
"""

import warnings

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)

# Lazy-import RocketPy symbols so the module can be loaded even if
# RocketPy is temporarily unavailable (e.g. during static analysis).
_ROCKETPY_AVAILABLE = True
try:
    from rocketpy import Environment, Flight, Rocket, SolidMotor
except ImportError:
    _ROCKETPY_AVAILABLE = False


class RocketPyFlightComp(om.ExplicitComponent):
    """
    OpenMDAO component wrapping a full RocketPy 6-DOF flight simulation.

    Given missile geometry, motor definition, environment, and launch
    conditions, this component builds the RocketPy object hierarchy and
    runs a Flight simulation.  Scalar performance outputs (apogee, range,
    max Mach, stability margins, etc.) are extracted for use by the
    Aviary optimizer.

    Notes
    -----
    - Finite-difference partials (``method='fd'``) with a step of 1e-4 are
      used because RocketPy is a black-box ODE integrator.
    - If the RocketPy simulation fails for any input combination the
      component prints a short warning and fills outputs with NaN / 0
      instead of raising, so the optimizer can recover gracefully.
    """

    def initialize(self):
        self.options.declare(
            'max_flight_time', default=600.0, types=float,
            desc='Maximum simulation time in seconds.',
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self):
        # ----- Body geometry -----
        self.add_input(Missile6DOF.Body.LENGTH, val=3.658, units='m')
        self.add_input(Missile6DOF.Body.RADIUS, val=0.1016, units='m')
        self.add_input(Missile6DOF.Body.MASS, val=100.0, units='kg')
        self.add_input(Missile6DOF.Body.INERTIA_I, val=50.0, units='kg*m**2')
        self.add_input(Missile6DOF.Body.INERTIA_Z, val=0.5, units='kg*m**2')
        self.add_input(Missile6DOF.Body.CG_WITHOUT_MOTOR, val=1.83, units='m')

        # ----- Nose cone -----
        self.add_input(Missile6DOF.NoseCone.LENGTH, val=0.488, units='m')
        self.add_discrete_input(Missile6DOF.NoseCone.KIND, val='ogive')
        self.add_input(Missile6DOF.NoseCone.BASE_RADIUS, val=0.1016, units='m')
        self.add_input(Missile6DOF.NoseCone.BLUFFNESS, val=0.0)

        # ----- Fins -----
        self.add_input(Missile6DOF.Fins.NUMBER, val=4.0)
        self.add_input(Missile6DOF.Fins.ROOT_CHORD, val=0.254, units='m')
        self.add_input(Missile6DOF.Fins.TIP_CHORD, val=0.076, units='m')
        self.add_input(Missile6DOF.Fins.SPAN, val=0.127, units='m')
        self.add_input(Missile6DOF.Fins.POSITION, val=3.2, units='m')
        self.add_input(Missile6DOF.Fins.CANT_ANGLE, val=0.0, units='deg')
        self.add_input(Missile6DOF.Fins.SWEEP_LENGTH, val=0.127, units='m')

        # ----- Tail -----
        self.add_input(Missile6DOF.Tail.TOP_RADIUS, val=0.1016, units='m')
        self.add_input(Missile6DOF.Tail.BOTTOM_RADIUS, val=0.0762, units='m')
        self.add_input(Missile6DOF.Tail.LENGTH, val=0.1, units='m')
        self.add_input(Missile6DOF.Tail.POSITION, val=3.558, units='m')

        # ----- Motor -----
        self.add_input(Missile6DOF.Motor.BURN_TIME, val=3.26, units='s')
        self.add_input(Missile6DOF.Motor.DRY_MASS, val=10.0, units='kg')
        self.add_input(Missile6DOF.Motor.DRY_INERTIA_I, val=5.0, units='kg*m**2')
        self.add_input(Missile6DOF.Motor.DRY_INERTIA_Z, val=0.1, units='kg*m**2')
        self.add_input(Missile6DOF.Motor.CENTER_OF_DRY_MASS, val=0.5, units='m')
        self.add_input(Missile6DOF.Motor.NOZZLE_RADIUS, val=0.0508, units='m')
        self.add_input(Missile6DOF.Motor.NOZZLE_POSITION, val=-1.0, units='m')

        # ----- Solid grain -----
        self.add_input(Missile6DOF.SolidGrain.PROPELLANT_DENSITY,
                       val=1750.0, units='kg/m**3')
        self.add_input(Missile6DOF.SolidGrain.NUMBER, val=4.0)
        self.add_input(Missile6DOF.SolidGrain.OUTER_RADIUS,
                       val=0.0889, units='m')
        self.add_input(Missile6DOF.SolidGrain.INNER_RADIUS,
                       val=0.0254, units='m')
        self.add_input(Missile6DOF.SolidGrain.HEIGHT, val=0.254, units='m')

        # ----- Environment -----
        self.add_input(Missile6DOF.Environment.LATITUDE, val=32.99, units='deg')
        self.add_input(Missile6DOF.Environment.LONGITUDE, val=-106.97, units='deg')
        self.add_input(Missile6DOF.Environment.ELEVATION, val=1400.0, units='m')

        # ----- Launch conditions -----
        self.add_input(Missile6DOF.Launch.RAIL_LENGTH, val=5.0, units='m')
        self.add_input(Missile6DOF.Launch.INCLINATION, val=85.0, units='deg')
        self.add_input(Missile6DOF.Launch.HEADING, val=0.0, units='deg')

        # ----- Thrust level (flat-curve magnitude) -----
        self.add_input(Missile6DOF.Motor.THRUST_SOURCE, val=10000.0, units='N')

        # ===== Outputs =====
        # Trajectory
        self.add_output(DynamicMissile6DOF.APOGEE, val=0.0, units='m')
        self.add_output(DynamicMissile6DOF.APOGEE_TIME, val=0.0, units='s')
        self.add_output(DynamicMissile6DOF.MAX_SPEED, val=0.0, units='m/s')
        self.add_output(DynamicMissile6DOF.MAX_MACH, val=0.0)
        self.add_output(DynamicMissile6DOF.MAX_ACCELERATION,
                        val=0.0, units='m/s**2')
        self.add_output(DynamicMissile6DOF.MAX_DYNAMIC_PRESSURE,
                        val=0.0, units='Pa')
        self.add_output(DynamicMissile6DOF.FLIGHT_TIME, val=0.0, units='s')
        self.add_output(DynamicMissile6DOF.IMPACT_VELOCITY,
                        val=0.0, units='m/s')
        self.add_output(DynamicMissile6DOF.RANGE_X, val=0.0, units='m')
        self.add_output(DynamicMissile6DOF.RANGE_Y, val=0.0, units='m')
        self.add_output(DynamicMissile6DOF.RANGE_TOTAL, val=0.0, units='m')

        # Motor performance
        self.add_output(DynamicMissile6DOF.TOTAL_IMPULSE, val=0.0, units='N*s')
        self.add_output(DynamicMissile6DOF.MAX_THRUST, val=0.0, units='N')
        self.add_output(DynamicMissile6DOF.AVERAGE_THRUST, val=0.0, units='N')
        self.add_output(DynamicMissile6DOF.AVERAGE_ISP, val=0.0, units='s')
        self.add_output(DynamicMissile6DOF.PROPELLANT_MASS, val=0.0, units='kg')

        # Stability
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION, val=0.0)
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT, val=0.0)
        self.add_output(DynamicMissile6DOF.CP_POSITION, val=0.0, units='m')
        self.add_output(DynamicMissile6DOF.CG_POSITION, val=0.0, units='m')

        # Aero
        self.add_output(DynamicMissile6DOF.POWER_ON_CD, val=0.0)
        self.add_output(DynamicMissile6DOF.POWER_OFF_CD, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', step=1e-4)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def compute(self, inputs, outputs, discrete_inputs=None,
                discrete_outputs=None):
        if not _ROCKETPY_AVAILABLE:
            warnings.warn(
                "RocketPyFlightComp: rocketpy is not installed; "
                "all outputs set to NaN.",
                stacklevel=2,
            )
            for name in outputs:
                outputs[name] = np.nan
            return

        try:
            self._run_rocketpy(inputs, outputs, discrete_inputs)
        except Exception as exc:
            warnings.warn(
                f"RocketPyFlightComp: flight simulation failed "
                f"({type(exc).__name__}: {exc}). Outputs set to NaN.",
                stacklevel=2,
            )
            for name in outputs:
                outputs[name] = np.nan

    # ------------------------------------------------------------------
    # Internal: build objects and run simulation
    # ------------------------------------------------------------------
    def _run_rocketpy(self, inputs, outputs, discrete_inputs):
        """Build RocketPy objects from inputs and run the Flight."""

        # --- Helper to pull scalar from OpenMDAO input array ---
        def _s(name):
            """Return a Python float from an OpenMDAO input."""
            v = inputs[name]
            return float(np.atleast_1d(v).ravel()[0])

        # ============================================================
        # 1. Environment
        # ============================================================
        lat = _s(Missile6DOF.Environment.LATITUDE)
        lon = _s(Missile6DOF.Environment.LONGITUDE)
        elev = _s(Missile6DOF.Environment.ELEVATION)

        env = Environment(latitude=lat, longitude=lon, elevation=elev)
        env.set_atmospheric_model(type="standard_atmosphere")

        # ============================================================
        # 2. Solid Motor
        # ============================================================
        thrust_N = _s(Missile6DOF.Motor.THRUST_SOURCE)
        burn_time = _s(Missile6DOF.Motor.BURN_TIME)
        dry_mass = _s(Missile6DOF.Motor.DRY_MASS)
        dry_I_i = _s(Missile6DOF.Motor.DRY_INERTIA_I)
        dry_I_z = _s(Missile6DOF.Motor.DRY_INERTIA_Z)
        center_of_dry_mass = _s(Missile6DOF.Motor.CENTER_OF_DRY_MASS)
        nozzle_radius = _s(Missile6DOF.Motor.NOZZLE_RADIUS)
        nozzle_position = _s(Missile6DOF.Motor.NOZZLE_POSITION)
        n_grains = int(round(_s(Missile6DOF.SolidGrain.NUMBER)))
        propellant_density = _s(Missile6DOF.SolidGrain.PROPELLANT_DENSITY)
        outer_radius = _s(Missile6DOF.SolidGrain.OUTER_RADIUS)
        inner_radius = _s(Missile6DOF.SolidGrain.INNER_RADIUS)
        grain_height = _s(Missile6DOF.SolidGrain.HEIGHT)

        motor = SolidMotor(
            thrust_source=[[0, thrust_N], [burn_time, thrust_N]],
            dry_mass=dry_mass,
            dry_inertia=(dry_I_i, dry_I_i, dry_I_z),
            center_of_dry_mass_position=center_of_dry_mass,
            nozzle_radius=nozzle_radius,
            nozzle_position=nozzle_position,
            burn_time=burn_time,
            grain_number=n_grains,
            grain_density=propellant_density,
            grain_outer_radius=outer_radius,
            grain_initial_inner_radius=inner_radius,
            grain_initial_height=grain_height,
            grain_separation=0.0,
            grains_center_of_mass_position=0.0,
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

        # ============================================================
        # 3. Rocket
        # ============================================================
        body_length = _s(Missile6DOF.Body.LENGTH)
        body_radius = _s(Missile6DOF.Body.RADIUS)
        body_mass = _s(Missile6DOF.Body.MASS)
        inertia_i = _s(Missile6DOF.Body.INERTIA_I)
        inertia_z = _s(Missile6DOF.Body.INERTIA_Z)
        cg_without_motor = _s(Missile6DOF.Body.CG_WITHOUT_MOTOR)

        rocket = Rocket(
            radius=body_radius,
            mass=body_mass,
            inertia=(inertia_i, inertia_i, inertia_z),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=cg_without_motor,
            coordinate_system_orientation="nose_to_tail",
        )

        # Motor position: nozzle at aft end (nose_to_tail convention)
        motor_pos = -(body_length - abs(nozzle_position))
        rocket.add_motor(motor, position=motor_pos)

        # Nose cone
        nose_length = _s(Missile6DOF.NoseCone.LENGTH)
        nose_kind = discrete_inputs[Missile6DOF.NoseCone.KIND]
        nose_base_r = _s(Missile6DOF.NoseCone.BASE_RADIUS)
        nose_bluffness = _s(Missile6DOF.NoseCone.BLUFFNESS)

        rocket.add_nose(
            length=nose_length,
            kind=nose_kind,
            position=0.0,
            bluffness=nose_bluffness,
            base_radius=nose_base_r,
            name="Nose",
        )

        # Trapezoidal fins
        n_fins = int(round(_s(Missile6DOF.Fins.NUMBER)))
        root_chord = _s(Missile6DOF.Fins.ROOT_CHORD)
        tip_chord = _s(Missile6DOF.Fins.TIP_CHORD)
        fin_span = _s(Missile6DOF.Fins.SPAN)
        fin_position = _s(Missile6DOF.Fins.POSITION)
        cant_angle = _s(Missile6DOF.Fins.CANT_ANGLE)
        sweep_length = _s(Missile6DOF.Fins.SWEEP_LENGTH)

        rocket.add_trapezoidal_fins(
            n=n_fins,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=fin_span,
            position=fin_position,
            cant_angle=cant_angle,
            sweep_length=sweep_length,
            name="Fins",
        )

        # Tail transition
        tail_top_r = _s(Missile6DOF.Tail.TOP_RADIUS)
        tail_bot_r = _s(Missile6DOF.Tail.BOTTOM_RADIUS)
        tail_length = _s(Missile6DOF.Tail.LENGTH)
        tail_position = _s(Missile6DOF.Tail.POSITION)

        rocket.add_tail(
            top_radius=tail_top_r,
            bottom_radius=tail_bot_r,
            length=tail_length,
            position=tail_position,
            name="Tail",
        )

        # ============================================================
        # 4. Flight simulation
        # ============================================================
        rail_length = _s(Missile6DOF.Launch.RAIL_LENGTH)
        inclination = _s(Missile6DOF.Launch.INCLINATION)
        heading = _s(Missile6DOF.Launch.HEADING)

        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
            max_time=self.options['max_flight_time'],
        )

        # ============================================================
        # 5. Extract outputs
        # ============================================================

        # --- Trajectory ---
        outputs[DynamicMissile6DOF.APOGEE] = (
            float(flight.apogee) - elev
        )
        outputs[DynamicMissile6DOF.APOGEE_TIME] = float(flight.apogee_time)
        outputs[DynamicMissile6DOF.MAX_SPEED] = float(flight.max_speed)
        outputs[DynamicMissile6DOF.MAX_MACH] = float(flight.max_mach_number)
        outputs[DynamicMissile6DOF.MAX_ACCELERATION] = float(
            flight.max_acceleration
        )
        outputs[DynamicMissile6DOF.MAX_DYNAMIC_PRESSURE] = float(
            flight.max_dynamic_pressure
        )
        outputs[DynamicMissile6DOF.FLIGHT_TIME] = float(flight.t_final)

        impact_vel = getattr(flight, 'impact_velocity', 0.0)
        outputs[DynamicMissile6DOF.IMPACT_VELOCITY] = abs(float(impact_vel))

        x_impact = float(getattr(flight, 'x_impact', 0.0))
        y_impact = float(getattr(flight, 'y_impact', 0.0))
        outputs[DynamicMissile6DOF.RANGE_X] = x_impact
        outputs[DynamicMissile6DOF.RANGE_Y] = y_impact
        outputs[DynamicMissile6DOF.RANGE_TOTAL] = (
            (x_impact**2 + y_impact**2) ** 0.5
        )

        # --- Motor performance ---
        total_impulse = float(motor.total_impulse)
        outputs[DynamicMissile6DOF.TOTAL_IMPULSE] = total_impulse

        max_thrust_val = float(getattr(motor, 'max_thrust', thrust_N))
        outputs[DynamicMissile6DOF.MAX_THRUST] = max_thrust_val

        # Average thrust = total impulse / burn duration
        burn_duration = motor.burn_time[1] - motor.burn_time[0]
        if burn_duration > 0:
            avg_thrust = total_impulse / burn_duration
        else:
            avg_thrust = 0.0
        outputs[DynamicMissile6DOF.AVERAGE_THRUST] = avg_thrust

        # Average Isp = total impulse / (propellant_mass * g0)
        prop_mass = float(motor.propellant_initial_mass)
        outputs[DynamicMissile6DOF.PROPELLANT_MASS] = prop_mass

        g0 = 9.80665
        if prop_mass > 0:
            avg_isp = total_impulse / (prop_mass * g0)
        else:
            avg_isp = 0.0
        outputs[DynamicMissile6DOF.AVERAGE_ISP] = avg_isp

        # --- Stability ---
        # static_margin is a Function(time) returning calibers
        sm_func = rocket.static_margin
        t_start = motor.burn_time[0]
        t_end = motor.burn_time[1]
        outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION] = float(
            sm_func(t_start)
        )
        outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT] = float(
            sm_func(t_end)
        )

        # CP and CG at ignition
        outputs[DynamicMissile6DOF.CP_POSITION] = float(
            rocket.cp_position(0)
        )
        outputs[DynamicMissile6DOF.CG_POSITION] = float(
            rocket.center_of_mass(0)
        )

        # --- Aero (constant drag model values) ---
        outputs[DynamicMissile6DOF.POWER_ON_CD] = float(
            rocket.power_on_drag(0)
        )
        outputs[DynamicMissile6DOF.POWER_OFF_CD] = float(
            rocket.power_off_drag(0)
        )
