"""
RocketPy aerodynamic coefficient extraction component.

Builds a RocketPy Rocket with NoseCone, TrapezoidalFins, and Tail surfaces
to extract aerodynamic properties (drag coefficients, center of pressure,
static margin) via the Barrowman method implemented in RocketPy.

Uses a minimal dummy motor so the Rocket object can be instantiated.
"""

import openmdao.api as om
from rocketpy import Rocket, SolidMotor

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)

# Default reference Mach number for drag coefficient extraction
_REF_MACH = 0.8


class RocketPyAero(om.ExplicitComponent):
    """Extract aerodynamic coefficients from RocketPy geometry model.

    Constructs a RocketPy Rocket with a minimal solid motor, adds the
    three aero surfaces (NoseCone, TrapezoidalFins, Tail), then queries
    the model for drag coefficients, CP location, and static margins.

    Notes
    -----
    * Drag coefficients are constants passed through (RocketPy's surface
      models compute stability, not parasitic drag). The power_on_drag
      and power_off_drag constructor values are returned at the reference
      Mach number.
    * Static margin is reported at t=0 (ignition) and t=burn_time
      (burnout) using the dummy motor's burn profile.
    * Finite-difference partials are used because RocketPy objects do not
      support complex-step differentiation.
    """

    def initialize(self):
        self.options.declare(
            'ref_mach', default=_REF_MACH, types=float,
            desc='Reference Mach number for drag coefficient evaluation.',
        )

    def setup(self):
        # ---- geometry inputs ----
        # Body
        self.add_input(Missile6DOF.Body.LENGTH, val=3.0, units='m',
                       desc='Total body length')
        self.add_input(Missile6DOF.Body.RADIUS, val=0.1, units='m',
                       desc='Body tube outer radius')

        # NoseCone
        self.add_input(Missile6DOF.NoseCone.LENGTH, val=0.3, units='m',
                       desc='Nose cone length')
        self.add_discrete_input(Missile6DOF.NoseCone.KIND, val='ogive',
                                desc='Nose cone profile kind (ogive, conical, etc.)')
        self.add_input(Missile6DOF.NoseCone.BASE_RADIUS, val=0.1, units='m',
                       desc='Nose cone base radius')
        self.add_input(Missile6DOF.NoseCone.BLUFFNESS, val=0.0, units=None,
                       desc='Nose cone bluffness ratio')

        # Fins
        self.add_input(Missile6DOF.Fins.NUMBER, val=4.0, units=None,
                       desc='Number of fins')
        self.add_input(Missile6DOF.Fins.ROOT_CHORD, val=0.25, units='m',
                       desc='Fin root chord length')
        self.add_input(Missile6DOF.Fins.TIP_CHORD, val=0.08, units='m',
                       desc='Fin tip chord length')
        self.add_input(Missile6DOF.Fins.SPAN, val=0.12, units='m',
                       desc='Fin semi-span')
        self.add_input(Missile6DOF.Fins.POSITION, val=2.8, units='m',
                       desc='Fin position from nose tip')
        self.add_input(Missile6DOF.Fins.CANT_ANGLE, val=0.0, units='deg',
                       desc='Fin cant angle')
        self.add_input(Missile6DOF.Fins.SWEEP_LENGTH, val=0.12, units='m',
                       desc='Fin sweep length (LE root to LE tip, parallel to body)')

        # Tail (boat-tail)
        self.add_input(Missile6DOF.Tail.TOP_RADIUS, val=0.1, units='m',
                       desc='Tail section top radius')
        self.add_input(Missile6DOF.Tail.BOTTOM_RADIUS, val=0.07, units='m',
                       desc='Tail section bottom radius')
        self.add_input(Missile6DOF.Tail.LENGTH, val=0.1, units='m',
                       desc='Tail section length')
        self.add_input(Missile6DOF.Tail.POSITION, val=2.9, units='m',
                       desc='Tail section position from nose tip')

        # ---- outputs ----
        self.add_output(DynamicMissile6DOF.POWER_ON_CD, val=0.5, units=None,
                        desc='Drag coefficient at reference Mach (power on)')
        self.add_output(DynamicMissile6DOF.POWER_OFF_CD, val=0.5, units=None,
                        desc='Drag coefficient at reference Mach (power off)')
        self.add_output(DynamicMissile6DOF.CP_POSITION, val=1.5, units='m',
                        desc='Center of pressure position from nose tip')
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION,
                        val=2.0, units=None,
                        desc='Static margin at ignition (calibers)')
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT,
                        val=1.0, units=None,
                        desc='Static margin at burnout (calibers)')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None,
                discrete_outputs=None):
        ref_mach = self.options['ref_mach']

        # Unpack geometry
        body_length = float(inputs[Missile6DOF.Body.LENGTH])
        body_radius = float(inputs[Missile6DOF.Body.RADIUS])

        nose_length = float(inputs[Missile6DOF.NoseCone.LENGTH])
        nose_kind = discrete_inputs[Missile6DOF.NoseCone.KIND]
        nose_base_radius = float(inputs[Missile6DOF.NoseCone.BASE_RADIUS])
        nose_bluffness = float(inputs[Missile6DOF.NoseCone.BLUFFNESS])

        n_fins = int(round(float(inputs[Missile6DOF.Fins.NUMBER])))
        root_chord = float(inputs[Missile6DOF.Fins.ROOT_CHORD])
        tip_chord = float(inputs[Missile6DOF.Fins.TIP_CHORD])
        fin_span = float(inputs[Missile6DOF.Fins.SPAN])
        fin_position = float(inputs[Missile6DOF.Fins.POSITION])
        cant_angle = float(inputs[Missile6DOF.Fins.CANT_ANGLE])
        sweep_length = float(inputs[Missile6DOF.Fins.SWEEP_LENGTH])

        tail_top_r = float(inputs[Missile6DOF.Tail.TOP_RADIUS])
        tail_bot_r = float(inputs[Missile6DOF.Tail.BOTTOM_RADIUS])
        tail_length = float(inputs[Missile6DOF.Tail.LENGTH])
        tail_position = float(inputs[Missile6DOF.Tail.POSITION])

        try:
            rocket, burn_time = self._build_rocket(
                body_radius=body_radius,
                body_length=body_length,
                nose_length=nose_length,
                nose_kind=nose_kind,
                nose_base_radius=nose_base_radius,
                nose_bluffness=nose_bluffness,
                n_fins=n_fins,
                root_chord=root_chord,
                tip_chord=tip_chord,
                fin_span=fin_span,
                fin_position=fin_position,
                cant_angle=cant_angle,
                sweep_length=sweep_length,
                tail_top_r=tail_top_r,
                tail_bot_r=tail_bot_r,
                tail_length=tail_length,
                tail_position=tail_position,
            )

            # Extract aero properties
            outputs[DynamicMissile6DOF.POWER_ON_CD] = float(
                rocket.power_on_drag(ref_mach)
            )
            outputs[DynamicMissile6DOF.POWER_OFF_CD] = float(
                rocket.power_off_drag(ref_mach)
            )
            outputs[DynamicMissile6DOF.CP_POSITION] = float(
                rocket.cp_position(0)
            )
            outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION] = float(
                rocket.static_margin(0)
            )
            outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT] = float(
                rocket.static_margin(burn_time)
            )

        except Exception:
            # During solver iterations, geometry can go out-of-bounds.
            # Return last-good values (OpenMDAO keeps outputs from previous
            # successful iteration) and let the solver recover.
            import traceback
            import warnings
            warnings.warn(
                f"RocketPyAero: RocketPy evaluation failed, retaining "
                f"previous outputs.\n{traceback.format_exc()}",
                stacklevel=2,
            )

    # -----------------------------------------------------------------
    @staticmethod
    def _build_rocket(
        body_radius,
        body_length,
        nose_length,
        nose_kind,
        nose_base_radius,
        nose_bluffness,
        n_fins,
        root_chord,
        tip_chord,
        fin_span,
        fin_position,
        cant_angle,
        sweep_length,
        tail_top_r,
        tail_bot_r,
        tail_length,
        tail_position,
    ):
        """Build a RocketPy Rocket with a dummy motor and aero surfaces.

        Returns
        -------
        rocket : rocketpy.Rocket
            Fully configured rocket instance.
        burn_time : float
            Motor burn time (s), needed for burnout margin evaluation.
        """
        burn_time = 3.0  # dummy burn time for the minimal motor

        # Minimal solid motor -- just enough to let RocketPy build
        # mass and CG evolution so static_margin can be evaluated.
        motor = SolidMotor(
            thrust_source=[[0, 1000], [burn_time, 1000], [burn_time + 0.01, 0]],
            dry_mass=5.0,
            dry_inertia=(3.0, 3.0, 0.05),
            center_of_dry_mass_position=0.3,
            nozzle_radius=0.04,
            nozzle_position=0.0,
            grain_number=4,
            grain_density=1750,
            grain_outer_radius=0.08,
            grain_initial_inner_radius=0.02,
            grain_initial_height=0.2,
            grain_separation=0.0,
            grains_center_of_mass_position=0.3,
            coordinate_system_orientation='nozzle_to_combustion_chamber',
            burn_time=burn_time,
        )

        # CG placed at body midpoint for the dummy configuration
        cg_without_motor = body_length / 2.0

        rocket = Rocket(
            radius=body_radius,
            mass=100.0,
            inertia=(50.0, 50.0, 0.5),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=cg_without_motor,
            coordinate_system_orientation='nose_to_tail',
        )

        # Motor positioned at the aft end of the body
        motor_position = -(body_length - nose_length)
        rocket.add_motor(motor, position=motor_position)

        # Aero surfaces
        rocket.add_nose(
            length=nose_length,
            kind=nose_kind,
            position=0.0,
            bluffness=nose_bluffness,
            base_radius=nose_base_radius,
            name='Nose',
        )

        rocket.add_trapezoidal_fins(
            n=n_fins,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=fin_span,
            position=fin_position,
            cant_angle=cant_angle,
            sweep_length=sweep_length,
            name='Fins',
        )

        rocket.add_tail(
            top_radius=tail_top_r,
            bottom_radius=tail_bot_r,
            length=tail_length,
            position=tail_position,
            name='Tail',
        )

        return rocket, burn_time
