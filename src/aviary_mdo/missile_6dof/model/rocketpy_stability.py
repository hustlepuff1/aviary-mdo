"""
RocketPy stability analysis component.

Builds a full RocketPy Rocket with a user-defined SolidMotor and aero
surfaces, then evaluates stability characteristics (static margin, CG, CP)
at ignition and burnout.

Unlike the lighter ``RocketPyAero`` component (which uses a dummy motor),
this component takes real motor parameters so the CG migration and
propellant-mass-dependent static margin are physically meaningful.
"""

import openmdao.api as om
from rocketpy import Rocket, SolidMotor

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)


class RocketPyStability(om.ExplicitComponent):
    """Evaluate missile stability using a full RocketPy Rocket + Motor model.

    Constructs the complete RocketPy Rocket with a user-specified SolidMotor,
    adds the three canonical aero surfaces (NoseCone, TrapezoidalFins, Tail),
    then queries stability at two critical flight times:

    * **Ignition** (t = 0): heaviest configuration, motor fully loaded.
    * **Burnout** (t = burn_time): lightest, CG has shifted forward.

    Notes
    -----
    * The ``nose:kind`` input is discrete (string). All other inputs are
      continuous.
    * Finite-difference partials are declared because RocketPy internals
      are not compatible with complex-step.
    * A try/except guard prevents solver crashes when geometry goes
      out-of-bounds during optimisation.
    """

    def setup(self):
        # ---- body inputs ----
        self.add_input(Missile6DOF.Body.MASS, val=100.0, units='kg',
                       desc='Body dry mass without motor')
        self.add_input(Missile6DOF.Body.INERTIA_I, val=50.0, units='kg*m**2',
                       desc='Body transverse moment of inertia (I_11 = I_22)')
        self.add_input(Missile6DOF.Body.INERTIA_Z, val=0.5, units='kg*m**2',
                       desc='Body axial moment of inertia (I_33)')
        self.add_input(Missile6DOF.Body.CG_WITHOUT_MOTOR, val=1.5, units='m',
                       desc='CG position from nose, without motor')
        self.add_input(Missile6DOF.Body.RADIUS, val=0.1, units='m',
                       desc='Body tube outer radius')
        self.add_input(Missile6DOF.Body.LENGTH, val=3.0, units='m',
                       desc='Total body length')

        # ---- nose cone inputs ----
        self.add_input(Missile6DOF.NoseCone.LENGTH, val=0.3, units='m',
                       desc='Nose cone length')
        self.add_discrete_input(Missile6DOF.NoseCone.KIND, val='ogive',
                                desc='Nose cone profile kind')
        self.add_input(Missile6DOF.NoseCone.BASE_RADIUS, val=0.1, units='m',
                       desc='Nose cone base radius')

        # ---- fin inputs ----
        self.add_input(Missile6DOF.Fins.NUMBER, val=4.0, units=None,
                       desc='Number of fins')
        self.add_input(Missile6DOF.Fins.ROOT_CHORD, val=0.25, units='m',
                       desc='Fin root chord')
        self.add_input(Missile6DOF.Fins.TIP_CHORD, val=0.08, units='m',
                       desc='Fin tip chord')
        self.add_input(Missile6DOF.Fins.SPAN, val=0.12, units='m',
                       desc='Fin semi-span')
        self.add_input(Missile6DOF.Fins.POSITION, val=2.8, units='m',
                       desc='Fin position from nose tip')
        self.add_input(Missile6DOF.Fins.SWEEP_LENGTH, val=0.12, units='m',
                       desc='Fin sweep length (LE root to LE tip)')

        # ---- motor inputs ----
        self.add_input(Missile6DOF.Motor.NOZZLE_POSITION, val=0.0, units='m',
                       desc='Nozzle position in motor coordinate system')
        self.add_input(Missile6DOF.Motor.THRUST_SOURCE, val=1000.0, units='N',
                       desc='Constant thrust level (flat thrust profile)')
        self.add_input(Missile6DOF.Motor.BURN_TIME, val=3.0, units='s',
                       desc='Motor burn time')

        # ---- grain inputs ----
        self.add_input(Missile6DOF.SolidGrain.NUMBER, val=4.0, units=None,
                       desc='Number of propellant grains')
        self.add_input(Missile6DOF.SolidGrain.PROPELLANT_DENSITY,
                       val=1750.0, units='kg/m**3',
                       desc='Propellant grain density')
        self.add_input(Missile6DOF.SolidGrain.OUTER_RADIUS, val=0.08,
                       units='m', desc='Grain outer radius')
        self.add_input(Missile6DOF.SolidGrain.INNER_RADIUS, val=0.02,
                       units='m', desc='Grain initial inner radius')
        self.add_input(Missile6DOF.SolidGrain.HEIGHT, val=0.2, units='m',
                       desc='Grain initial height')

        # ---- motor dry properties ----
        self.add_input(Missile6DOF.Motor.DRY_MASS, val=5.0, units='kg',
                       desc='Motor casing dry mass')
        self.add_input(Missile6DOF.Motor.DRY_INERTIA_I, val=3.0,
                       units='kg*m**2',
                       desc='Motor dry transverse inertia')
        self.add_input(Missile6DOF.Motor.DRY_INERTIA_Z, val=0.05,
                       units='kg*m**2',
                       desc='Motor dry axial inertia')
        self.add_input(Missile6DOF.Motor.CENTER_OF_DRY_MASS, val=0.3,
                       units='m',
                       desc='Motor dry CG in motor coordinates')
        self.add_input(Missile6DOF.Motor.NOZZLE_RADIUS, val=0.04, units='m',
                       desc='Motor nozzle exit radius')

        # ---- tail inputs ----
        self.add_input(Missile6DOF.Tail.TOP_RADIUS, val=0.1, units='m',
                       desc='Tail section top radius')
        self.add_input(Missile6DOF.Tail.BOTTOM_RADIUS, val=0.07, units='m',
                       desc='Tail section bottom radius')
        self.add_input(Missile6DOF.Tail.LENGTH, val=0.1, units='m',
                       desc='Tail section length')
        self.add_input(Missile6DOF.Tail.POSITION, val=2.9, units='m',
                       desc='Tail section position from nose tip')

        # ---- outputs ----
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION,
                        val=2.0, units=None,
                        desc='Static margin at ignition (calibers)')
        self.add_output(DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT,
                        val=1.0, units=None,
                        desc='Static margin at burnout (calibers)')
        self.add_output(DynamicMissile6DOF.CP_POSITION, val=1.5, units='m',
                        desc='Center of pressure from nose tip')
        self.add_output(DynamicMissile6DOF.CG_POSITION, val=1.5, units='m',
                        desc='CG position at ignition from nose tip')

        # Additional CG at burnout (no pre-existing variable -- use a local
        # name following the same convention)
        self.add_output('dynamic:missile_6dof:cg_position_burnout',
                        val=1.5, units='m',
                        desc='CG position at burnout from nose tip')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None,
                discrete_outputs=None):
        # Unpack body
        body_mass = float(inputs[Missile6DOF.Body.MASS])
        body_I_i = float(inputs[Missile6DOF.Body.INERTIA_I])
        body_I_z = float(inputs[Missile6DOF.Body.INERTIA_Z])
        cg_without_motor = float(inputs[Missile6DOF.Body.CG_WITHOUT_MOTOR])
        body_radius = float(inputs[Missile6DOF.Body.RADIUS])
        body_length = float(inputs[Missile6DOF.Body.LENGTH])

        # Unpack nose
        nose_length = float(inputs[Missile6DOF.NoseCone.LENGTH])
        nose_kind = discrete_inputs[Missile6DOF.NoseCone.KIND]
        nose_base_radius = float(inputs[Missile6DOF.NoseCone.BASE_RADIUS])

        # Unpack fins
        n_fins = int(round(float(inputs[Missile6DOF.Fins.NUMBER])))
        root_chord = float(inputs[Missile6DOF.Fins.ROOT_CHORD])
        tip_chord = float(inputs[Missile6DOF.Fins.TIP_CHORD])
        fin_span = float(inputs[Missile6DOF.Fins.SPAN])
        fin_position = float(inputs[Missile6DOF.Fins.POSITION])
        sweep_length = float(inputs[Missile6DOF.Fins.SWEEP_LENGTH])

        # Unpack motor
        nozzle_position = float(inputs[Missile6DOF.Motor.NOZZLE_POSITION])
        thrust_level = float(inputs[Missile6DOF.Motor.THRUST_SOURCE])
        burn_time = float(inputs[Missile6DOF.Motor.BURN_TIME])

        # Unpack grain
        grain_number = int(round(float(inputs[Missile6DOF.SolidGrain.NUMBER])))
        grain_density = float(inputs[Missile6DOF.SolidGrain.PROPELLANT_DENSITY])
        grain_outer_r = float(inputs[Missile6DOF.SolidGrain.OUTER_RADIUS])
        grain_inner_r = float(inputs[Missile6DOF.SolidGrain.INNER_RADIUS])
        grain_height = float(inputs[Missile6DOF.SolidGrain.HEIGHT])

        # Unpack motor dry properties
        motor_dry_mass = float(inputs[Missile6DOF.Motor.DRY_MASS])
        motor_dry_I_i = float(inputs[Missile6DOF.Motor.DRY_INERTIA_I])
        motor_dry_I_z = float(inputs[Missile6DOF.Motor.DRY_INERTIA_Z])
        motor_dry_cg = float(inputs[Missile6DOF.Motor.CENTER_OF_DRY_MASS])
        nozzle_radius = float(inputs[Missile6DOF.Motor.NOZZLE_RADIUS])

        # Unpack tail
        tail_top_r = float(inputs[Missile6DOF.Tail.TOP_RADIUS])
        tail_bot_r = float(inputs[Missile6DOF.Tail.BOTTOM_RADIUS])
        tail_length = float(inputs[Missile6DOF.Tail.LENGTH])
        tail_position = float(inputs[Missile6DOF.Tail.POSITION])

        try:
            rocket = self._build_rocket(
                body_mass=body_mass,
                body_I_i=body_I_i,
                body_I_z=body_I_z,
                cg_without_motor=cg_without_motor,
                body_radius=body_radius,
                body_length=body_length,
                nose_length=nose_length,
                nose_kind=nose_kind,
                nose_base_radius=nose_base_radius,
                n_fins=n_fins,
                root_chord=root_chord,
                tip_chord=tip_chord,
                fin_span=fin_span,
                fin_position=fin_position,
                sweep_length=sweep_length,
                nozzle_position=nozzle_position,
                thrust_level=thrust_level,
                burn_time=burn_time,
                grain_number=grain_number,
                grain_density=grain_density,
                grain_outer_r=grain_outer_r,
                grain_inner_r=grain_inner_r,
                grain_height=grain_height,
                motor_dry_mass=motor_dry_mass,
                motor_dry_I_i=motor_dry_I_i,
                motor_dry_I_z=motor_dry_I_z,
                motor_dry_cg=motor_dry_cg,
                nozzle_radius=nozzle_radius,
                tail_top_r=tail_top_r,
                tail_bot_r=tail_bot_r,
                tail_length=tail_length,
                tail_position=tail_position,
            )

            # Evaluate stability at ignition and burnout
            outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION] = float(
                rocket.static_margin(0)
            )
            outputs[DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT] = float(
                rocket.static_margin(burn_time)
            )
            outputs[DynamicMissile6DOF.CP_POSITION] = float(
                rocket.cp_position(0)
            )
            outputs[DynamicMissile6DOF.CG_POSITION] = float(
                rocket.center_of_mass(0)
            )
            outputs['dynamic:missile_6dof:cg_position_burnout'] = float(
                rocket.center_of_mass(burn_time)
            )

        except Exception:
            import traceback
            import warnings
            warnings.warn(
                f"RocketPyStability: RocketPy evaluation failed, retaining "
                f"previous outputs.\n{traceback.format_exc()}",
                stacklevel=2,
            )

    # -----------------------------------------------------------------
    @staticmethod
    def _build_rocket(
        body_mass,
        body_I_i,
        body_I_z,
        cg_without_motor,
        body_radius,
        body_length,
        nose_length,
        nose_kind,
        nose_base_radius,
        n_fins,
        root_chord,
        tip_chord,
        fin_span,
        fin_position,
        sweep_length,
        nozzle_position,
        thrust_level,
        burn_time,
        grain_number,
        grain_density,
        grain_outer_r,
        grain_inner_r,
        grain_height,
        motor_dry_mass,
        motor_dry_I_i,
        motor_dry_I_z,
        motor_dry_cg,
        nozzle_radius,
        tail_top_r,
        tail_bot_r,
        tail_length,
        tail_position,
    ):
        """Build a complete RocketPy Rocket with real motor and aero surfaces.

        Parameters
        ----------
        (all scalar, see ``setup`` for descriptions)

        Returns
        -------
        rocket : rocketpy.Rocket
            Fully configured rocket instance ready for stability queries.
        """
        # Build thrust profile as a flat ramp with clean shut-down
        thrust_source = [
            [0, thrust_level],
            [burn_time, thrust_level],
            [burn_time + 0.01, 0],
        ]

        # Grains centre-of-mass in motor coordinates: midpoint of grain
        # stack relative to nozzle (nozzle_to_combustion_chamber orientation).
        grains_com = motor_dry_cg  # co-located with dry CG as approximation

        motor = SolidMotor(
            thrust_source=thrust_source,
            dry_mass=motor_dry_mass,
            dry_inertia=(motor_dry_I_i, motor_dry_I_i, motor_dry_I_z),
            center_of_dry_mass_position=motor_dry_cg,
            nozzle_radius=nozzle_radius,
            nozzle_position=nozzle_position,
            grain_number=grain_number,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_r,
            grain_initial_inner_radius=grain_inner_r,
            grain_initial_height=grain_height,
            grain_separation=0.0,
            grains_center_of_mass_position=grains_com,
            coordinate_system_orientation='nozzle_to_combustion_chamber',
            burn_time=burn_time,
        )

        rocket = Rocket(
            radius=body_radius,
            mass=body_mass,
            inertia=(body_I_i, body_I_i, body_I_z),
            power_off_drag=0.5,
            power_on_drag=0.5,
            center_of_mass_without_motor=cg_without_motor,
            coordinate_system_orientation='nose_to_tail',
        )

        # Motor at the aft end of the body
        motor_mount_position = -(body_length - nose_length)
        rocket.add_motor(motor, position=motor_mount_position)

        # Aero surfaces
        rocket.add_nose(
            length=nose_length,
            kind=nose_kind,
            position=0.0,
            base_radius=nose_base_radius,
            name='Nose',
        )

        rocket.add_trapezoidal_fins(
            n=n_fins,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=fin_span,
            position=fin_position,
            cant_angle=0.0,
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

        return rocket
