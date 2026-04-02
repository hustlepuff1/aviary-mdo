"""
RocketPy SolidMotor wrapper for the 6-DOF missile subsystem.

Constructs a ``rocketpy.SolidMotor`` from grain geometry, motor inertia, and a
flat thrust level, then extracts aggregate motor performance metrics (total
impulse, max/average thrust, average Isp, propellant mass).

Because RocketPy objects are opaque to complex-step differentiation, partials
are declared with finite differences.
"""

import numpy as np
import openmdao.api as om
from rocketpy import SolidMotor

from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)

# Gravitational acceleration for Isp conversion
_G0 = 9.80665  # m/s^2


class RocketPyMotor(om.ExplicitComponent):
    """
    OpenMDAO wrapper around RocketPy's ``SolidMotor`` grain-regression model.

    A flat (constant) thrust curve is synthesised from the scalar thrust-level
    input.  RocketPy then computes grain regression, propellant mass, and
    exhaust velocity internally.

    Inputs
    ------
    missile_6dof:motor:thrust_source       : float  (N)
        Constant thrust level used to build a flat thrust curve.
    missile_6dof:motor:burn_time           : float  (s)
    missile_6dof:motor:dry_mass            : float  (kg)
    missile_6dof:motor:dry_inertia_i       : float  (kg*m**2)
    missile_6dof:motor:dry_inertia_z       : float  (kg*m**2)
    missile_6dof:motor:center_of_dry_mass  : float  (m)
    missile_6dof:motor:nozzle_radius       : float  (m)
    missile_6dof:motor:nozzle_position     : float  (m)
    missile_6dof:grain:propellant_density  : float  (kg/m**3)
    missile_6dof:grain:number              : float  (unitless)
    missile_6dof:grain:outer_radius        : float  (m)
    missile_6dof:grain:inner_radius        : float  (m)
    missile_6dof:grain:height              : float  (m)

    Outputs
    -------
    dynamic:missile_6dof:total_impulse   : float  (N*s)
    dynamic:missile_6dof:max_thrust      : float  (N)
    dynamic:missile_6dof:average_thrust  : float  (N)
    dynamic:missile_6dof:average_isp     : float  (s)
    dynamic:missile_6dof:propellant_mass : float  (kg)
    """

    def setup(self):
        # --- Motor inputs ---
        self.add_input(
            Missile6DOF.Motor.THRUST_SOURCE,
            val=5000.0,
            units='N',
            desc='Constant thrust level for flat thrust curve',
        )
        self.add_input(
            Missile6DOF.Motor.BURN_TIME,
            val=3.26,
            units='s',
            desc='Total motor burn time',
        )
        self.add_input(
            Missile6DOF.Motor.DRY_MASS,
            val=10.0,
            units='kg',
            desc='Motor dry mass (casing + nozzle)',
        )
        self.add_input(
            Missile6DOF.Motor.DRY_INERTIA_I,
            val=5.0,
            units='kg*m**2',
            desc='Motor dry lateral moment of inertia',
        )
        self.add_input(
            Missile6DOF.Motor.DRY_INERTIA_Z,
            val=0.1,
            units='kg*m**2',
            desc='Motor dry axial moment of inertia',
        )
        self.add_input(
            Missile6DOF.Motor.CENTER_OF_DRY_MASS,
            val=0.5,
            units='m',
            desc='Motor dry CG position from motor origin',
        )
        self.add_input(
            Missile6DOF.Motor.NOZZLE_RADIUS,
            val=0.0508,
            units='m',
            desc='Nozzle exit radius',
        )
        self.add_input(
            Missile6DOF.Motor.NOZZLE_POSITION,
            val=-1.0,
            units='m',
            desc='Nozzle position from rocket origin',
        )

        # --- Grain inputs ---
        self.add_input(
            Missile6DOF.SolidGrain.PROPELLANT_DENSITY,
            val=1750.0,
            units='kg/m**3',
            desc='Solid propellant density',
        )
        self.add_input(
            Missile6DOF.SolidGrain.NUMBER,
            val=4.0,
            units=None,
            desc='Number of grain segments',
        )
        self.add_input(
            Missile6DOF.SolidGrain.OUTER_RADIUS,
            val=0.0889,
            units='m',
            desc='Grain outer radius',
        )
        self.add_input(
            Missile6DOF.SolidGrain.INNER_RADIUS,
            val=0.0254,
            units='m',
            desc='Grain initial inner radius (port)',
        )
        self.add_input(
            Missile6DOF.SolidGrain.HEIGHT,
            val=0.254,
            units='m',
            desc='Height of each grain segment',
        )

        # --- Motor performance outputs ---
        self.add_output(
            DynamicMissile6DOF.TOTAL_IMPULSE,
            val=0.0,
            units='N*s',
            desc='Motor total impulse',
        )
        self.add_output(
            DynamicMissile6DOF.MAX_THRUST,
            val=0.0,
            units='N',
            desc='Motor maximum thrust',
        )
        self.add_output(
            DynamicMissile6DOF.AVERAGE_THRUST,
            val=0.0,
            units='N',
            desc='Motor average thrust',
        )
        self.add_output(
            DynamicMissile6DOF.AVERAGE_ISP,
            val=0.0,
            units='s',
            desc='Motor average specific impulse',
        )
        self.add_output(
            DynamicMissile6DOF.PROPELLANT_MASS,
            val=0.0,
            units='kg',
            desc='Total propellant mass',
        )

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Extract scalar values from input arrays
        thrust_level = float(np.real(inputs[Missile6DOF.Motor.THRUST_SOURCE]))
        burn_time = float(np.real(inputs[Missile6DOF.Motor.BURN_TIME]))
        dry_mass = float(np.real(inputs[Missile6DOF.Motor.DRY_MASS]))
        dry_I_i = float(np.real(inputs[Missile6DOF.Motor.DRY_INERTIA_I]))
        dry_I_z = float(np.real(inputs[Missile6DOF.Motor.DRY_INERTIA_Z]))
        cg_dry = float(np.real(inputs[Missile6DOF.Motor.CENTER_OF_DRY_MASS]))
        nozzle_r = float(np.real(inputs[Missile6DOF.Motor.NOZZLE_RADIUS]))
        nozzle_pos = float(np.real(inputs[Missile6DOF.Motor.NOZZLE_POSITION]))

        prop_density = float(np.real(inputs[Missile6DOF.SolidGrain.PROPELLANT_DENSITY]))
        n_grains = int(np.real(inputs[Missile6DOF.SolidGrain.NUMBER]))
        outer_r = float(np.real(inputs[Missile6DOF.SolidGrain.OUTER_RADIUS]))
        inner_r = float(np.real(inputs[Missile6DOF.SolidGrain.INNER_RADIUS]))
        grain_h = float(np.real(inputs[Missile6DOF.SolidGrain.HEIGHT]))

        # Guard against invalid inputs during solver iterations
        burn_time = max(burn_time, 0.01)
        thrust_level = max(thrust_level, 0.1)
        n_grains = max(n_grains, 1)
        outer_r = max(outer_r, 1.0e-3)
        inner_r = max(inner_r, 1.0e-4)
        grain_h = max(grain_h, 1.0e-3)

        # Ensure inner < outer radius
        if inner_r >= outer_r:
            inner_r = outer_r * 0.5

        try:
            motor = SolidMotor(
                thrust_source=[[0.0, thrust_level], [burn_time, thrust_level]],
                burn_time=burn_time,
                dry_mass=dry_mass,
                dry_inertia=(dry_I_i, dry_I_i, dry_I_z),
                center_of_dry_mass_position=cg_dry,
                nozzle_radius=nozzle_r,
                nozzle_position=nozzle_pos,
                grain_number=n_grains,
                grain_density=prop_density,
                grain_outer_radius=outer_r,
                grain_initial_inner_radius=inner_r,
                grain_initial_height=grain_h,
                grain_separation=0.005,
                grains_center_of_mass_position=0.0,
                coordinate_system_orientation='nozzle_to_combustion_chamber',
            )

            outputs[DynamicMissile6DOF.TOTAL_IMPULSE] = float(motor.total_impulse)
            outputs[DynamicMissile6DOF.MAX_THRUST] = float(motor.max_thrust)
            outputs[DynamicMissile6DOF.AVERAGE_THRUST] = float(motor.average_thrust)
            outputs[DynamicMissile6DOF.PROPELLANT_MASS] = float(
                motor.propellant_initial_mass
            )

            # Average Isp from average exhaust velocity
            avg_ve = float(motor.exhaust_velocity.average(0.0, burn_time))
            outputs[DynamicMissile6DOF.AVERAGE_ISP] = avg_ve / _G0

        except Exception:
            # If RocketPy cannot build a valid motor (e.g. bad geometry during
            # early solver iterations), return safe fallback values so the
            # optimizer can recover.
            outputs[DynamicMissile6DOF.TOTAL_IMPULSE] = thrust_level * burn_time
            outputs[DynamicMissile6DOF.MAX_THRUST] = thrust_level
            outputs[DynamicMissile6DOF.AVERAGE_THRUST] = thrust_level
            outputs[DynamicMissile6DOF.AVERAGE_ISP] = 200.0
            outputs[DynamicMissile6DOF.PROPELLANT_MASS] = 0.0
