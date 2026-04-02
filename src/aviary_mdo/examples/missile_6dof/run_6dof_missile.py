"""
RocketPy 6-DOF Missile Analysis Example
========================================

Demonstrates the Aviary missile_6dof subsystem that wraps RocketPy
for high-fidelity 6-DOF trajectory simulation with Barrowman aerodynamics
and solid motor grain regression.

This complements the TMD (Tactical Missile Design) analytical module
with physics-based simulation for design validation.

Usage:
    conda run -n OpenMDAO python run_6dof_missile.py
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_6dof.missile_6dof_builder import Missile6DOFBuilder
from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)


def run_6dof_analysis():
    """Run a 6-DOF missile flight analysis."""

    builder = Missile6DOFBuilder(name='missile_6dof')

    prob = om.Problem()
    prob.model.add_subsystem('missile_6dof', builder.build_pre_mission(None))
    prob.setup()

    # --- Missile Geometry (Sparrow-class) ---
    prefix = 'missile_6dof.'
    prob.set_val(prefix + Missile6DOF.Body.LENGTH, 3.658, units='m')
    prob.set_val(prefix + Missile6DOF.Body.RADIUS, 0.1016, units='m')
    prob.set_val(prefix + Missile6DOF.Body.MASS, 100.0, units='kg')
    prob.set_val(prefix + Missile6DOF.Body.INERTIA_I, 50.0, units='kg*m**2')
    prob.set_val(prefix + Missile6DOF.Body.INERTIA_Z, 0.5, units='kg*m**2')
    prob.set_val(prefix + Missile6DOF.Body.CG_WITHOUT_MOTOR, 1.83, units='m')

    # Nose cone
    prob.set_val(prefix + Missile6DOF.NoseCone.LENGTH, 0.488, units='m')
    prob.set_val(prefix + Missile6DOF.NoseCone.BASE_RADIUS, 0.1016, units='m')

    # Fins (4 trapezoidal)
    prob.set_val(prefix + Missile6DOF.Fins.ROOT_CHORD, 0.254, units='m')
    prob.set_val(prefix + Missile6DOF.Fins.TIP_CHORD, 0.076, units='m')
    prob.set_val(prefix + Missile6DOF.Fins.SPAN, 0.127, units='m')
    prob.set_val(prefix + Missile6DOF.Fins.POSITION, 3.2, units='m')
    prob.set_val(prefix + Missile6DOF.Fins.SWEEP_LENGTH, 0.127, units='m')

    # Motor
    prob.set_val(prefix + Missile6DOF.Motor.THRUST_SOURCE, 30000.0, units='N')
    prob.set_val(prefix + Missile6DOF.Motor.BURN_TIME, 3.26, units='s')
    prob.set_val(prefix + Missile6DOF.Motor.DRY_MASS, 15.0, units='kg')
    prob.set_val(prefix + Missile6DOF.Motor.NOZZLE_RADIUS, 0.0508, units='m')

    # Grain
    prob.set_val(
        prefix + Missile6DOF.SolidGrain.OUTER_RADIUS, 0.0889, units='m'
    )
    prob.set_val(
        prefix + Missile6DOF.SolidGrain.INNER_RADIUS, 0.0254, units='m'
    )
    prob.set_val(prefix + Missile6DOF.SolidGrain.HEIGHT, 0.254, units='m')

    # Launch conditions
    prob.set_val(prefix + Missile6DOF.Launch.RAIL_LENGTH, 5.0, units='m')
    prob.set_val(prefix + Missile6DOF.Launch.INCLINATION, 45.0, units='deg')
    prob.set_val(prefix + Missile6DOF.Launch.HEADING, 0.0, units='deg')
    prob.set_val(prefix + Missile6DOF.Environment.ELEVATION, 0.0, units='m')

    # Run
    prob.run_model()

    # Helper to get a scalar value
    def gv(var, **kw):
        return prob.get_val(prefix + var, **kw)[0]

    # Results
    print("=" * 60)
    print("6-DOF MISSILE FLIGHT ANALYSIS (RocketPy)")
    print("=" * 60)
    print()
    print("TRAJECTORY RESULTS:")
    apogee = gv(DynamicMissile6DOF.APOGEE, units='m')
    max_speed = gv(DynamicMissile6DOF.MAX_SPEED, units='m/s')
    max_mach = gv(DynamicMissile6DOF.MAX_MACH)
    max_accel = gv(DynamicMissile6DOF.MAX_ACCELERATION, units='m/s**2')
    max_q = gv(DynamicMissile6DOF.MAX_DYNAMIC_PRESSURE, units='Pa')
    flight_time = gv(DynamicMissile6DOF.FLIGHT_TIME, units='s')
    impact_vel = gv(DynamicMissile6DOF.IMPACT_VELOCITY, units='m/s')
    range_x = gv(DynamicMissile6DOF.RANGE_X, units='m')
    range_y = gv(DynamicMissile6DOF.RANGE_Y, units='m')
    range_total = gv(DynamicMissile6DOF.RANGE_TOTAL, units='m')

    print(f"  Apogee:               {apogee:,.1f} m")
    print(f"  Max Speed:            {max_speed:,.1f} m/s")
    print(f"  Max Mach:             {max_mach:.2f}")
    print(f"  Max Acceleration:     {max_accel:,.1f} m/s^2 ({max_accel / 9.81:.1f} g)")
    print(f"  Max Dynamic Pressure: {max_q:,.0f} Pa")
    print(f"  Flight Time:          {flight_time:.1f} s")
    print(f"  Impact Velocity:      {impact_vel:,.1f} m/s")
    print(f"  Downrange (X):        {range_x:,.1f} m")
    print(f"  Crossrange (Y):       {range_y:,.1f} m")
    print(f"  Total Range:          {range_total:,.1f} m")
    print()

    total_impulse = gv(DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s')
    max_thrust = gv(DynamicMissile6DOF.MAX_THRUST, units='N')
    avg_thrust = gv(DynamicMissile6DOF.AVERAGE_THRUST, units='N')
    avg_isp = gv(DynamicMissile6DOF.AVERAGE_ISP, units='s')
    prop_mass = gv(DynamicMissile6DOF.PROPELLANT_MASS, units='kg')

    print("MOTOR PERFORMANCE:")
    print(f"  Total Impulse:        {total_impulse:,.0f} N*s")
    print(f"  Max Thrust:           {max_thrust:,.0f} N")
    print(f"  Avg Thrust:           {avg_thrust:,.0f} N")
    print(f"  Avg Isp:              {avg_isp:.1f} s")
    print(f"  Propellant Mass:      {prop_mass:.2f} kg")
    print()

    sm_ign = gv(DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION)
    sm_bo = gv(DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT)
    cp_pos = gv(DynamicMissile6DOF.CP_POSITION, units='m')
    cg_pos = gv(DynamicMissile6DOF.CG_POSITION, units='m')

    print("STABILITY:")
    print(f"  Static Margin (ign):  {sm_ign:.2f} cal")
    print(f"  Static Margin (bo):   {sm_bo:.2f} cal")
    print(f"  CP Position:          {cp_pos:.3f} m from nose")
    print(f"  CG Position:          {cg_pos:.3f} m from nose")
    print()
    print("=" * 60)

    return prob


def run_inclination_sweep():
    """Sweep launch inclination to find max-range angle."""

    builder = Missile6DOFBuilder(name='missile_6dof')
    prob = om.Problem()
    prob.model.add_subsystem('missile_6dof', builder.build_pre_mission(None))
    prob.setup()

    prefix = 'missile_6dof.'

    # Set baseline missile config
    prob.set_val(prefix + Missile6DOF.Motor.THRUST_SOURCE, 30000.0, units='N')
    prob.set_val(prefix + Missile6DOF.Motor.BURN_TIME, 3.26, units='s')
    prob.set_val(prefix + Missile6DOF.Launch.RAIL_LENGTH, 5.0, units='m')

    angles = np.arange(15, 90, 5)
    results = []

    header = (
        f"{'Angle (deg)':>12} {'Range (m)':>12} {'Apogee (m)':>12} "
        f"{'Max Mach':>10} {'Flight Time (s)':>16}"
    )
    print("\nINCLINATION SWEEP:")
    print(header)
    print("-" * 65)

    for angle in angles:
        prob.set_val(
            prefix + Missile6DOF.Launch.INCLINATION, angle, units='deg'
        )
        prob.run_model()

        rng = prob.get_val(
            prefix + DynamicMissile6DOF.RANGE_TOTAL, units='m'
        )[0]
        apo = prob.get_val(
            prefix + DynamicMissile6DOF.APOGEE, units='m'
        )[0]
        mach = prob.get_val(prefix + DynamicMissile6DOF.MAX_MACH)[0]
        t_flight = prob.get_val(
            prefix + DynamicMissile6DOF.FLIGHT_TIME, units='s'
        )[0]

        results.append((angle, rng, apo, mach, t_flight))
        print(
            f"{angle:>12.0f} {rng:>12.1f} {apo:>12.1f} "
            f"{mach:>10.2f} {t_flight:>16.1f}"
        )

    # Find max range
    best = max(results, key=lambda x: x[1])
    print(f"\nMax range {best[1]:.0f} m at {best[0]:.0f} deg inclination")


if __name__ == '__main__':
    run_6dof_analysis()
    print()
    run_inclination_sweep()
