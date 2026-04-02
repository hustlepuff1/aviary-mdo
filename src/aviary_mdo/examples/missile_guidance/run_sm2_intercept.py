"""
SM-2 Class Interceptor Engagement Analysis
===========================================

Demonstrates the missile guidance subsystem with a surface-to-air
engagement scenario representative of an SM-2 Standard Missile
intercepting a maneuvering aircraft target.

Three scenarios are simulated:
1. Head-on, non-maneuvering target (baseline)
2. Crossing engagement, maneuvering target (challenging)
3. Navigation ratio sweep (trade study)

Usage:
    conda run -n OpenMDAO python run_sm2_intercept.py
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder
from aviary_mdo.missile_guidance.guidance_variables import Guidance, DynamicGuidance


def run_scenario(name, missile_cfg, target_cfg, guidance_cfg):
    """Run a single engagement scenario and print results."""
    builder = GuidanceBuilder(name='guidance')
    prob = om.Problem()
    prob.model.add_subsystem('guidance', builder.build_pre_mission(None))
    prob.setup()

    # Missile
    for key, (val, units) in missile_cfg.items():
        if units:
            prob.set_val(f'guidance.{key}', val, units=units)
        else:
            prob.set_val(f'guidance.{key}', val)

    # Target
    for key, (val, units) in target_cfg.items():
        if units:
            prob.set_val(f'guidance.{key}', val, units=units)
        else:
            prob.set_val(f'guidance.{key}', val)

    # Guidance
    for key, (val, units) in guidance_cfg.items():
        if units:
            prob.set_val(f'guidance.{key}', val, units=units)
        else:
            prob.set_val(f'guidance.{key}', val)

    prob.run_model()

    miss = prob.get_val(f'guidance.{DynamicGuidance.MISS_DISTANCE}', units='m')[0]
    pk = prob.get_val(f'guidance.{DynamicGuidance.SINGLE_SHOT_PK}')[0]
    t_int = prob.get_val(f'guidance.{DynamicGuidance.TIME_TO_INTERCEPT}', units='s')[0]
    max_g = prob.get_val(f'guidance.{DynamicGuidance.REQUIRED_LOAD_FACTOR}')[0]
    max_cmd = prob.get_val(f'guidance.{DynamicGuidance.MAX_COMMANDED_ACCEL}', units='m/s**2')[0]
    divert = prob.get_val(f'guidance.{DynamicGuidance.DIVERT_REQUIREMENT}', units='m/s')[0]
    intercept = prob.get_val(f'guidance.{DynamicGuidance.INTERCEPT_ACHIEVED}')[0]

    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"  Miss Distance:     {miss:.2f} m")
    print(f"  Single-Shot Pk:    {pk:.3f}")
    print(f"  Intercept Time:    {t_int:.1f} s")
    print(f"  Peak Load Factor:  {max_g:.1f} g")
    print(f"  Max Cmd Accel:     {max_cmd:.0f} m/s^2")
    print(f"  Divert (delta-V):  {divert:.0f} m/s")
    print(f"  Intercept:         {'YES' if intercept else 'NO'}")

    return miss, pk


def main():
    print("SM-2 CLASS INTERCEPTOR ENGAGEMENT ANALYSIS")
    print("=" * 60)

    # SM-2 missile parameters (approximate)
    sm2_missile = {
        Guidance.Missile.POSITION_X: (0.0, 'm'),
        Guidance.Missile.POSITION_Y: (0.0, 'm'),
        Guidance.Missile.VELOCITY: (1000.0, 'm/s'),       # Mach ~3 at altitude
        Guidance.Missile.MAX_ACCELERATION: (300.0, 'm/s**2'),  # ~30g
        Guidance.Missile.DRAG_COEFFICIENT: (0.4, None),
        Guidance.Missile.REFERENCE_AREA: (0.0324, 'm**2'),  # 8in diameter
        Guidance.Missile.MASS: (700.0, 'kg'),
        Guidance.Missile.THRUST: (0.0, 'N'),  # coasting in terminal
        Guidance.Missile.BURN_TIME: (0.0, 's'),
    }

    # Scenario 1: Head-on, non-maneuvering
    print("\n--- Scenario 1: Head-on, Non-Maneuvering Target ---")
    run_scenario(
        "Head-on / Non-Maneuv",
        {**sm2_missile, Guidance.Missile.HEADING: (0.0, 'deg')},
        {
            Guidance.Target.POSITION_X: (15000.0, 'm'),
            Guidance.Target.POSITION_Y: (0.0, 'm'),
            Guidance.Target.VELOCITY: (250.0, 'm/s'),
            Guidance.Target.HEADING: (180.0, 'deg'),
            Guidance.Target.ACCELERATION_MAX: (0.0, 'm/s**2'),
            Guidance.Target.MANEUVER_TIME: (99.0, 's'),
        },
        {
            Guidance.Navigation.RATIO: (4.0, None),
            Guidance.Autopilot.TIME_CONSTANT: (0.2, 's'),
            Guidance.Environment.DENSITY: (0.66, 'kg/m**3'),
        },
    )

    # Scenario 2: Crossing, maneuvering target (5g)
    print("\n--- Scenario 2: Crossing, 5g Maneuvering Target ---")
    run_scenario(
        "Crossing / 5g Maneuver",
        {**sm2_missile, Guidance.Missile.HEADING: (45.0, 'deg')},
        {
            Guidance.Target.POSITION_X: (15000.0, 'm'),
            Guidance.Target.POSITION_Y: (15000.0, 'm'),
            Guidance.Target.VELOCITY: (250.0, 'm/s'),
            Guidance.Target.HEADING: (180.0, 'deg'),
            Guidance.Target.ACCELERATION_MAX: (50.0, 'm/s**2'),  # ~5g
            Guidance.Target.MANEUVER_TIME: (5.0, 's'),
        },
        {
            Guidance.Navigation.RATIO: (4.0, None),
            Guidance.Autopilot.TIME_CONSTANT: (0.2, 's'),
            Guidance.Environment.DENSITY: (0.66, 'kg/m**3'),
        },
    )

    # Scenario 3: Navigation ratio sweep
    print("\n\n--- Navigation Ratio Sweep (Crossing + 3g Maneuver) ---")
    print(f"{'N':>6} {'Miss (m)':>10} {'Pk':>8} {'Peak g':>8} {'Time (s)':>10}")
    print("-" * 45)

    builder = GuidanceBuilder(name='guidance')
    prob = om.Problem()
    prob.model.add_subsystem('guidance', builder.build_pre_mission(None))
    prob.setup()

    # Set baseline
    prob.set_val(f'guidance.{Guidance.Missile.POSITION_X}', 0.0, units='m')
    prob.set_val(f'guidance.{Guidance.Missile.POSITION_Y}', 0.0, units='m')
    prob.set_val(f'guidance.{Guidance.Missile.VELOCITY}', 1000.0, units='m/s')
    prob.set_val(f'guidance.{Guidance.Missile.HEADING}', 45.0, units='deg')
    prob.set_val(f'guidance.{Guidance.Missile.MAX_ACCELERATION}', 300.0, units='m/s**2')
    prob.set_val(f'guidance.{Guidance.Missile.DRAG_COEFFICIENT}', 0.4)
    prob.set_val(f'guidance.{Guidance.Missile.REFERENCE_AREA}', 0.0324, units='m**2')
    prob.set_val(f'guidance.{Guidance.Missile.MASS}', 700.0, units='kg')
    prob.set_val(f'guidance.{Guidance.Target.POSITION_X}', 15000.0, units='m')
    prob.set_val(f'guidance.{Guidance.Target.POSITION_Y}', 15000.0, units='m')
    prob.set_val(f'guidance.{Guidance.Target.VELOCITY}', 250.0, units='m/s')
    prob.set_val(f'guidance.{Guidance.Target.HEADING}', 180.0, units='deg')
    prob.set_val(f'guidance.{Guidance.Target.ACCELERATION_MAX}', 30.0, units='m/s**2')
    prob.set_val(f'guidance.{Guidance.Target.MANEUVER_TIME}', 5.0, units='s')
    prob.set_val(f'guidance.{Guidance.Autopilot.TIME_CONSTANT}', 0.2, units='s')
    prob.set_val(f'guidance.{Guidance.Environment.DENSITY}', 0.66, units='kg/m**3')

    for N in np.arange(2.5, 6.5, 0.5):
        prob.set_val(f'guidance.{Guidance.Navigation.RATIO}', N)
        prob.run_model()

        miss = prob.get_val(f'guidance.{DynamicGuidance.MISS_DISTANCE}', units='m')[0]
        pk = prob.get_val(f'guidance.{DynamicGuidance.SINGLE_SHOT_PK}')[0]
        g = prob.get_val(f'guidance.{DynamicGuidance.REQUIRED_LOAD_FACTOR}')[0]
        t = prob.get_val(f'guidance.{DynamicGuidance.TIME_TO_INTERCEPT}', units='s')[0]
        print(f"{N:>6.1f} {miss:>10.2f} {pk:>8.3f} {g:>8.1f} {t:>10.1f}")


if __name__ == '__main__':
    main()
