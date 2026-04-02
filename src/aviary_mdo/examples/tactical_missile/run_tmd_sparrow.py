"""
Example: Sparrow MRAAM Tactical Missile Design using Aviary + TMD Subsystem.

This script demonstrates using the MissileBuilder to perform conceptual
missile sizing following Fleeman's TMD methodology. It runs the baseline
Sparrow MRAAM case and performs a diameter trade study.

Usage:
    conda activate OpenMDAO
    python run_tmd_sparrow.py

Reference: Fleeman, E., "Tactical Missile Design," AIAA Education Series, 2001.
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile.missile_builder import MissileBuilder


def run_baseline():
    """Run the Sparrow MRAAM baseline case."""

    print("=" * 65)
    print("  TACTICAL MISSILE DESIGN — Sparrow MRAAM Baseline")
    print("  Built on NASA Aviary + TMD Subsystem")
    print("=" * 65)

    builder = MissileBuilder(name='missile', sustain_engine='rocket')
    pre_mission = builder.build_pre_mission(aviary_inputs=None)

    prob = om.Problem()
    prob.model.add_subsystem('missile', pre_mission)
    prob.setup()

    # === Sparrow MRAAM Baseline Inputs ===
    # Body geometry
    prob.set_val('missile.launch_mach', 0.8)
    prob.set_val('missile.launch_alt', 20000.0, units='ft')
    prob.set_val('missile.missile_length', 143.9, units='inch')
    prob.set_val('missile.missile_diameter', 8.0, units='inch')
    prob.set_val('missile.nose_length', 19.2, units='inch')
    prob.set_val('missile.nose_bluntness', 0.05)
    prob.set_val('missile.ref_area', 50.27, units='inch**2')
    prob.set_val('missile.cg_station', 76.2, units='inch')

    # Wing geometry
    prob.set_val('missile.num_wings', 2.0)
    prob.set_val('missile.wing_area', 400.0, units='inch**2')
    prob.set_val('missile.wing_AR', 2.82)
    prob.set_val('missile.wing_taper', 0.175)
    prob.set_val('missile.wing_le_station', 60.8, units='inch')

    # Tail geometry
    prob.set_val('missile.num_tails', 2.0)
    prob.set_val('missile.tail_area', 87.0, units='inch**2')
    prob.set_val('missile.tail_AR', 2.59)
    prob.set_val('missile.tail_taper', 0.0)
    prob.set_val('missile.tail_le_station', 125.4, units='inch')

    # Propulsion — boost motor
    prob.set_val('missile.boost_Pc', 1769.0, units='lbf/inch**2')
    prob.set_val('missile.boost_expansion_ratio', 6.0)
    prob.set_val('missile.boost_fuel_type', 4.0)  # High Smoke Composite
    prob.set_val('missile.W_boost_prop', 84.8, units='lbm')
    prob.set_val('missile.boost_burn_time', 3.26, units='s')
    prob.set_val('missile.Pa', 6.76, units='lbf/inch**2')

    # Propulsion — sustain motor
    prob.set_val('missile.sustain_Pc', 301.0, units='lbf/inch**2')
    prob.set_val('missile.sustain_expansion_ratio', 6.2)
    prob.set_val('missile.sustain_fuel_type', 4.0)
    prob.set_val('missile.W_sustain_prop', 48.2, units='lbm')
    prob.set_val('missile.sustain_burn_time', 10.86, units='s')

    # Trajectory
    prob.set_val('missile.trajectory.boost.W_launch', 500.0, units='lbm')
    prob.set_val('missile.trajectory.sustain.W_ejectables', 0.0, units='lbm')
    prob.set_val('missile.trajectory.sustain.CD0', 0.42)
    prob.set_val('missile.trajectory.coast.mach_terminal', 1.5)
    prob.set_val('missile.trajectory.coast.CD0', 0.46)

    prob.run_model()

    # === Extract Results ===
    boost_Isp = prob.get_val('missile.boost_motor.Isp', units='s')[0]
    boost_thrust = prob.get_val('missile.boost_motor.thrust', units='lbf')[0]
    sustain_Isp = prob.get_val('missile.sustain_motor.Isp', units='s')[0]

    R_boost = prob.get_val('missile.trajectory.boost.range_boost', units='nmi')[0]
    R_sustain = prob.get_val('missile.trajectory.sustain.range_sustain', units='nmi')[0]
    R_coast = prob.get_val('missile.trajectory.coast.range_coast', units='nmi')[0]
    R_total = prob.get_val('missile.range_total', units='nmi')[0]
    t_total = prob.get_val('missile.time_total', units='s')[0]

    M_boost_end = prob.get_val('missile.trajectory.boost.mach_boost_end')[0]
    T_skin = prob.get_val('missile.skin_temp.T_recovery')[0]

    print("\n  PROPULSION")
    print(f"    Boost:   Isp = {boost_Isp:.1f} s, Thrust = {boost_thrust:.0f} lbf")
    print(f"    Sustain: Isp = {sustain_Isp:.1f} s")

    print("\n  TRAJECTORY (co-altitude at 20,000 ft)")
    print(f"    {'Phase':<12} {'Range (nmi)':>12} {'Mach':>8}")
    print(f"    {'─'*12} {'─'*12} {'─'*8}")
    print(f"    {'Launch':<12} {'0.00':>12} {'0.80':>8}")
    print(f"    {'Boost':<12} {R_boost:>12.2f} {M_boost_end:>8.2f}")
    print(f"    {'Sustain':<12} {R_sustain:>12.2f}")
    print(f"    {'Coast':<12} {R_coast:>12.2f} {'1.50':>8}")
    print(f"    {'─'*12} {'─'*12} {'─'*8}")
    print(f"    {'TOTAL':<12} {R_total:>12.2f} {'':>8}")
    print(f"    Flight time: {t_total:.1f} s")

    print(f"\n  THERMAL")
    print(f"    Peak skin temperature: {T_skin:.0f} R ({T_skin*5/9:.0f} K)")

    print("\n" + "=" * 65)

    return prob


def run_diameter_trade():
    """Diameter trade study — key finding from Fleeman."""

    print("\n" + "=" * 65)
    print("  DIAMETER TRADE STUDY")
    print("  (Fleeman: smaller diameter -> longer range)")
    print("=" * 65)

    diameters = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
    results = []

    for d in diameters:
        builder = MissileBuilder(sustain_engine='rocket')
        pre_mission = builder.build_pre_mission(aviary_inputs=None)

        prob = om.Problem()
        prob.model.add_subsystem('m', pre_mission)
        prob.setup()

        # Keep volume constant: adjust length as diameter changes
        # V = pi/4 * d^2 * L -> L = V / (pi/4 * d^2)
        V_baseline = np.pi / 4.0 * 8.0**2 * 143.9  # baseline volume
        L = V_baseline / (np.pi / 4.0 * d**2)

        prob.set_val('m.missile_diameter', d, units='inch')
        prob.set_val('m.missile_length', L, units='inch')
        prob.set_val('m.ref_area', np.pi / 4.0 * d**2, units='inch**2')

        prob.set_val('m.trajectory.boost.W_launch', 500.0, units='lbm')
        prob.set_val('m.trajectory.sustain.W_ejectables', 0.0, units='lbm')
        prob.set_val('m.trajectory.sustain.CD0', 0.42)
        prob.set_val('m.trajectory.coast.mach_terminal', 1.5)
        prob.set_val('m.trajectory.coast.CD0', 0.46)

        prob.run_model()

        R = prob.get_val('m.range_total', units='nmi')[0]
        t = prob.get_val('m.time_total', units='s')[0]
        results.append((d, L, R, t))

    print(f"\n  {'Dia (in)':>10} {'Length (in)':>12} {'Range (nmi)':>12} {'Time (s)':>10}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10}")
    for d, L, R, t in results:
        marker = " <-- baseline" if abs(d - 8.0) < 0.01 else ""
        print(f"  {d:>10.1f} {L:>12.1f} {R:>12.2f} {t:>10.1f}{marker}")

    print("\n" + "=" * 65)


if __name__ == '__main__':
    run_baseline()
    run_diameter_trade()
