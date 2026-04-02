"""
Example: Supersonic UAV Aerothermodynamic Analysis using Aviary.

This script demonstrates using the AerothermoBuilder to perform
aerothermal analysis and TPS sizing for a supersonic/hypersonic UAV.
It evaluates stagnation heating, surface heat flux distribution,
and thermal protection system mass across a range of Mach numbers.

Usage:
    conda activate OpenMDAO
    python run_supersonic_uav.py

References:
    Fay & Riddell (1958), J. Aeronautical Sciences, Vol 25, No 2.
    Sutton & Graves (1971), NASA TR R-376.
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.aerothermodynamics.aerothermo_builder import AerothermoBuilder


def run_mach_sweep():
    """Evaluate aerothermal environment across Mach 2-8."""

    print("=" * 70)
    print("  SUPERSONIC UAV — Aerothermodynamic Analysis")
    print("  Built on NASA Aviary + Aerothermodynamics Subsystem")
    print("=" * 70)

    print("\n  Vehicle Parameters:")
    print("    Nose radius:  0.10 m (sharp nose, high-speed UAV)")
    print("    Body length:  4.0 m")
    print("    Wetted area:  8.0 m^2")
    print("    Wall temp:    400 K (actively cooled or radiative)")
    print("    TPS type:     Ceramic (shuttle-tile class)")

    machs = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    altitudes_m = [15000, 20000, 25000, 30000, 35000, 45000]  # meters

    results = []

    for M, alt_m in zip(machs, altitudes_m):
        builder = AerothermoBuilder()
        pre_mission = builder.build_pre_mission(aviary_inputs=None)

        prob = om.Problem()
        prob.model.add_subsystem('at', pre_mission)
        prob.setup()

        prob.set_val('at.mach', M)
        prob.set_val('at.altitude', alt_m, units='m')
        prob.set_val('at.nose_radius', 0.10)
        prob.set_val('at.body_length', 4.0)
        prob.set_val('at.wetted_area', 8.0)
        prob.set_val('at.wall_temperature', 400.0)

        prob.run_model()

        q_stag = prob.get_val('at.q_stag')[0]
        T_eq = prob.get_val('at.T_wall_eq')[0]
        tps_mass = prob.get_val('at.mass_tps')[0]
        tps_thick = prob.get_val('at.thickness')[0]

        results.append({
            'mach': M,
            'alt_km': alt_m / 1000.0,
            'q_stag_kW': q_stag / 1000.0,
            'T_eq_K': T_eq,
            'tps_mass_kg': tps_mass,
            'tps_thick_mm': tps_thick * 1000.0,
        })

    # Print results table
    print(f"\n  {'Mach':>6} {'Alt (km)':>9} {'q_stag (kW/m2)':>15} "
          f"{'T_rad_eq (K)':>13} {'TPS thick (mm)':>15} {'TPS mass (kg)':>14}")
    print(f"  {'─'*6} {'─'*9} {'─'*15} {'─'*13} {'─'*15} {'─'*14}")

    for r in results:
        print(f"  {r['mach']:>6.1f} {r['alt_km']:>9.0f} {r['q_stag_kW']:>15.1f} "
              f"{r['T_eq_K']:>13.0f} {r['tps_thick_mm']:>15.1f} {r['tps_mass_kg']:>14.1f}")

    print(f"\n  KEY OBSERVATIONS:")
    print(f"  - Heat flux scales roughly as V^3 (Mach^3 at similar altitude)")
    print(f"  - Higher altitude partially offsets heating (lower density)")
    print(f"  - TPS mass is a critical design driver above Mach 5")
    print(f"  - Radiative equilibrium temp indicates material limits")
    print(f"\n  MATERIAL LIMITS (approximate):")
    print(f"    Aluminum:    ~450 K   (Mach < 2.5)")
    print(f"    Titanium:    ~800 K   (Mach < 4)")
    print(f"    Nickel alloy:~1300 K  (Mach < 6)")
    print(f"    Ceramic TPS: ~1900 K  (Mach < 8)")
    print(f"    Ablative:    ~3000 K+ (re-entry)")

    print("\n" + "=" * 70)

    return results


def run_nose_radius_trade():
    """Trade study: nose radius effect on heating."""

    print("\n" + "=" * 70)
    print("  NOSE RADIUS TRADE STUDY at Mach 5, 30 km")
    print("  (Larger radius -> lower heat flux but more drag)")
    print("=" * 70)

    radii = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0]
    results = []

    for R in radii:
        builder = AerothermoBuilder()
        pre_mission = builder.build_pre_mission(aviary_inputs=None)

        prob = om.Problem()
        prob.model.add_subsystem('at', pre_mission)
        prob.setup()

        prob.set_val('at.mach', 5.0)
        prob.set_val('at.altitude', 30000.0, units='m')
        prob.set_val('at.nose_radius', R)
        prob.set_val('at.body_length', 4.0)
        prob.set_val('at.wetted_area', 8.0)
        prob.set_val('at.wall_temperature', 400.0)

        prob.run_model()

        q_stag = prob.get_val('at.q_stag')[0]
        tps_mass = prob.get_val('at.mass_tps')[0]
        results.append((R, q_stag / 1000.0, tps_mass))

    print(f"\n  {'R_nose (m)':>12} {'q_stag (kW/m2)':>16} {'TPS mass (kg)':>14}")
    print(f"  {'─'*12} {'─'*16} {'─'*14}")
    for R, q, m in results:
        print(f"  {R:>12.3f} {q:>16.1f} {m:>14.1f}")

    print(f"\n  Note: q_stag ~ 1/sqrt(R_nose)")
    print(f"  This is a fundamental trade — blunter nose reduces heating")
    print(f"  but increases wave drag. Optimization finds the sweet spot.")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    run_mach_sweep()
    run_nose_radius_trade()
