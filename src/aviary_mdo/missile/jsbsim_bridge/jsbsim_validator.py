"""
JSBSim 6DOF Validator for TMD missile designs.

Runs a JSBSim simulation using the missile definition generated from TMD
parameters, then compares trajectory results against TMD's simplified
(1DOF co-altitude) trajectory model.

This provides a fidelity check: how well does the TMD conceptual design
tool predict the trajectory compared to a full 6DOF simulation?

Usage:
    from aviary_mdo.missile.jsbsim_bridge.jsbsim_validator import (
        JSBSimValidator
    )

    validator = JSBSimValidator()
    validator.setup_from_tmd(
        length=143.9, diameter=8.0, nose_length=19.2,
        launch_weight=500.0, cg_station=76.2,
        thrust=7036.0, isp=271.0, burn_time=3.26,
        propellant_weight=84.8, launch_mach=0.8, launch_alt=20000.0,
    )
    results = validator.run_simulation(max_time=60.0)
    validator.print_comparison(tmd_range=10.71, tmd_time=30.9)
"""

import os
import tempfile
import numpy as np

try:
    import jsbsim
    JSBSIM_AVAILABLE = True
except ImportError:
    JSBSIM_AVAILABLE = False

from aviary_mdo.missile.jsbsim_bridge.tmd_to_jsbsim import (
    TMDToJSBSim, generate_reset_file,
)


class JSBSimValidator:
    """
    Validates TMD trajectory predictions against JSBSim 6DOF simulation.
    """

    def __init__(self):
        self.translator = TMDToJSBSim()
        self.launch_mach = 0.8
        self.launch_alt = 20000.0
        self.terminal_mach = 1.5
        self.results = None
        self._work_dir = None

    def setup_from_tmd(self, length=143.9, diameter=8.0, nose_length=19.2,
                       wing_area=400.0, launch_weight=500.0, empty_weight=367.0,
                       cg_station=76.2, Iy=94.0,
                       thrust=7036.0, isp=271.0, burn_time=3.26,
                       propellant_weight=84.8,
                       sustain_thrust=1119.0, sustain_isp=252.0,
                       sustain_burn_time=10.86, sustain_propellant=48.2,
                       launch_mach=0.8, launch_alt=20000.0, terminal_mach=1.5,
                       machs=None, cd0_values=None):
        """Configure the validator from TMD design parameters."""

        self.launch_mach = launch_mach
        self.launch_alt = launch_alt
        self.terminal_mach = terminal_mach

        self.translator.set_geometry(
            length=length, diameter=diameter, nose_length=nose_length,
            wing_area=wing_area)
        self.translator.set_mass(
            launch_weight=launch_weight, empty_weight=empty_weight,
            cg_station=cg_station, Iy=Iy)
        self.translator.set_aero_table(machs=machs, cd0_values=cd0_values)
        self.translator.set_propulsion(
            thrust=thrust, isp=isp, burn_time=burn_time,
            propellant_weight=propellant_weight,
            sustain_thrust=sustain_thrust, sustain_isp=sustain_isp,
            sustain_burn_time=sustain_burn_time,
            sustain_propellant_weight=sustain_propellant)

    def run_simulation(self, max_time=120.0, dt=0.01):
        """
        Run the JSBSim 6DOF simulation.

        Parameters
        ----------
        max_time : float
            Maximum simulation time in seconds.
        dt : float
            Time step in seconds.

        Returns
        -------
        results : dict
            Dictionary with time histories and summary metrics.
        """
        if not JSBSIM_AVAILABLE:
            raise ImportError("JSBSim is not installed. Run: pip install jsbsim")

        # Create temp directory for JSBSim files
        self._work_dir = tempfile.mkdtemp(prefix='tmd_jsbsim_')
        aircraft_name = 'tmd_missile'

        # Create aircraft directory structure
        aircraft_dir = os.path.join(self._work_dir, 'aircraft', aircraft_name)
        engine_dir = os.path.join(self._work_dir, 'engine')
        os.makedirs(aircraft_dir, exist_ok=True)
        os.makedirs(engine_dir, exist_ok=True)

        # Write aircraft XML
        aircraft_file = os.path.join(aircraft_dir, f'{aircraft_name}.xml')
        self.translator.write_xml(aircraft_file)

        # Write reset file (initial conditions)
        reset_file = os.path.join(aircraft_dir, 'reset00.xml')
        generate_reset_file(reset_file, mach=self.launch_mach,
                           altitude=self.launch_alt)

        # Write engine file (JSBSim needs it in the engine directory)
        # For simplicity, we'll use a direct thrust specification in the
        # aircraft file instead of a separate engine file

        # Initialize JSBSim
        fdm = jsbsim.FGFDMExec(self._work_dir)
        fdm.set_aircraft_path(os.path.join(self._work_dir, 'aircraft'))
        fdm.set_engine_path(engine_dir)
        fdm.set_systems_path(os.path.join(self._work_dir, 'systems'))
        os.makedirs(os.path.join(self._work_dir, 'systems'), exist_ok=True)

        # Load the model
        fdm.load_model(aircraft_name, False)

        # Set initial conditions
        fdm['ic/vc-kts'] = self.launch_mach * 661.5  # approximate conversion
        fdm['ic/h-sl-ft'] = self.launch_alt
        fdm['ic/gamma-deg'] = 0.0  # level flight
        fdm['ic/psi-true-deg'] = 0.0

        fdm.run_ic()
        fdm.set_dt(dt)

        # Data recording
        time_hist = []
        mach_hist = []
        alt_hist = []
        vel_hist = []
        range_hist = []
        lat_hist = []
        lon_hist = []

        # Run simulation
        total_range_ft = 0.0
        prev_x = 0.0

        initial_lat = fdm['position/lat-gc-deg']
        initial_lon = fdm['position/long-gc-deg']

        for step in range(int(max_time / dt)):
            fdm.run()

            t = fdm['simulation/sim-time-sec']
            M = fdm['velocities/mach']
            h = fdm['position/h-sl-ft']
            v = fdm['velocities/vc-fps']

            # Approximate range from ground speed integration
            vg = fdm['velocities/v-fps'] if 'velocities/v-fps' in dir(fdm) else v
            total_range_ft += abs(vg) * dt

            time_hist.append(t)
            mach_hist.append(M)
            alt_hist.append(h)
            vel_hist.append(v)
            range_hist.append(total_range_ft / 6076.12)  # ft to nmi

            # Check termination
            if M < self.terminal_mach and t > 5.0:
                break
            if h < 0:
                break
            if t > max_time:
                break

        self.results = {
            'time': np.array(time_hist),
            'mach': np.array(mach_hist),
            'altitude': np.array(alt_hist),
            'velocity': np.array(vel_hist),
            'range_nmi': np.array(range_hist),
            # Summary
            'final_time': time_hist[-1] if time_hist else 0.0,
            'final_range_nmi': range_hist[-1] if range_hist else 0.0,
            'max_mach': max(mach_hist) if mach_hist else 0.0,
            'final_mach': mach_hist[-1] if mach_hist else 0.0,
            'final_alt': alt_hist[-1] if alt_hist else 0.0,
        }

        return self.results

    def print_comparison(self, tmd_range=None, tmd_time=None, tmd_max_mach=None):
        """
        Print comparison between TMD predictions and JSBSim results.

        Parameters
        ----------
        tmd_range : float, optional
            TMD predicted range in nmi.
        tmd_time : float, optional
            TMD predicted time-to-target in seconds.
        tmd_max_mach : float, optional
            TMD predicted maximum Mach number.
        """
        if self.results is None:
            print("No simulation results. Run run_simulation() first.")
            return

        r = self.results

        print("=" * 65)
        print("  TMD vs JSBSim 6DOF — TRAJECTORY COMPARISON")
        print("=" * 65)
        print(f"\n  {'Metric':<25} {'TMD':>12} {'JSBSim 6DOF':>14} {'Delta':>10}")
        print(f"  {'─'*25} {'─'*12} {'─'*14} {'─'*10}")

        if tmd_range is not None:
            delta = ((r['final_range_nmi'] - tmd_range) / tmd_range * 100
                     if tmd_range > 0 else 0)
            print(f"  {'Range (nmi)':<25} {tmd_range:>12.2f} "
                  f"{r['final_range_nmi']:>14.2f} {delta:>9.1f}%")

        if tmd_time is not None:
            delta = ((r['final_time'] - tmd_time) / tmd_time * 100
                     if tmd_time > 0 else 0)
            print(f"  {'Flight time (s)':<25} {tmd_time:>12.1f} "
                  f"{r['final_time']:>14.1f} {delta:>9.1f}%")

        if tmd_max_mach is not None:
            delta = ((r['max_mach'] - tmd_max_mach) / tmd_max_mach * 100
                     if tmd_max_mach > 0 else 0)
            print(f"  {'Max Mach':<25} {tmd_max_mach:>12.2f} "
                  f"{r['max_mach']:>14.2f} {delta:>9.1f}%")

        print(f"\n  {'Final Mach':<25} {'':>12} {r['final_mach']:>14.2f}")
        print(f"  {'Final altitude (ft)':<25} {'':>12} {r['final_alt']:>14.0f}")
        print(f"  {'Simulation steps':<25} {'':>12} {len(r['time']):>14}")

        print(f"\n  NOTE: TMD uses 1DOF co-altitude model; JSBSim uses full 6DOF.")
        print(f"  Differences are expected and indicate the fidelity gap between")
        print(f"  conceptual (TMD) and high-fidelity (JSBSim) trajectory models.")
        print("=" * 65)

    def cleanup(self):
        """Remove temporary files."""
        if self._work_dir and os.path.exists(self._work_dir):
            import shutil
            shutil.rmtree(self._work_dir, ignore_errors=True)
