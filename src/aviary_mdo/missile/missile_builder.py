"""
MissileBuilder — Aviary SubsystemBuilder for tactical missile design.

Integrates the TMD (Tactical Missile Design) disciplines into Aviary's
problem structure following the SubsystemBuilder pattern.

Usage:
    from aviary_mdo.missile.missile_builder import MissileBuilder

    missile = MissileBuilder(name='missile', sustain_engine='rocket')

    # Pass to AviaryProblem as an external subsystem:
    prob.load_inputs(..., external_subsystems=[missile])

    # Or use standalone:
    prob = om.Problem()
    prob.model.add_subsystem('missile', missile.build_pre_mission(aviary_inputs))
    prob.setup()
    prob.run_model()

Reference: Fleeman, E., "Tactical Missile Design," AIAA Education Series, 2001.
"""

from aviary_mdo.missile.missile_meta_data import ExtendedMetaData
from aviary_mdo.missile.missile_variables import Missile, DynamicMissile
from aviary_mdo.missile.model.missile_premission import MissilePreMission
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class MissileBuilder(SubsystemBuilder):
    """
    Aviary SubsystemBuilder for the Tactical Missile Design subsystem.

    This builder provides a complete missile conceptual sizing tool based
    on Fleeman's TMD methodology. It integrates aerodynamics, propulsion,
    trajectory, structures, warhead, radar, and dynamics disciplines.

    The missile sizing is primarily a pre-mission computation — given
    geometry, propulsion parameters, and flight conditions, it computes
    range, time-to-target, skin temperature, structural loads, lethality,
    and detection range.

    Parameters
    ----------
    name : str
        Name of the subsystem (default: 'missile').
    sustain_engine : str
        Type of sustain engine: 'rocket' or 'ramjet' (default: 'rocket').
    """

    default_name = 'missile'

    def __init__(self, name=None, sustain_engine='rocket'):
        self.sustain_engine = sustain_engine
        super().__init__(name, meta_data=ExtendedMetaData)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build the pre-mission system for missile sizing.

        Returns an OpenMDAO Group containing all TMD disciplines:
        - Aerodynamics (drag buildup, normal force)
        - Propulsion (rocket / ramjet cycle analysis)
        - Trajectory (boost / sustain / coast)
        - Structures (motor case, skin temperature)

        Parameters
        ----------
        aviary_inputs : AviaryValues or dict
            Input values for the missile design.

        Returns
        -------
        pre_mission_sys : MissilePreMission
            OpenMDAO Group with all pre-mission computations.
        """
        return MissilePreMission(sustain_engine=self.sustain_engine)

    def get_mass_names(self):
        """
        Return mass variable names for the missile subsystem.

        These are used by Aviary to include the missile mass in the
        overall vehicle mass buildup.
        """
        return [Missile.Weight.LAUNCH]

    def get_design_vars(self):
        """
        Return design variables for missile optimization.

        These are the primary parameters a designer would vary during
        conceptual design trade studies.
        """
        return {
            Missile.Body.LENGTH: {
                'units': 'inch',
                'lower': 60.0,
                'upper': 250.0,
            },
            Missile.Body.DIAMETER_MAJOR: {
                'units': 'inch',
                'lower': 4.0,
                'upper': 24.0,
            },
            Missile.Wing.PLANFORM_AREA: {
                'units': 'inch**2',
                'lower': 100.0,
                'upper': 800.0,
            },
            Missile.Propulsion.BOOST_EXPANSION_RATIO: {
                'lower': 2.0,
                'upper': 20.0,
            },
            Missile.Propulsion.BOOST_CHAMBER_PRESSURE: {
                'units': 'lbf/inch**2',
                'lower': 500.0,
                'upper': 3000.0,
            },
            Missile.Weight.LAUNCH: {
                'units': 'lbm',
                'lower': 200.0,
                'upper': 2500.0,
            },
        }

    def get_parameters(self, aviary_inputs=None, **kwargs):
        """
        Return fixed parameters for the missile subsystem.

        These are values that remain constant during a mission phase
        but can be changed between analyses.
        """
        return {
            Missile.Flight.LAUNCH_MACH: {
                'val': 0.8,
                'units': None,
            },
            Missile.Flight.LAUNCH_ALTITUDE: {
                'val': 20000.0,
                'units': 'ft',
            },
            Missile.Flight.TERMINAL_MACH: {
                'val': 1.5,
                'units': None,
            },
        }

    def preprocess_inputs(self, aviary_inputs):
        """
        Preprocess missile inputs.

        Computes derived geometry values (reference area, fineness ratio)
        from the primary inputs before the analysis runs.
        """
        import numpy as np

        if aviary_inputs is not None:
            try:
                d_major = aviary_inputs.get_val(
                    Missile.Body.DIAMETER_MAJOR, units='inch')
                d_minor = aviary_inputs.get_val(
                    Missile.Body.DIAMETER_MINOR, units='inch')

                # Reference area (elliptical cross-section)
                a = d_major / 2.0
                b = d_minor / 2.0
                S_ref = np.pi * a * b
                aviary_inputs.set_val(
                    Missile.Body.REFERENCE_AREA, S_ref, units='inch**2',
                    meta_data=ExtendedMetaData)

                # Fineness ratio
                length = aviary_inputs.get_val(
                    Missile.Body.LENGTH, units='inch')
                d_eff = np.sqrt(d_major * d_minor)
                aviary_inputs.set_val(
                    Missile.Body.FINENESS_RATIO, length / d_eff,
                    meta_data=ExtendedMetaData)
            except Exception:
                pass

        return aviary_inputs

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate a TMD summary report.

        Outputs key performance metrics matching the TMD spreadsheet
        Summary sheet format.
        """
        try:
            report_path = reports_folder / f'{self.name}_summary.txt'

            range_total = prob.get_val(f'{self.name}.range_total', units='nmi')
            time_total = prob.get_val(f'{self.name}.time_total', units='s')

            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("TACTICAL MISSILE DESIGN - SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total Flight Range: {range_total[0]:.2f} nmi\n")
                f.write(f"Total Flight Time:  {time_total[0]:.1f} s\n")
                f.write("\n" + "=" * 60 + "\n")
        except Exception:
            pass
