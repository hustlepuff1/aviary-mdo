"""
Missile6DOFBuilder -- Aviary SubsystemBuilder for the RocketPy 6-DOF subsystem.

Integrates high-fidelity 6-DOF trajectory simulation into Aviary's
problem structure following the SubsystemBuilder pattern.

Usage:
    from aviary_mdo.missile_6dof.missile_6dof_builder import (
        Missile6DOFBuilder,
    )

    builder = Missile6DOFBuilder(name='missile_6dof')

    # Pass to AviaryProblem as an external subsystem:
    prob.load_inputs(..., external_subsystems=[builder])

    # Or use standalone:
    prob = om.Problem()
    prob.model.add_subsystem(
        'missile_6dof', builder.build_pre_mission(aviary_inputs))
    prob.setup()
    prob.run_model()
"""

from aviary_mdo.missile_6dof.missile_6dof_meta_data import (
    ExtendedMetaData6DOF,
)
from aviary_mdo.missile_6dof.missile_6dof_variables import (
    DynamicMissile6DOF,
    Missile6DOF,
)
from aviary_mdo.missile_6dof.model.rocketpy_premission import (
    Missile6DOFPreMission,
)
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class Missile6DOFBuilder(SubsystemBuilder):
    """
    Aviary SubsystemBuilder for the RocketPy 6-DOF missile subsystem.

    Provides high-fidelity 6-DOF trajectory simulation, Barrowman
    aerodynamics, and solid motor grain regression via RocketPy.

    Parameters
    ----------
    name : str, optional
        Name of the subsystem (default: ``'missile_6dof'``).
    """

    default_name = 'missile_6dof'

    def __init__(self, name=None):
        super().__init__(name, meta_data=ExtendedMetaData6DOF)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build the pre-mission system for 6-DOF missile analysis.

        Returns an OpenMDAO Group that runs the full RocketPy flight
        simulation, producing trajectory, motor, stability, and aero
        outputs.

        Parameters
        ----------
        aviary_inputs : AviaryValues or dict
            Input values for the missile design.

        Returns
        -------
        pre_mission_sys : Missile6DOFPreMission
            OpenMDAO Group with all pre-mission computations.
        """
        return Missile6DOFPreMission()

    def get_mass_names(self):
        """Return mass variable names for Aviary's mass buildup."""
        return [Missile6DOF.Body.MASS]

    def get_design_vars(self):
        """
        Return design variables for 6-DOF missile optimization.

        These are the primary parameters a designer would vary during
        conceptual design trade studies.
        """
        return {
            Missile6DOF.Body.LENGTH: {
                'units': 'm',
                'lower': 1.0,
                'upper': 8.0,
            },
            Missile6DOF.Body.RADIUS: {
                'units': 'm',
                'lower': 0.05,
                'upper': 0.3,
            },
            Missile6DOF.Fins.SPAN: {
                'units': 'm',
                'lower': 0.05,
                'upper': 0.5,
            },
            Missile6DOF.Fins.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 0.5,
            },
            Missile6DOF.Launch.INCLINATION: {
                'units': 'deg',
                'lower': 5.0,
                'upper': 90.0,
            },
        }

    def get_parameters(self, aviary_inputs=None, **kwargs):
        """
        Return fixed parameters for the 6-DOF missile subsystem.

        These are values that remain constant during an analysis but
        can be changed between runs.
        """
        return {
            Missile6DOF.Launch.RAIL_LENGTH: {
                'val': 5.0,
                'units': 'm',
            },
            Missile6DOF.Launch.HEADING: {
                'val': 0.0,
                'units': 'deg',
            },
            Missile6DOF.Environment.ELEVATION: {
                'val': 0.0,
                'units': 'm',
            },
        }

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate a 6-DOF missile analysis summary report.

        Parameters
        ----------
        prob : AviaryProblem
            The solved problem instance.
        reports_folder : Path
            Destination directory for the report file.
        """
        try:
            report_path = reports_folder / f'{self.name}_summary.txt'
            apogee = prob.get_val(
                f'{self.name}.{DynamicMissile6DOF.APOGEE}', units='m')
            max_speed = prob.get_val(
                f'{self.name}.{DynamicMissile6DOF.MAX_SPEED}', units='m/s')
            flight_time = prob.get_val(
                f'{self.name}.{DynamicMissile6DOF.FLIGHT_TIME}', units='s')
            range_total = prob.get_val(
                f'{self.name}.{DynamicMissile6DOF.RANGE_TOTAL}', units='m')
            max_mach = prob.get_val(
                f'{self.name}.{DynamicMissile6DOF.MAX_MACH}')
            sm_ign = prob.get_val(
                f'{self.name}.'
                f'{DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION}')

            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("6-DOF MISSILE ANALYSIS (RocketPy) - SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Apogee:              {apogee[0]:.1f} m\n")
                f.write(f"Max Speed:           {max_speed[0]:.1f} m/s\n")
                f.write(f"Max Mach:            {max_mach[0]:.2f}\n")
                f.write(f"Flight Time:         {flight_time[0]:.1f} s\n")
                f.write(f"Total Range:         {range_total[0]:.1f} m\n")
                f.write(
                    f"Static Margin (ign): {sm_ign[0]:.2f} cal\n")
                f.write("\n" + "=" * 60 + "\n")
        except Exception:
            pass
