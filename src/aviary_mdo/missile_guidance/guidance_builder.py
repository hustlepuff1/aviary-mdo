"""
GuidanceBuilder -- Aviary SubsystemBuilder for missile guidance analysis.

Integrates the missile guidance subsystem into Aviary's problem structure
following the SubsystemBuilder pattern. Provides engagement simulation
with proportional navigation, augmented PN, and optimal guidance laws
for interceptor missile MDO studies.

Usage:
    from aviary_mdo.missile_guidance.guidance_builder import GuidanceBuilder

    guidance = GuidanceBuilder(name='guidance')

    # Pass to AviaryProblem as an external subsystem:
    prob.load_inputs(..., external_subsystems=[guidance])

    # Or use standalone:
    prob = om.Problem()
    prob.model.add_subsystem('guidance', guidance.build_pre_mission(aviary_inputs))
    prob.setup()
    prob.run_model()
"""

from aviary_mdo.missile_guidance.guidance_meta_data import ExtendedMetaDataGuidance
from aviary_mdo.missile_guidance.guidance_variables import Guidance, DynamicGuidance
from aviary_mdo.missile_guidance.model.guidance_premission import GuidancePremission
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class GuidanceBuilder(SubsystemBuilder):
    """
    Aviary SubsystemBuilder for missile guidance analysis.

    Provides engagement simulation with proportional navigation,
    augmented PN, and optimal guidance laws for interceptor missile
    MDO studies.
    """

    default_name = 'guidance'

    def __init__(self, name=None):
        super().__init__(name, meta_data=ExtendedMetaDataGuidance)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build the pre-mission system for guidance analysis.

        Returns an OpenMDAO Group containing the guided engagement
        simulation with PN/APN/OGL guidance laws and first-order
        autopilot lag.

        Parameters
        ----------
        aviary_inputs : AviaryValues or dict
            Input values for the guidance analysis.

        Returns
        -------
        pre_mission_sys : GuidancePremission
            OpenMDAO Group with engagement simulation.
        """
        return GuidancePremission()

    def get_mass_names(self):
        """
        Return mass variable names for the guidance subsystem.
        """
        return [Guidance.Missile.MASS]

    def get_design_vars(self):
        """
        Return design variables for guidance optimization.
        """
        return {
            Guidance.Missile.MAX_ACCELERATION: {
                'units': 'm/s**2', 'lower': 50.0, 'upper': 500.0,
            },
            Guidance.Navigation.RATIO: {
                'lower': 2.0, 'upper': 6.0,
            },
            Guidance.Autopilot.TIME_CONSTANT: {
                'units': 's', 'lower': 0.01, 'upper': 1.0,
            },
        }

    def get_parameters(self, aviary_inputs=None, **kwargs):
        """
        Return fixed parameters for the guidance subsystem.
        """
        return {
            Guidance.Target.VELOCITY: {'val': 300.0, 'units': 'm/s'},
            Guidance.Target.ACCELERATION_MAX: {'val': 50.0, 'units': 'm/s**2'},
        }

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate a guidance analysis summary report.
        """
        try:
            report_path = reports_folder / f'{self.name}_summary.txt'
            miss = prob.get_val(
                f'{self.name}.{DynamicGuidance.MISS_DISTANCE}', units='m')
            pk = prob.get_val(
                f'{self.name}.{DynamicGuidance.SINGLE_SHOT_PK}')
            t_int = prob.get_val(
                f'{self.name}.{DynamicGuidance.TIME_TO_INTERCEPT}', units='s')
            max_g = prob.get_val(
                f'{self.name}.{DynamicGuidance.REQUIRED_LOAD_FACTOR}')

            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("MISSILE GUIDANCE ANALYSIS - SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Miss Distance:    {miss[0]:.2f} m\n")
                f.write(f"Single-Shot Pk:   {pk[0]:.3f}\n")
                f.write(f"Time to Intercept:{t_int[0]:.1f} s\n")
                f.write(f"Peak Load Factor: {max_g[0]:.1f} g\n")
                f.write("\n" + "=" * 60 + "\n")
        except Exception:
            pass
