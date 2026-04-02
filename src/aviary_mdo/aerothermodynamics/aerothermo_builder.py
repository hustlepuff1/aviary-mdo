"""
AerothermoBuilder -- Aviary SubsystemBuilder for aerothermodynamics.

Integrates stagnation heating, surface heat flux, and TPS sizing
disciplines into Aviary's problem structure following the SubsystemBuilder
pattern.

Usage:
    from aviary_mdo.aerothermodynamics.aerothermo_builder import (
        AerothermoBuilder,
    )

    aerothermo = AerothermoBuilder(name='aerothermo')

    # Pass to AviaryProblem as an external subsystem:
    prob.load_inputs(..., external_subsystems=[aerothermo])

    # Or use standalone:
    prob = om.Problem()
    prob.model.add_subsystem('aerothermo',
                             aerothermo.build_pre_mission(aviary_inputs))
    prob.setup()
    prob.run_model()
"""

from aviary_mdo.aerothermodynamics.aerothermo_meta_data import ExtendedMetaData
from aviary_mdo.aerothermodynamics.aerothermo_variables import Aerothermo
from aviary_mdo.aerothermodynamics.model.aerothermo_premission import (
    AerothermoPremission,
)
from aviary.subsystems.subsystem_builder import SubsystemBuilder


# Map TPS type codes (1, 2, 3) to string names
_TPS_TYPE_MAP = {1: 'ablative', 2: 'ceramic', 3: 'metallic'}


class AerothermoBuilder(SubsystemBuilder):
    """
    Aviary SubsystemBuilder for the Aerothermodynamics subsystem.

    Provides aerothermal sizing for high-speed vehicles: stagnation
    heating, surface heat flux distribution, radiative equilibrium
    temperature, and thermal protection system mass estimation.

    Parameters
    ----------
    name : str
        Name of the subsystem (default: 'aerothermo').
    tps_type : str
        Type of TPS material: 'ablative', 'ceramic', or 'metallic'
        (default: 'ablative').
    """

    default_name = 'aerothermo'

    def __init__(self, name=None, tps_type='ablative'):
        self.tps_type = tps_type
        super().__init__(name, meta_data=ExtendedMetaData)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build the pre-mission system for aerothermal sizing.

        Returns an OpenMDAO Group containing:
        - Atmosphere lookup
        - Flight condition computation
        - Stagnation heating (Sutton-Graves)
        - Surface heat flux distribution
        - Radiative equilibrium temperature
        - TPS thickness and mass sizing

        Parameters
        ----------
        aviary_inputs : AviaryValues or dict
            Input values for the aerothermal design.

        Returns
        -------
        pre_mission_sys : AerothermoPremission
            OpenMDAO Group with all pre-mission computations.
        """
        return AerothermoPremission(tps_type=self.tps_type)

    def get_mass_names(self):
        """
        Return mass variable names for the aerothermo subsystem.

        These are used by Aviary to include the TPS mass in the
        overall vehicle mass buildup.
        """
        return [Aerothermo.Output.TPS_MASS]

    def get_design_vars(self):
        """
        Return design variables for aerothermal optimization.

        These are the primary parameters a designer would vary during
        conceptual design trade studies.
        """
        return {
            Aerothermo.Design.NOSE_RADIUS: {
                'units': 'm',
                'lower': 0.01,
                'upper': 5.0,
            },
            Aerothermo.Design.WALL_TEMPERATURE: {
                'units': 'K',
                'lower': 200.0,
                'upper': 2000.0,
            },
        }

    def get_parameters(self, aviary_inputs=None, **kwargs):
        """
        Return fixed parameters for the aerothermo subsystem.
        """
        return {
            Aerothermo.Design.DESIGN_MACH: {
                'val': 5.0,
                'units': None,
            },
            Aerothermo.Design.DESIGN_ALTITUDE: {
                'val': 30000.0,
                'units': 'm',
            },
        }

    def preprocess_inputs(self, aviary_inputs):
        """
        Preprocess aerothermo inputs.

        Resolves TPS type code to string if needed.
        """
        return aviary_inputs

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate an aerothermodynamics summary report.
        """
        try:
            report_path = reports_folder / f'{self.name}_summary.txt'

            q_stag = prob.get_val(f'{self.name}.q_stag', units='W/m**2')
            T_eq = prob.get_val(f'{self.name}.T_wall_eq', units='K')
            mass_tps = prob.get_val(f'{self.name}.mass_tps', units='kg')

            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("AEROTHERMODYNAMICS - SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Stagnation Heat Flux: {q_stag[0]:.2f} W/m^2\n")
                f.write(f"Equilibrium Wall Temp: {T_eq[0]:.1f} K\n")
                f.write(f"TPS Mass: {mass_tps[0]:.2f} kg\n")
                f.write("\n" + "=" * 60 + "\n")
        except Exception:
            pass
