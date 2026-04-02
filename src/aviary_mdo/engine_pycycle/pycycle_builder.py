"""
PyCycleEngineBuilder — Aviary SubsystemBuilder for pyCycle engines.

Wraps pyCycle thermodynamic cycle models as Aviary external subsystems.
Supports turbojet, turbofan, and turboshaft configurations.

Usage:
    from aviary_mdo.engine_pycycle import PyCycleEngineBuilder

    # Simple turbojet
    engine = PyCycleEngineBuilder(
        name='turbojet',
        cycle_type='turbojet',
        design_thrust=11800.0,
        design_T4=2370.0,
    )

    # High-bypass turbofan
    engine = PyCycleEngineBuilder(
        name='turbofan',
        cycle_type='turbofan',
        design_thrust=28000.0,
        design_T4=3200.0,
    )

Reference: Hendricks & Gray, "pyCycle: A Tool for Efficient Optimization
of Gas Turbine Engine Cycles," Aerospace, vol. 6, no. 87, 2019.
"""

import openmdao.api as om
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class PyCycleEngineBuilder(SubsystemBuilder):
    """Aviary SubsystemBuilder wrapping pyCycle thermodynamic cycle models.

    Parameters
    ----------
    name : str
        Subsystem name (default: 'pycycle_engine').
    cycle_type : str
        Engine type: 'turbojet', 'turbofan', or 'turboshaft'.
    design_thrust : float
        Design-point net thrust (lbf). For turboshaft, this is shaft power (hp).
    design_T4 : float
        Design-point turbine inlet temperature T4 (degR).
    design_alt : float
        Design-point altitude (ft).
    design_MN : float
        Design-point Mach number.
    comp_PR : float
        Compressor pressure ratio (turbojet/turboshaft only).
    use_tabular : bool
        Use tabular thermo (faster) vs CEA.
    """

    default_name = 'pycycle_engine'

    VALID_TYPES = ('turbojet', 'turbofan', 'turboshaft')

    def __init__(self, name=None, cycle_type='turbojet',
                 design_thrust=11800.0, design_T4=2370.0,
                 design_alt=0.0, design_MN=0.000001,
                 comp_PR=13.5, use_tabular=True):
        if cycle_type not in self.VALID_TYPES:
            raise ValueError(
                f"cycle_type must be one of {self.VALID_TYPES}, got '{cycle_type}'")
        self.cycle_type = cycle_type
        self.design_thrust = design_thrust
        self.design_T4 = design_T4
        self.design_alt = design_alt
        self.design_MN = design_MN
        self.comp_PR = comp_PR
        self.use_tabular = use_tabular
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs=None, **kwargs):
        """Build a pyCycle multi-point engine model.

        Returns an OpenMDAO Group containing the design-point cycle.
        """
        group = om.Group()

        if self.cycle_type == 'turbojet':
            from aviary_mdo.engine_pycycle.cycles.turbojet import MPTurbojet
            mp = MPTurbojet(use_tabular=self.use_tabular)
            group.add_subsystem('mp_cycle', mp)

        elif self.cycle_type == 'turbofan':
            from aviary_mdo.engine_pycycle.cycles.turbofan import MPTurbofan
            mp = MPTurbofan(use_tabular=self.use_tabular)
            group.add_subsystem('mp_cycle', mp)

        elif self.cycle_type == 'turboshaft':
            from aviary_mdo.engine_pycycle.cycles.turboshaft import MPTurboshaft
            mp = MPTurboshaft(use_tabular=self.use_tabular)
            group.add_subsystem('mp_cycle', mp)

        return group

    def get_mass_names(self):
        return []

    def get_outputs(self):
        return []
