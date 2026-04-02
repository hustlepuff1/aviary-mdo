"""NPSS Tabular Engine Builder for Aviary.

Provides an EngineModel that wraps NPSS for design-point analysis and
uses a MetaModelSemiStructuredComp for off-design mission interpolation.

Source: OpenMDAO/Aviary_Community (adapted for aviary-mdo package).
"""

import os
from pathlib import Path

import numpy as np
import openmdao.api as om

from aviary_mdo.engine_npss.connected_variables import vars_to_connect
from aviary_mdo.engine_npss.NPSS_Model.design_engine_group import DesignEngineGroup
from aviary_mdo.engine_npss.npss_variable_meta_data import ExtendedMetaData
from aviary_mdo.engine_npss.npss_variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues


# Path to bundled engine output data
_OUTPUT_DIR = Path(__file__).parent / 'NPSS_Model' / 'Output'


class NPSSTabularEngineBuilder(EngineModel):
    """NPSS engine builder from table.

    During pre-mission the engine is designed. During mission, the
    engine deck is interpolated for off-design performance.

    Parameters
    ----------
    name : str
        Engine model name (default: 'NPSS_prop_system').
    aviary_inputs : AviaryValues
        Default Aviary options.
    engine_output : str or Path, optional
        Path to RefEngine.outputAviary file. Defaults to bundled data.
    """

    def __init__(self, name='NPSS_prop_system', aviary_inputs=AviaryValues(),
                 engine_output=None):
        super().__init__(name, options=aviary_inputs, meta_data=ExtendedMetaData)
        self._engine_output = engine_output

    def build_pre_mission(self, aviary_inputs=AviaryValues()):
        return DesignEngineGroup()

    def build_mission(self, num_nodes, aviary_inputs):
        interp_method = aviary_inputs.get_val(Aircraft.Engine.INTERPOLATION_METHOD)[0]

        engine = om.MetaModelSemiStructuredComp(
            method=interp_method,
            extrapolate=True,
            vec_size=num_nodes,
            training_data_gradients=True,
        )

        # Load engine performance data
        if self._engine_output:
            csv_path = str(self._engine_output)
        else:
            csv_path = str(_OUTPUT_DIR / 'RefEngine.outputAviary')

        engine_data = np.genfromtxt(csv_path, skip_header=0)

        # Sort by Mach, altitude, throttle
        engine_data = engine_data[
            np.lexsort((engine_data[:, 2], engine_data[:, 1], engine_data[:, 0]))
        ]

        zeros_array = np.zeros((engine_data.shape[0], 1))
        thrust_max = np.zeros((engine_data.shape[0], 1))

        for i in range(engine_data.shape[0]):
            index = np.where(
                (engine_data[:, 0] == engine_data[i, 0])
                & (engine_data[:, 1] == engine_data[i, 1])
                & (engine_data[:, 2] == 1.0)
            )[0][0]
            thrust_max[i] = engine_data[index, 3]

        engine.add_input(
            Dynamic.Atmosphere.MACH, engine_data[:, 0],
            units='unitless', desc='Current flight Mach number')
        engine.add_input(
            Dynamic.Mission.ALTITUDE, engine_data[:, 1],
            units='ft', desc='Current flight altitude')
        engine.add_input(
            Dynamic.Vehicle.Propulsion.THROTTLE, engine_data[:, 2],
            units='unitless', desc='Current engine throttle')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.THRUST, engine_data[:, 3],
            units='lbf', desc='Current net thrust produced')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.THRUST_MAX, thrust_max,
            units='lbf', desc='Max net thrust produced')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE, -engine_data[:, 4],
            units='lbm/s', desc='Current fuel flow rate')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN, zeros_array,
            units='kW', desc='Current electric energy rate')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.NOX_RATE, zeros_array,
            units='lb/h', desc='Current NOx emission rate')
        engine.add_output(
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4, zeros_array,
            units='degR', desc='Current turbine exit temperature')
        return engine

    def get_pre_mission_bus_variables(self, aviary_inputs=None):
        return vars_to_connect

    def get_controls(self, phase_name):
        return {}

    def get_design_vars(self):
        mass_flow_dict = {
            'units': 'lbm/s',
            'upper': 450,
            'lower': 100,
            'ref': 450,
        }
        return {Aircraft.Engine.DESIGN_MASS_FLOW: mass_flow_dict}
