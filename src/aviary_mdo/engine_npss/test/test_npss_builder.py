"""Tests for the NPSS engine builder integration.

The full NPSS benchmark test requires NPSS installed (runnpss on PATH).
Import and metadata tests run without NPSS.
"""

import os
import unittest

import pytest
import numpy as np


class TestNPSSImports(unittest.TestCase):
    """Test that NPSS modules import correctly."""

    def test_import_variables(self):
        from aviary_mdo.engine_npss.npss_variables import Aircraft, Dynamic
        assert hasattr(Aircraft.Engine, 'DESIGN_MACH')
        assert hasattr(Aircraft.Engine, 'DESIGN_MASS_FLOW')
        assert hasattr(Dynamic.Engine, 'SHAFT_MECH_SPEED')

    def test_import_metadata(self):
        from aviary_mdo.engine_npss.npss_variable_meta_data import ExtendedMetaData
        assert ExtendedMetaData is not None

    def test_import_connected_variables(self):
        from aviary_mdo.engine_npss.connected_variables import vars_to_connect
        assert 'Fn_train' in vars_to_connect
        assert 'Fn_max_train' in vars_to_connect
        assert 'Wf_inv_train' in vars_to_connect

    def test_import_builder(self):
        from aviary_mdo.engine_npss import NPSSTabularEngineBuilder
        builder = NPSSTabularEngineBuilder()
        assert builder.name == 'NPSS_prop_system'

    def test_engine_output_data_exists(self):
        """Bundled RefEngine.outputAviary should be present."""
        from pathlib import Path
        output_file = (
            Path(__file__).parent.parent / 'NPSS_Model' / 'Output' /
            'RefEngine.outputAviary')
        assert output_file.exists(), f'{output_file} not found'

        data = np.genfromtxt(str(output_file), skip_header=0)
        assert data.shape[1] >= 5  # Mach, Alt, Throttle, Thrust, FuelFlow
        assert data.shape[0] > 0

    def test_builder_pre_mission_creates_group(self):
        """build_pre_mission returns an OpenMDAO Group."""
        import openmdao.api as om
        from aviary_mdo.engine_npss import NPSSTabularEngineBuilder
        builder = NPSSTabularEngineBuilder()
        group = builder.build_pre_mission()
        assert isinstance(group, om.Group)


class TestNPSSBenchmark(unittest.TestCase):
    """Full NPSS benchmark — requires NPSS installed."""

    @unittest.skipUnless(
        os.environ.get('NPSS_TOP', False),
        'NPSS not installed (set NPSS_TOP env var)')
    def test_aviary_npss_mission(self):
        """Run a full Aviary mission with NPSS engine."""
        import aviary.api as av
        from aviary_mdo.engine_npss import NPSSTabularEngineBuilder
        from aviary_mdo.engine_npss.npss_variable_meta_data import ExtendedMetaData
        from copy import deepcopy
        from openmdao.utils.assert_utils import assert_near_equal

        phase_info = deepcopy(av.default_height_energy_phase_info)
        prob = av.AviaryProblem()
        prob.options['group_by_pre_opt_post'] = True

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            phase_info, meta_data=ExtendedMetaData)
        prob.load_external_subsystems([NPSSTabularEngineBuilder()])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('SLSQP')
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        assert_near_equal(
            prob.get_val('aircraft:engine:design_mass_flow'),
            315.1648646, tolerance=0.01)
