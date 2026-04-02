"""Tests for the pyCycle engine integration.

Tests import, builder creation, and design-point cycle execution.
"""

import pytest
import numpy as np

try:
    import pycycle
    HAS_PYCYCLE = True
except ImportError:
    HAS_PYCYCLE = False

pytestmark = pytest.mark.skipif(
    not HAS_PYCYCLE,
    reason='pyCycle not installed (pip install om-pycycle)')


class TestPyCycleImports:
    """Test that pyCycle modules import correctly."""

    def test_import_builder(self):
        from aviary_mdo.engine_pycycle import PyCycleEngineBuilder
        builder = PyCycleEngineBuilder()
        assert builder.cycle_type == 'turbojet'

    def test_import_cycles(self):
        from aviary_mdo.engine_pycycle.cycles import (
            TurbojetCycle, MPTurbojet,
            TurbofanCycle, MPTurbofan,
            TurboshaftCycle, MPTurboshaft,
        )
        assert TurbojetCycle is not None
        assert MPTurbojet is not None

    def test_invalid_cycle_type(self):
        from aviary_mdo.engine_pycycle import PyCycleEngineBuilder
        with pytest.raises(ValueError, match='cycle_type must be one of'):
            PyCycleEngineBuilder(cycle_type='scramjet')

    def test_builder_creates_group(self):
        import openmdao.api as om
        from aviary_mdo.engine_pycycle import PyCycleEngineBuilder
        builder = PyCycleEngineBuilder(cycle_type='turbojet')
        group = builder.build_pre_mission()
        assert isinstance(group, om.Group)


class TestTurbojetCycle:
    """Test turbojet design-point analysis."""

    def test_design_point(self):
        """Run a turbojet design point and verify thrust/TSFC."""
        import openmdao.api as om
        from aviary_mdo.engine_pycycle.cycles.turbojet import MPTurbojet

        prob = om.Problem()
        prob.model = MPTurbojet()
        prob.setup(check=False)

        # Design conditions
        prob.set_val('DESIGN.fc.alt', 0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)

        # Initial guesses
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 168.453135137
        prob['DESIGN.balance.turb_PR'] = 4.46138725662
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        for pt in prob.model.od_pts:
            prob[f'{pt}.balance.W'] = 166.073
            prob[f'{pt}.balance.FAR'] = 0.01680
            prob[f'{pt}.balance.Nmech'] = 8197.38
            prob[f'{pt}.fc.balance.Pt'] = 15.703
            prob[f'{pt}.fc.balance.Tt'] = 558.31
            prob[f'{pt}.turb.PR'] = 4.6690

        prob.set_solver_print(level=-1)
        prob.run_model()

        # Verify design point converged near target
        fn = prob.get_val('DESIGN.perf.Fn', units='lbf')
        assert abs(fn - 11800.0) < 50.0, f'Fn={fn}, expected ~11800 lbf'

        tsfc = prob.get_val('DESIGN.perf.TSFC')
        assert 0.5 < tsfc < 2.0, f'TSFC={tsfc}, outside reasonable range'

        opr = prob.get_val('DESIGN.perf.OPR')
        assert opr > 10.0, f'OPR={opr}, expected >10'


class TestTurbofanCycle:
    """Test turbofan design-point setup."""

    def test_turbofan_setup(self):
        """Verify turbofan cycle sets up without errors."""
        import openmdao.api as om
        from aviary_mdo.engine_pycycle.cycles.turbofan import MPTurbofan

        prob = om.Problem()
        prob.model = MPTurbofan()
        prob.setup(check=False)

        # Just verify it set up — full convergence takes longer
        assert 'DESIGN' in prob.model._subsystems_allprocs


class TestTurboshaftCycle:
    """Test turboshaft design-point setup."""

    def test_turboshaft_setup(self):
        """Verify turboshaft cycle sets up without errors."""
        import openmdao.api as om
        from aviary_mdo.engine_pycycle.cycles.turboshaft import MPTurboshaft

        prob = om.Problem()
        prob.model = MPTurboshaft()
        prob.setup(check=False)

        assert 'DESIGN' in prob.model._subsystems_allprocs
