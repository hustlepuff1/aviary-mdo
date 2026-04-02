"""Tests for the Missile DATCOM Python wrapper and OpenMDAO integration.

These tests require the misdat executable to be built. They are skipped
if the executable is not found.
"""

import os
import pytest
import numpy as np

from aviary_mdo.datcom.model.datcom_wrapper import MissileDATCOM, find_executable


# Skip all tests if misdat executable not available
try:
    _exe = find_executable()
    HAS_DATCOM = True
except FileNotFoundError:
    HAS_DATCOM = False

pytestmark = pytest.mark.skipif(
    not HAS_DATCOM,
    reason='misdat executable not found — build with ifx/gfortran first'
)


# --- Reference data from the sample case (Mach 2.36, body+2 fin sets) ---
# These are the values DATCOM produces for the for005.dat sample input
SAMPLE_ALPHAS = [0., 4., 8., 12., 16., 20., 24., 28.]


class TestMissileDATCOMWrapper:
    """Tests for the MissileDATCOM Python wrapper."""

    def test_basic_run(self):
        """DATCOM runs and returns results for a simple body."""
        md = MissileDATCOM()
        results = md.run(
            mach=[2.36],
            alpha=[0, 4, 8, 12],
            reynolds=3e6,
            body={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25},
            xcg=18.75
        )
        assert 1 in results
        case = results[1]
        assert 'CN' in case
        assert len(case['CN']) == 4

    def test_coefficients_reasonable(self):
        """CN, CD values are physically reasonable at Mach 2.36."""
        md = MissileDATCOM()
        results = md.run(
            mach=[2.36],
            alpha=[0, 4, 8, 12, 16, 20, 24, 28],
            reynolds=3e6,
            body={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25, 'dexit': 2.0},
            xcg=18.75
        )
        case = results[1]

        # CN should be ~0 at alpha=0, increasing with alpha
        assert abs(case['CN'][0]) < 0.05
        assert case['CN'][-1] > 1.0  # significant CN at 28 deg

        # CD should be positive and increase with alpha
        assert all(case['CD'] > 0)
        assert case['CD'][-1] > case['CD'][0]

    def test_with_fins(self):
        """DATCOM handles body + fin sets."""
        md = MissileDATCOM()
        results = md.run(
            mach=[2.36],
            alpha=[0, 4, 8, 12],
            reynolds=3e6,
            body={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25, 'dexit': 2.0},
            fins=[
                {'xle': 15.42, 'npanel': 2, 'phif': [90, 270],
                 'sweep': 0, 'sta': 1,
                 'chord': [6.96, 0], 'sspan': [1.875, 5.355],
                 'zupper': [0.02238, 0.02238],
                 'lmaxu': [0.238, 0.238],
                 'lflatu': [0.524, 0.524],
                 'ler': [0.015, 0.015]},
            ],
            xcg=18.75,
            controls=['PART']
        )
        assert 1 in results
        # With fins, CN at alpha>0 should be larger than body-alone
        assert results[1]['CN'][1] > 0.1

    def test_subsonic(self):
        """DATCOM runs at subsonic Mach."""
        md = MissileDATCOM()
        results = md.run(
            mach=[0.6],
            alpha=[0, 4, 8],
            reynolds=1e6,
            body={'lnose': 5.0, 'dnose': 2.0, 'lcentr': 10.0},
            xcg=7.0
        )
        assert 1 in results

    def test_multiple_mach(self):
        """DATCOM handles multiple Mach numbers in one run."""
        md = MissileDATCOM()
        results = md.run(
            mach=[1.5, 2.0, 2.5],
            alpha=[0, 4, 8],
            reynolds=3e6,
            body={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25},
            xcg=18.75
        )
        assert 1 in results

    def test_input_file(self):
        """DATCOM runs with a pre-built for005.dat."""
        sample_input = os.path.join(
            os.path.expanduser('~'), 'MDO', 'Missile-Datcom', 'for005.dat')
        if not os.path.exists(sample_input):
            pytest.skip('Sample for005.dat not found')

        md = MissileDATCOM()
        results = md.run_input_file(sample_input)
        assert 1 in results


class TestDATCOMAeroComponent:
    """Tests for the OpenMDAO DATCOMAero component."""

    def test_openmdao_component(self):
        """DATCOMAero runs as an OpenMDAO component."""
        import openmdao.api as om
        from aviary_mdo.datcom.model.datcom_aero import DATCOMAero

        prob = om.Problem()
        prob.model.add_subsystem('datcom', DATCOMAero(
            n_alpha=4,
            body_config={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25}
        ))
        prob.setup()

        prob['datcom.mach'] = 2.36
        prob['datcom.alpha'] = [0, 4, 8, 12]
        prob['datcom.reynolds'] = 3e6
        prob['datcom.xcg'] = 18.75

        prob.run_model()

        cn = prob['datcom.CN']
        cd = prob['datcom.CD']
        assert len(cn) == 4
        assert abs(cn[0]) < 0.05  # CN ~ 0 at alpha=0
        assert all(cd > 0)

    def test_builder_integration(self):
        """DATCOMBuilder creates a working Aviary subsystem."""
        import openmdao.api as om
        from aviary_mdo.datcom import DATCOMBuilder

        builder = DATCOMBuilder(
            body_config={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25},
            n_alpha=4
        )

        prob = om.Problem()
        group = builder.build_pre_mission(aviary_inputs=None)
        prob.model.add_subsystem('datcom', group)
        prob.setup()

        prob['datcom.mach'] = 2.36
        prob['datcom.alpha'] = [0, 4, 8, 12]
        prob['datcom.reynolds'] = 3e6
        prob['datcom.xcg'] = 18.75

        prob.run_model()

        cn = prob['datcom.CN']
        assert len(cn) == 4
        assert cn[1] > cn[0]  # CN increases with alpha
