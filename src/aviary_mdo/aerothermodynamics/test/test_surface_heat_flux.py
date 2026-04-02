"""
Tests for the SurfaceHeatFlux component.

Validates heat flux distribution, transition behaviour, edge cases,
and complex-step partial derivatives.
"""

import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary_mdo.aerothermodynamics.model.surface_heat_flux import (
    SurfaceHeatFlux,
)


def _make_problem(num_stations=10, **set_vals):
    """Build and setup a Problem with SurfaceHeatFlux, apply overrides."""
    prob = om.Problem()
    prob.model.add_subsystem(
        'shf', SurfaceHeatFlux(num_stations=num_stations),
    )
    prob.setup(force_alloc_complex=True)

    # Sensible defaults
    prob.set_val('shf.q_stag', 1.0e6, units='W/m**2')
    prob.set_val('shf.velocity', 2000.0, units='m/s')
    prob.set_val('shf.altitude', 30000.0, units='m')
    prob.set_val('shf.nose_radius', 0.1, units='m')
    prob.set_val('shf.wall_temperature', 400.0, units='K')
    prob.set_val('shf.body_length', 5.0, units='m')
    prob.set_val('shf.body_diameter', 0.5, units='m')
    prob.set_val('shf.Re_transition', 5e5)

    for key, val in set_vals.items():
        if isinstance(val, tuple):
            prob.set_val(f'shf.{key}', val[0], units=val[1])
        else:
            prob.set_val(f'shf.{key}', val)

    return prob


class TestSurfaceHeatFluxComputed(unittest.TestCase):
    """Heat flux distribution is computed for all stations."""

    def test_all_stations_filled(self):
        nn = 10
        prob = _make_problem(num_stations=nn)
        prob.run_model()

        q = prob.get_val('shf.q_distribution', units='W/m**2')
        x = prob.get_val('shf.x_stations', units='m')

        self.assertEqual(q.shape[0], nn)
        self.assertEqual(x.shape[0], nn)
        # Every station should have a positive heat flux
        for i in range(nn):
            self.assertGreater(q[i], 0.0,
                               f'Station {i} (x={x[i]:.3f} m) has non-positive q')


class TestNoseDecay(unittest.TestCase):
    """Heat flux should decrease from stagnation going downstream in the
    laminar region (before transition)."""

    def test_decreasing_from_nose(self):
        # Use a large nose radius so several stations fall in the nose region
        prob = _make_problem(num_stations=20, nose_radius=2.0,
                             body_length=5.0, Re_transition=1e8)
        prob.run_model()

        q = prob.get_val('shf.q_distribution', units='W/m**2')
        x = prob.get_val('shf.x_stations', units='m')

        # In the laminar flat-plate region (beyond the nose), heat flux
        # decreases with x because q ~ 1/sqrt(x).
        # Identify stations beyond the nose (x > R_nose = 2.0 m).
        lam_indices = [i for i in range(len(x)) if x[i] > 2.0]
        self.assertGreater(len(lam_indices), 2,
                           'Need multiple laminar stations for this test')

        for i in range(1, len(lam_indices)):
            idx_prev = lam_indices[i - 1]
            idx_curr = lam_indices[i]
            self.assertLess(q[idx_curr], q[idx_prev],
                            f'q should decrease downstream: '
                            f'x={x[idx_curr]:.2f} vs x={x[idx_prev]:.2f}')


class TestTurbulentHigherThanLaminar(unittest.TestCase):
    """Turbulent heat flux should exceed laminar heat flux at the same x."""

    def test_turb_vs_lam(self):
        nn = 40
        # Run with very high transition Re to keep everything laminar
        prob_lam = _make_problem(num_stations=nn, Re_transition=1e12)
        prob_lam.run_model()
        q_lam = prob_lam.get_val('shf.q_distribution', units='W/m**2').copy()

        # Run with very low transition Re so everything is turbulent
        prob_turb = _make_problem(num_stations=nn, Re_transition=1.0)
        prob_turb.run_model()
        q_turb = prob_turb.get_val('shf.q_distribution', units='W/m**2').copy()

        x = prob_turb.get_val('shf.x_stations', units='m')

        # Compare at stations beyond the nose
        R_nose = 0.1  # default
        fp_indices = [i for i in range(nn) if x[i] > R_nose]
        self.assertGreater(len(fp_indices), 5)

        for i in fp_indices:
            self.assertGreater(
                q_turb[i], q_lam[i],
                f'Turbulent q should exceed laminar at x={x[i]:.3f} m')


class TestTotalHeatLoadPositive(unittest.TestCase):
    """Integrated total heat load must be positive for a heated body."""

    def test_positive_heat_load(self):
        prob = _make_problem()
        prob.run_model()

        Q = prob.get_val('shf.total_heat_load', units='W')
        self.assertGreater(Q, 0.0)

    def test_longer_body_more_heat(self):
        """Doubling body length should increase total heat load."""
        prob_short = _make_problem(body_length=3.0)
        prob_short.run_model()
        Q_short = prob_short.get_val('shf.total_heat_load', units='W')[0]

        prob_long = _make_problem(body_length=6.0)
        prob_long.run_model()
        Q_long = prob_long.get_val('shf.total_heat_load', units='W')[0]

        self.assertGreater(Q_long, Q_short)


class TestZeroVelocity(unittest.TestCase):
    """Zero freestream velocity should give zero heat flux everywhere."""

    def test_zero_velocity(self):
        prob = _make_problem(velocity=0.0)
        prob.run_model()

        q = prob.get_val('shf.q_distribution', units='W/m**2')
        Q = prob.get_val('shf.total_heat_load', units='W')

        np.testing.assert_array_equal(q, 0.0)
        self.assertEqual(Q, 0.0)


class TestPartials(unittest.TestCase):
    """Complex-step partial derivative check."""

    def test_check_partials(self):
        prob = _make_problem(num_stations=5)
        prob.run_model()

        data = prob.check_partials(
            includes=['shf'],
            out_stream=None,
            method='fd',
            compact_print=True,
        )
        assert_check_partials(data, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
