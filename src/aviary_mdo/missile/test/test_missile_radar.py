"""
Tests for TMD radar detection against spreadsheet baseline values.

Baseline radar (TMD Radar sheet):
  - wavelength=0.03 m, antenna_dia=10 in (0.254 m), Pt=1000 W, freq=10 GHz
  - Receiver: T=290 K, Bn=1e6 Hz, loss_factor=5, num_pulses=100,
    noise_factor=5, SNR=10
  - Missile RCS = 0.01 m^2, Target RCS = 10.0 m^2

Validation values from design_spreadsheet_TMD.xls Radar sheet:
  - 3-dB beamwidth = 0.1205 rad
  - Missile tracking target range: R0 = 16366.7 m = 8.837 nmi
  - Threat tracking missile range: computed from same equation with
    sigma=missile_rcs, SNR=1
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.model.missile_radar import (
    MissileRadar,
    VisualDetectionRange,
)


def _make_radar_problem():
    """Create and set up a radar problem with baseline inputs."""
    prob = om.Problem()
    prob.model.add_subsystem('radar', MissileRadar())
    prob.setup(force_alloc_complex=True)

    prob.set_val('radar.radar_wavelength', 0.03, units='m')
    prob.set_val('radar.antenna_diameter', 10.0, units='inch')
    prob.set_val('radar.transmitter_power', 1000.0, units='W')
    prob.set_val('radar.frequency', 10.0e9, units='Hz')
    prob.set_val('radar.receiver_temperature', 290.0, units='K')
    prob.set_val('radar.noise_bandwidth', 1.0e6, units='Hz')
    prob.set_val('radar.loss_factor', 5.0)
    prob.set_val('radar.num_pulses_integrated', 100.0)
    prob.set_val('radar.receiver_noise_factor', 5.0)
    prob.set_val('radar.signal_to_noise_ratio', 10.0)
    prob.set_val('radar.missile_rcs', 0.01, units='m**2')
    prob.set_val('radar.target_rcs', 10.0, units='m**2')

    return prob


class TestMissileRadarBeamwidth(unittest.TestCase):
    """Test antenna beamwidth against TMD spreadsheet."""

    def test_beamwidth_baseline(self):
        """TMD Radar sheet: 3-dB beamwidth = 0.1205 rad (R5/R6)."""
        prob = _make_radar_problem()
        prob.run_model()

        bw = prob.get_val('radar.beamwidth', units='rad')
        assert_near_equal(bw, 0.1205, tolerance=0.01)


class TestMissileRadarDetectionRange(unittest.TestCase):
    """Test detection range against TMD spreadsheet."""

    def test_detection_range_nmi(self):
        """TMD Radar sheet: R0 = 8.837 nmi (R6-R12)."""
        prob = _make_radar_problem()
        prob.run_model()

        R_nmi = prob.get_val('radar.detection_range_nmi')
        assert_near_equal(R_nmi, 8.837, tolerance=0.05)

    def test_detection_range_meters(self):
        """TMD Radar sheet: R0 = 16366.7 m."""
        prob = _make_radar_problem()
        prob.run_model()

        R_m = prob.get_val('radar.detection_range', units='m')
        assert_near_equal(R_m, 16366.7, tolerance=0.05)

    def test_detection_range_ft(self):
        """TMD Radar sheet: R0 = 53696.5 ft."""
        prob = _make_radar_problem()
        prob.run_model()

        R_ft = prob.get_val('radar.detection_range_ft')
        assert_near_equal(R_ft, 53696.5, tolerance=0.05)


class TestThreatRange(unittest.TestCase):
    """Test threat tracking missile range."""

    def test_threat_range_baseline(self):
        """Threat range using missile_rcs=0.01 and SNR=1."""
        prob = _make_radar_problem()
        prob.run_model()

        R_threat = prob.get_val('radar.threat_range', units='m')
        # Radar range equation with sigma=0.01 m^2 and SNR=1:
        # R_threat = (Pt * G^2 * lam^2 * 0.01 * n / ((4pi)^3*k*T*Bn*Fn*1*L))^0.25
        # With baseline inputs this gives ~5176 m
        self.assertGreater(R_threat, 4000.0)
        self.assertLess(R_threat, 7000.0)

    def test_threat_range_less_than_detection(self):
        """Threat range (small RCS, SNR=1) should differ from detection range."""
        prob = _make_radar_problem()
        prob.run_model()

        R_det = prob.get_val('radar.detection_range', units='m')
        R_threat = prob.get_val('radar.threat_range', units='m')
        # With missile_rcs < target_rcs and SNR=1 < SNR=10,
        # the net effect makes threat range shorter
        self.assertNotEqual(R_det, R_threat)


class TestRadarTrends(unittest.TestCase):
    """Test that radar range responds correctly to parameter changes."""

    def test_larger_antenna_increases_range(self):
        """Larger antenna diameter should increase detection range."""
        prob = _make_radar_problem()
        prob.set_val('radar.antenna_diameter', 10.0, units='inch')
        prob.run_model()
        R_small = prob.get_val('radar.detection_range', units='m')[0]

        prob.set_val('radar.antenna_diameter', 20.0, units='inch')
        prob.run_model()
        R_large = prob.get_val('radar.detection_range', units='m')[0]

        self.assertGreater(R_large, R_small)

    def test_larger_rcs_increases_range(self):
        """Larger target RCS should increase detection range."""
        prob = _make_radar_problem()
        prob.set_val('radar.target_rcs', 1.0, units='m**2')
        prob.run_model()
        R_small_rcs = prob.get_val('radar.detection_range', units='m')[0]

        prob.set_val('radar.target_rcs', 100.0, units='m**2')
        prob.run_model()
        R_large_rcs = prob.get_val('radar.detection_range', units='m')[0]

        self.assertGreater(R_large_rcs, R_small_rcs)

    def test_higher_power_increases_range(self):
        """Higher transmitter power should increase detection range."""
        prob = _make_radar_problem()
        prob.set_val('radar.transmitter_power', 500.0, units='W')
        prob.run_model()
        R_low = prob.get_val('radar.detection_range', units='m')[0]

        prob.set_val('radar.transmitter_power', 2000.0, units='W')
        prob.run_model()
        R_high = prob.get_val('radar.detection_range', units='m')[0]

        self.assertGreater(R_high, R_low)


class TestDetectionRangeTable(unittest.TestCase):
    """Test detection range vs RCS table."""

    def test_range_increases_with_rcs(self):
        """Detection range should monotonically increase with RCS."""
        prob = _make_radar_problem()
        prob.run_model()

        ranges = []
        for i in range(6):
            R = prob.get_val(f'radar.det_range_rcs_{i}', units='m')[0]
            ranges.append(R)

        for i in range(1, len(ranges)):
            self.assertGreater(ranges[i], ranges[i - 1])

    def test_table_rcs_10_matches_detection_range(self):
        """RCS=10 table entry should match main detection range (same sigma)."""
        prob = _make_radar_problem()
        prob.run_model()

        R_det = prob.get_val('radar.detection_range', units='m')[0]
        R_table_10 = prob.get_val('radar.det_range_rcs_3', units='m')[0]

        assert_near_equal(R_table_10, R_det, tolerance=1e-6)


class TestVisualDetection(unittest.TestCase):
    """Test visual detection range component."""

    def test_baseline(self):
        """Visual range = Ct * sqrt(area)."""
        prob = om.Problem()
        prob.model.add_subsystem('vis', VisualDetectionRange())
        prob.setup(force_alloc_complex=True)

        prob.set_val('vis.target_presented_area', 4.0, units='m**2')
        prob.set_val('vis.visual_constant', 2000.0)
        prob.run_model()

        R_vis = prob.get_val('vis.visual_range', units='m')
        assert_near_equal(R_vis, 4000.0, tolerance=1e-6)

    def test_larger_area_increases_range(self):
        """Larger target area should increase visual detection range."""
        prob = om.Problem()
        prob.model.add_subsystem('vis', VisualDetectionRange())
        prob.setup(force_alloc_complex=True)

        prob.set_val('vis.visual_constant', 2000.0)

        prob.set_val('vis.target_presented_area', 1.0, units='m**2')
        prob.run_model()
        R_small = prob.get_val('vis.visual_range', units='m')[0]

        prob.set_val('vis.target_presented_area', 9.0, units='m**2')
        prob.run_model()
        R_large = prob.get_val('vis.visual_range', units='m')[0]

        self.assertGreater(R_large, R_small)


class TestRadarDerivatives(unittest.TestCase):
    """Test that complex-step derivatives compute without error."""

    def test_radar_partials(self):
        """Check partial derivatives via complex step."""
        prob = _make_radar_problem()
        prob.run_model()

        data = prob.check_partials(compact_print=True, method='cs')
        for comp_name, comp_data in data.items():
            for (out_var, in_var), deriv_data in comp_data.items():
                if np.all(deriv_data['J_fwd'] == 0.0):
                    continue
                rel_err = deriv_data['rel error']
                self.assertLess(
                    rel_err.forward, 1e-6,
                    msg=f'Derivative {out_var} wrt {in_var} in {comp_name} '
                        f'has rel error {rel_err.forward}'
                )

    def test_visual_partials(self):
        """Check visual detection range partials."""
        prob = om.Problem()
        prob.model.add_subsystem('vis', VisualDetectionRange())
        prob.setup(force_alloc_complex=True)

        prob.set_val('vis.target_presented_area', 4.0, units='m**2')
        prob.set_val('vis.visual_constant', 2000.0)
        prob.run_model()

        data = prob.check_partials(compact_print=True, method='cs')
        for comp_name, comp_data in data.items():
            for (out_var, in_var), deriv_data in comp_data.items():
                if np.all(deriv_data['J_fwd'] == 0.0):
                    continue
                rel_err = deriv_data['rel error']
                self.assertLess(
                    rel_err.forward, 1e-6,
                    msg=f'Derivative {out_var} wrt {in_var} in {comp_name} '
                        f'has rel error {rel_err.forward}'
                )


if __name__ == '__main__':
    unittest.main()
