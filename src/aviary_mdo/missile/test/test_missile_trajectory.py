"""
Tests for TMD trajectory against spreadsheet baseline values.

Baseline rocket trajectory:
  - Launch: M=0.8, h=20000 ft, W=500 lbm
  - End boost: M~2.34, t=3.26s, range=0.87 nmi, W=415.2 lbm
  - End sustain: M~2.43, t=14.12s, range=5.29 nmi, W=367.0 lbm
  - End coast (co-alt): M=1.5, t=30.93s, range=10.71 nmi

Values from design_spreadsheet_TMD.xls Summary sheet.
"""

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.model.missile_trajectory import (
    BoostPhase,
    CoastPhase,
    MissileTrajectoryGroup,
)


class TestBoostPhase(unittest.TestCase):
    """Test boost phase against TMD baseline."""

    def test_rocket_baseline(self):
        prob = om.Problem()
        prob.model.add_subsystem('boost', BoostPhase())
        prob.setup(force_alloc_complex=True)

        prob.set_val('boost.mach_launch', 0.8)
        prob.set_val('boost.alt', 20000.0, units='ft')
        prob.set_val('boost.W_launch', 500.0, units='lbm')
        prob.set_val('boost.W_prop_boost', 84.8, units='lbm')
        prob.set_val('boost.thrust_boost', 7036.0, units='lbf')
        prob.set_val('boost.Isp_boost', 271.0, units='s')
        prob.set_val('boost.burn_time', 3.26, units='s')
        prob.set_val('boost.CD0_launch', 0.31)
        prob.set_val('boost.ref_area', 50.27, units='inch**2')

        prob.run_model()

        V_launch = prob.get_val('boost.V_launch')
        V_end = prob.get_val('boost.V_boost_end')
        M_end = prob.get_val('boost.mach_boost_end')
        W_end = prob.get_val('boost.W_boost_end')
        R = prob.get_val('boost.range_boost')

        # Launch velocity: 0.8 * 1036.9 = 829.5 ft/s
        assert_near_equal(V_launch, 829.5, tolerance=0.01)

        # Weight at end of boost
        assert_near_equal(W_end, 415.2, tolerance=1e-6)

        # End-of-boost Mach should be roughly 2.0-2.5
        self.assertGreater(M_end, 1.8)
        self.assertLess(M_end, 2.8)

        # Range should be roughly 0.5-1.5 nmi
        self.assertGreater(R, 0.3)
        self.assertLess(R, 2.0)


class TestCoastPhase(unittest.TestCase):
    """Test coast phase deceleration model."""

    def test_coast_decelerates(self):
        """Missile should decelerate during coast."""
        prob = om.Problem()
        prob.model.add_subsystem('coast', CoastPhase())
        prob.setup(force_alloc_complex=True)

        prob.set_val('coast.mach_start', 2.4)
        prob.set_val('coast.mach_terminal', 1.5)
        prob.set_val('coast.alt', 20000.0, units='ft')
        prob.set_val('coast.W_coast', 367.0, units='lbm')
        prob.set_val('coast.CD0', 0.46)
        prob.set_val('coast.ref_area', 50.27, units='inch**2')

        prob.run_model()

        t_coast = prob.get_val('coast.time_coast')
        R_coast = prob.get_val('coast.range_coast')
        V_end = prob.get_val('coast.V_coast_end')

        # Coast time should be positive
        self.assertGreater(t_coast, 0.0)
        # Coast range should be positive
        self.assertGreater(R_coast, 0.0)
        # End velocity should be at terminal Mach
        a = 1036.9  # speed of sound at 20,000 ft
        assert_near_equal(V_end, 1.5 * a, tolerance=0.01)


class TestFullTrajectory(unittest.TestCase):
    """Test assembled three-phase trajectory."""

    def test_trajectory_runs(self):
        """Verify the full trajectory group assembles and runs."""
        prob = om.Problem()
        prob.model.add_subsystem('traj',
                                MissileTrajectoryGroup(sustain_engine='rocket'))
        prob.setup(force_alloc_complex=True)

        # Boost inputs
        prob.set_val('traj.boost.mach_launch', 0.8)
        prob.set_val('traj.alt', 20000.0, units='ft')
        prob.set_val('traj.boost.W_launch', 500.0, units='lbm')
        prob.set_val('traj.boost.W_prop_boost', 84.8, units='lbm')
        prob.set_val('traj.boost.thrust_boost', 7036.0, units='lbf')
        prob.set_val('traj.boost.Isp_boost', 271.0, units='s')
        prob.set_val('traj.boost.burn_time', 3.26, units='s')
        prob.set_val('traj.boost.CD0_launch', 0.31)
        prob.set_val('traj.ref_area', 50.27, units='inch**2')

        # Sustain inputs
        prob.set_val('traj.sustain.W_ejectables', 0.0, units='lbm')
        prob.set_val('traj.sustain.W_prop_sustain', 48.2, units='lbm')
        prob.set_val('traj.sustain.thrust_sustain', 1119.0, units='lbf')
        prob.set_val('traj.sustain.Isp_sustain', 252.0, units='s')
        prob.set_val('traj.sustain.burn_time', 10.86, units='s')
        prob.set_val('traj.sustain.CD0', 0.42)

        # Coast inputs
        prob.set_val('traj.coast.mach_terminal', 1.5)
        prob.set_val('traj.coast.CD0', 0.46)

        prob.run_model()

        R_total = prob.get_val('traj.range_total', units='nmi')
        t_total = prob.get_val('traj.time_total', units='s')

        # TMD spreadsheet: total range ~10.7 nmi, time ~31 s
        # Our simplified model won't match exactly, but should be in range
        self.assertGreater(R_total, 5.0)
        self.assertLess(R_total, 20.0)
        self.assertGreater(t_total, 15.0)
        self.assertLess(t_total, 60.0)

        print(f"\nTrajectory results:")
        print(f"  Total range: {R_total[0]:.2f} nmi (TMD baseline: 10.71 nmi)")
        print(f"  Total time:  {t_total[0]:.1f} s (TMD baseline: 30.9 s)")


if __name__ == '__main__':
    unittest.main()
