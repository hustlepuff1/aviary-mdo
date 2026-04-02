"""
Integration tests for MissileBuilder.

Tests the complete missile sizing pipeline:
1. Builder instantiation
2. Pre-mission system build and run
3. Baseline rocket case validation against TMD spreadsheet
4. Design variable sweeps
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary_mdo.missile.missile_builder import MissileBuilder


class TestMissileBuilderAPI(unittest.TestCase):
    """Test the SubsystemBuilder API methods."""

    def test_instantiation(self):
        builder = MissileBuilder()
        self.assertEqual(builder.name, 'missile')

    def test_custom_name(self):
        builder = MissileBuilder(name='sparrow')
        self.assertEqual(builder.name, 'sparrow')

    def test_design_vars(self):
        builder = MissileBuilder()
        dvs = builder.get_design_vars()
        self.assertIn('missile:body:length', dvs)
        self.assertIn('missile:body:diameter_major', dvs)
        self.assertIn('missile:propulsion:boost_expansion_ratio', dvs)

    def test_mass_names(self):
        builder = MissileBuilder()
        masses = builder.get_mass_names()
        self.assertIn('missile:weight:launch', masses)

    def test_parameters(self):
        builder = MissileBuilder()
        params = builder.get_parameters()
        self.assertIn('missile:flight:launch_mach', params)
        self.assertIn('missile:flight:launch_altitude', params)

    def test_build_pre_mission_returns_group(self):
        builder = MissileBuilder()
        sys = builder.build_pre_mission(aviary_inputs=None)
        self.assertIsInstance(sys, om.Group)

    def test_build_mission_returns_none(self):
        """Missile sizing is pre-mission only (no dynamic mission states)."""
        builder = MissileBuilder()
        sys = builder.build_mission(num_nodes=10, aviary_inputs=None)
        self.assertIsNone(sys)

    def test_get_states_empty(self):
        """No mission states for the missile subsystem."""
        builder = MissileBuilder()
        states = builder.get_states()
        self.assertEqual(states, {})


class TestMissilePreMissionRun(unittest.TestCase):
    """Test the full pre-mission system runs with baseline inputs."""

    def test_rocket_baseline(self):
        """Run the full missile sizing with Sparrow MRAAM baseline."""
        builder = MissileBuilder(sustain_engine='rocket')
        pre_mission = builder.build_pre_mission(aviary_inputs=None)

        prob = om.Problem()
        prob.model.add_subsystem('missile', pre_mission)
        prob.setup()

        # Set baseline rocket inputs (Sparrow MRAAM from TMD spreadsheet)
        prob.set_val('missile.launch_mach', 0.8)
        prob.set_val('missile.launch_alt', 20000.0, units='ft')
        prob.set_val('missile.missile_length', 143.9, units='inch')
        prob.set_val('missile.missile_diameter', 8.0, units='inch')
        prob.set_val('missile.nose_length', 19.2, units='inch')
        prob.set_val('missile.ref_area', 50.27, units='inch**2')
        prob.set_val('missile.cg_station', 76.2, units='inch')

        # Wing
        prob.set_val('missile.num_wings', 2.0)
        prob.set_val('missile.wing_area', 400.0, units='inch**2')
        prob.set_val('missile.wing_AR', 2.82)
        prob.set_val('missile.wing_taper', 0.175)
        prob.set_val('missile.wing_le_station', 60.8, units='inch')

        # Tail
        prob.set_val('missile.num_tails', 2.0)
        prob.set_val('missile.tail_area', 87.0, units='inch**2')
        prob.set_val('missile.tail_AR', 2.59)
        prob.set_val('missile.tail_taper', 0.0)
        prob.set_val('missile.tail_le_station', 125.4, units='inch')

        # Propulsion
        prob.set_val('missile.boost_Pc', 1769.0, units='lbf/inch**2')
        prob.set_val('missile.boost_expansion_ratio', 6.0)
        prob.set_val('missile.boost_fuel_type', 4.0)
        prob.set_val('missile.W_boost_prop', 84.8, units='lbm')
        prob.set_val('missile.boost_burn_time', 3.26, units='s')
        prob.set_val('missile.Pa', 6.76, units='lbf/inch**2')

        prob.set_val('missile.sustain_Pc', 301.0, units='lbf/inch**2')
        prob.set_val('missile.sustain_expansion_ratio', 6.2)
        prob.set_val('missile.sustain_fuel_type', 4.0)
        prob.set_val('missile.W_sustain_prop', 48.2, units='lbm')
        prob.set_val('missile.sustain_burn_time', 10.86, units='s')

        # Trajectory inputs
        prob.set_val('missile.trajectory.boost.W_launch', 500.0, units='lbm')
        prob.set_val('missile.trajectory.sustain.W_ejectables', 0.0, units='lbm')
        prob.set_val('missile.trajectory.sustain.CD0', 0.42)
        prob.set_val('missile.trajectory.coast.mach_terminal', 1.5)
        prob.set_val('missile.trajectory.coast.CD0', 0.46)

        prob.run_model()

        # Check key outputs
        range_total = prob.get_val('missile.range_total', units='nmi')[0]
        time_total = prob.get_val('missile.time_total', units='s')[0]
        CN = prob.get_val('missile.CN_total')[0]
        Xcp = prob.get_val('missile.Xcp', units='inch')[0]

        # Boost motor outputs
        boost_Isp = prob.get_val('missile.boost_motor.Isp', units='s')[0]
        boost_thrust = prob.get_val('missile.boost_motor.thrust', units='lbf')[0]

        print(f"\n{'='*50}")
        print(f"TMD MISSILE SIZING — SPARROW BASELINE")
        print(f"{'='*50}")
        print(f"  Boost Isp:      {boost_Isp:.1f} s")
        print(f"  Boost Thrust:   {boost_thrust:.0f} lbf")
        print(f"  Total Range:    {range_total:.2f} nmi (TMD: 10.71)")
        print(f"  Total Time:     {time_total:.1f} s (TMD: 30.9)")
        print(f"  CN (10 deg):    {CN:.3f}")
        print(f"  Xcp:            {Xcp:.1f} in")
        print(f"{'='*50}")

        # Range should be positive and in a reasonable ballpark
        self.assertGreater(range_total, 3.0)
        self.assertLess(range_total, 25.0)

        # Time should be positive and reasonable
        self.assertGreater(time_total, 10.0)
        self.assertLess(time_total, 80.0)

        # Isp should match TMD range
        self.assertGreater(boost_Isp, 200.0)
        self.assertLess(boost_Isp, 350.0)

    def test_diameter_increases_drag(self):
        """Larger diameter should generally increase drag and reduce range."""
        builder = MissileBuilder()

        ranges = []
        for diameter in [6.0, 8.0, 12.0]:
            pre_mission = builder.build_pre_mission(aviary_inputs=None)
            prob = om.Problem()
            prob.model.add_subsystem('m', pre_mission)
            prob.setup()

            prob.set_val('m.missile_diameter', diameter, units='inch')
            prob.set_val('m.trajectory.boost.W_launch', 500.0, units='lbm')
            prob.set_val('m.trajectory.sustain.W_ejectables', 0.0, units='lbm')
            prob.set_val('m.trajectory.sustain.CD0', 0.42)
            prob.set_val('m.trajectory.coast.mach_terminal', 1.5)
            prob.set_val('m.trajectory.coast.CD0', 0.46)

            prob.run_model()
            R = prob.get_val('m.range_total', units='nmi')[0]
            ranges.append(R)

        # Smaller diameter -> longer range (Fleeman key finding)
        self.assertGreater(ranges[0], ranges[2],
                           msg="Smaller diameter should give longer range")


if __name__ == '__main__':
    unittest.main()
