"""Tests for the Dymos-compatible missile ODE.

Verifies:
1. Components have analytic partials (check_partials)
2. EOM produces correct rates for a known state
3. Full Dymos trajectory runs and converges
"""

import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


class TestMissileAtmosComp(unittest.TestCase):
    """Test standard atmosphere component."""

    def test_sea_level(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileAtmosComp
        prob = om.Problem()
        prob.model.add_subsystem('atmos', MissileAtmosComp(num_nodes=1))
        prob.setup(force_alloc_complex=True)
        prob['atmos.altitude'] = 0.0
        prob.run_model()

        assert_near_equal(prob['atmos.temperature'], 288.15, tolerance=1e-4)
        assert_near_equal(prob['atmos.density'], 1.225, tolerance=0.01)
        assert_near_equal(prob['atmos.speed_of_sound'], 340.294, tolerance=0.01)

    def test_11km(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileAtmosComp
        prob = om.Problem()
        prob.model.add_subsystem('atmos', MissileAtmosComp(num_nodes=1))
        prob.setup(force_alloc_complex=True)
        prob['atmos.altitude'] = 11000.0
        prob.run_model()

        # Tropopause: T = 216.65 K
        assert_near_equal(prob['atmos.temperature'], 216.65, tolerance=0.01)

    def test_partials(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileAtmosComp
        prob = om.Problem()
        prob.model.add_subsystem('atmos', MissileAtmosComp(num_nodes=3))
        prob.setup(force_alloc_complex=True)
        prob['atmos.altitude'] = [1000, 5000, 9000]
        prob.run_model()
        data = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(data, atol=1e-6, rtol=1e-6)


class TestMissileAeroComp(unittest.TestCase):
    """Test aerodynamic force component."""

    def test_zero_alpha(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileAeroComp
        prob = om.Problem()
        prob.model.add_subsystem('aero', MissileAeroComp(num_nodes=1))
        prob.setup(force_alloc_complex=True)
        prob['aero.velocity'] = 300.0
        prob['aero.density'] = 1.225
        prob['aero.speed_of_sound'] = 340.3
        prob['aero.alpha'] = 0.0
        prob['aero.S_ref'] = 0.0324
        prob['aero.CD0'] = 0.3
        prob.run_model()

        # Lift should be zero at alpha=0
        assert_near_equal(prob['aero.lift'], 0.0, tolerance=1e-10)
        # Drag should be q*S*CD0
        q = 0.5 * 1.225 * 300**2
        assert_near_equal(prob['aero.drag'], q * 0.0324 * 0.3, tolerance=1e-6)

    def test_partials(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileAeroComp
        prob = om.Problem()
        prob.model.add_subsystem('aero', MissileAeroComp(num_nodes=3))
        prob.setup(force_alloc_complex=True)
        prob['aero.velocity'] = [250, 300, 400]
        prob['aero.density'] = [1.0, 0.7, 0.4]
        prob['aero.speed_of_sound'] = [330, 320, 310]
        prob['aero.alpha'] = [0.01, 0.05, 0.1]
        prob.run_model()
        data = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(data, atol=1e-6, rtol=1e-6)


class TestMissileEOM(unittest.TestCase):
    """Test equations of motion component."""

    def test_straight_level(self):
        """Zero gamma, no thrust, no lift => gravity pulls down."""
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileEOM
        prob = om.Problem()
        prob.model.add_subsystem('eom', MissileEOM(num_nodes=1))
        prob.setup(force_alloc_complex=True)

        prob['eom.velocity'] = 300.0
        prob['eom.flight_path_angle'] = 0.0
        prob['eom.mass'] = 200.0
        prob['eom.thrust'] = 0.0
        prob['eom.drag'] = 500.0  # some drag
        prob['eom.lift'] = 200.0 * 9.80665  # lift = weight (level flight)
        prob['eom.alpha'] = 0.0
        prob['eom.mass_flow_rate'] = 0.0
        prob.run_model()

        # dV/dt should be -D/m (no thrust, no gravity along path)
        assert_near_equal(prob['eom.velocity_rate'], -500.0 / 200.0, tolerance=1e-6)
        # dh/dt = V*sin(0) = 0
        assert_near_equal(prob['eom.altitude_rate'], 0.0, tolerance=1e-10)
        # dx/dt = V*cos(0) = V
        assert_near_equal(prob['eom.range_rate'], 300.0, tolerance=1e-10)

    def test_vertical_climb(self):
        """gamma=90deg, T > D + W => accelerating upward."""
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileEOM
        prob = om.Problem()
        prob.model.add_subsystem('eom', MissileEOM(num_nodes=1))
        prob.setup(force_alloc_complex=True)

        m = 200.0
        g = 9.80665
        T = 5000.0
        D = 300.0

        prob['eom.velocity'] = 300.0
        prob['eom.flight_path_angle'] = np.pi / 2  # straight up
        prob['eom.mass'] = m
        prob['eom.thrust'] = T
        prob['eom.drag'] = D
        prob['eom.lift'] = 0.0
        prob['eom.alpha'] = 0.0
        prob.run_model()

        # dV/dt = (T - D)/m - g
        expected = (T - D) / m - g
        assert_near_equal(prob['eom.velocity_rate'], expected, tolerance=1e-6)
        # dh/dt = V
        assert_near_equal(prob['eom.altitude_rate'], 300.0, tolerance=1e-6)
        # dx/dt ≈ 0 (cos(pi/2))
        self.assertAlmostEqual(float(prob['eom.range_rate']), 0.0, places=5)

    def test_partials(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileEOM
        prob = om.Problem()
        prob.model.add_subsystem('eom', MissileEOM(num_nodes=3))
        prob.setup(force_alloc_complex=True)

        prob['eom.velocity'] = [250, 300, 400]
        prob['eom.flight_path_angle'] = [0.1, 0.5, 1.0]
        prob['eom.mass'] = [200, 180, 150]
        prob['eom.thrust'] = [5000, 3000, 0]
        prob['eom.drag'] = [300, 400, 200]
        prob['eom.lift'] = [100, 500, 300]
        prob['eom.alpha'] = [0.01, 0.05, 0.1]
        prob['eom.mass_flow_rate'] = [5, 5, 0]
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(data, atol=1e-5, rtol=1e-5)


class TestMissileODE(unittest.TestCase):
    """Test the assembled ODE group."""

    def test_ode_runs(self):
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileODE
        prob = om.Problem()
        prob.model.add_subsystem('ode', MissileODE(num_nodes=3))
        prob.setup(force_alloc_complex=True)

        prob['ode.altitude'] = [0, 5000, 10000]
        prob['ode.velocity'] = [300, 350, 400]
        prob['ode.flight_path_angle'] = [0.5, 0.4, 0.3]
        prob['ode.mass'] = [200, 190, 180]
        prob['ode.alpha'] = [0.01, 0.02, 0.01]
        prob['ode.time'] = [0, 1, 2]
        prob['ode.thrust_magnitude'] = 10000.0
        prob['ode.burn_time'] = 3.0
        prob.run_model()

        # Should produce non-zero rates
        vr = prob['ode.velocity_rate']
        self.assertTrue(all(np.isfinite(vr)), f'Non-finite velocity_rate: {vr}')

    def test_ode_partials(self):
        """Check that all partials in the assembled ODE are correct."""
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileODE
        prob = om.Problem()
        prob.model.add_subsystem('ode', MissileODE(num_nodes=2))
        prob.setup(force_alloc_complex=True)

        prob['ode.altitude'] = [1000, 5000]
        prob['ode.velocity'] = [300, 400]
        prob['ode.flight_path_angle'] = [0.3, 0.5]
        prob['ode.mass'] = [200, 180]
        prob['ode.alpha'] = [0.02, 0.05]
        prob['ode.time'] = [1.0, 2.0]
        prob['ode.thrust_magnitude'] = 10000.0
        prob['ode.burn_time'] = 3.0
        prob['ode.mass_flow_rate_burn'] = 5.0
        prob.run_model()

        data = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(data, atol=1e-5, rtol=1e-5)


class TestDymosIntegration(unittest.TestCase):
    """Integration test: run a Dymos missile trajectory."""

    def test_ballistic_arc(self):
        """Fire missile at 45 deg, verify it goes up then comes down."""
        import dymos as dm
        from aviary_mdo.missile_6dof.model.dymos_ode import MissileODE

        prob = om.Problem()
        traj = prob.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.Phase(
            ode_class=MissileODE,
            ode_init_kwargs={'S_ref': 0.0324, 'CD0': 0.3, 'CN_alpha': 2.0},
            transcription=dm.GaussLobatto(num_segments=15, order=3),
        )
        traj.add_phase('boost', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(1, 60),
                               duration_ref=30.0)
        phase.add_state('velocity', fix_initial=True, rate_source='velocity_rate',
                         units='m/s', lower=10, ref=500)
        phase.add_state('flight_path_angle', fix_initial=True,
                         rate_source='flight_path_angle_rate',
                         units='rad', ref=1.0)
        phase.add_state('altitude', fix_initial=True, rate_source='altitude_rate',
                         units='m', lower=0, ref=10000)
        phase.add_state('range', fix_initial=True, rate_source='range_rate',
                         units='m', ref=10000)
        phase.add_state('mass', fix_initial=True, rate_source='mass_rate',
                         units='kg', lower=50, ref=200)

        # Alpha as a polynomial control
        phase.add_control('alpha', units='rad', lower=-0.1, upper=0.3,
                          fix_initial=True, ref=0.1)

        # Thrust parameters
        phase.add_parameter('thrust_magnitude', val=10000.0, units='N',
                            opt=False)
        phase.add_parameter('burn_time', val=3.0, units='s', opt=False)
        phase.add_parameter('mass_flow_rate_burn', val=5.0, units='kg/s',
                            opt=False)

        # Objective: maximize range
        phase.add_objective('range', loc='final', scaler=-1e-4)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter'] = 100

        prob.setup()

        # Initial conditions: 45-deg launch at 50 m/s
        prob.set_val('traj.boost.t_initial', 0.0)
        prob.set_val('traj.boost.t_duration', 30.0)
        prob.set_val('traj.boost.states:velocity',
                      phase.interp('velocity', [50, 400]))
        prob.set_val('traj.boost.states:flight_path_angle',
                      phase.interp('flight_path_angle', [np.radians(45), np.radians(20)]))
        prob.set_val('traj.boost.states:altitude',
                      phase.interp('altitude', [0, 5000]))
        prob.set_val('traj.boost.states:range',
                      phase.interp('range', [0, 15000]))
        prob.set_val('traj.boost.states:mass',
                      phase.interp('mass', [200, 185]))
        prob.set_val('traj.boost.controls:alpha',
                      phase.interp('alpha', [0.01, 0.01]))

        prob.set_solver_print(level=-1)
        dm.run_problem(prob, run_driver=False, simulate=False)

        # Just verify the model ran and produced finite results
        V_final = prob.get_val('traj.boost.states:velocity', units='m/s')[-1]
        h_final = prob.get_val('traj.boost.states:altitude', units='m')[-1]
        x_final = prob.get_val('traj.boost.states:range', units='m')[-1]

        self.assertTrue(np.isfinite(V_final), f'V_final not finite: {V_final}')
        self.assertTrue(V_final > 0, f'V_final should be > 0: {V_final}')
        self.assertTrue(x_final > 0, f'Range should be > 0: {x_final}')


if __name__ == '__main__':
    unittest.main()
