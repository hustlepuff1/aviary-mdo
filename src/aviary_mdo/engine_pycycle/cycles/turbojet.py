"""Simple turbojet cycle model using pyCycle.

Single-spool turbojet: inlet -> compressor -> burner -> turbine -> nozzle.
Adapted from pyCycle example_cycles/simple_turbojet.py.
"""

import openmdao.api as om
import pycycle.api as pyc


class TurbojetCycle(pyc.Cycle):
    """Single-spool turbojet cycle.

    Options
    -------
    design : bool
        If True, run design-point analysis. If False, off-design.
    use_tabular : bool
        If True, use tabular thermo (faster). If False, use CEA.
    """

    def initialize(self):
        self.options.declare('use_tabular', default=True, types=bool)
        super().initialize()

    def setup(self):
        design = self.options['design']

        if self.options['use_tabular']:
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = 'FAR'
        else:
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = 'Jet-A(g)'

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')

        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf',
                                rhs_name='Fn_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017,
                                rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')
        else:
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.3,
                                rhs_name='Fn_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=1.5, units='rpm', lower=500.,
                                eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            balance.add_balance('W', val=168.0, units='lbm/s',
                                eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False

        self.linear_solver = om.DirectSolver()

        super().setup()


class MPTurbojet(pyc.MPCycle):
    """Multi-point turbojet: design + off-design points.

    Parameters
    ----------
    od_alts : list of float
        Off-design altitudes (ft).
    od_MNs : list of float
        Off-design Mach numbers.
    od_Fns : list of float
        Off-design thrust targets (lbf).
    use_tabular : bool
        Thermo method selection.
    """

    def initialize(self):
        self.options.declare('od_alts', default=[0.0, 5000.0], types=list)
        self.options.declare('od_MNs', default=[0.000001, 0.2], types=list)
        self.options.declare('od_Fns', default=[11000.0, 8000.0], types=list)
        self.options.declare('use_tabular', default=True, types=bool)
        super().initialize()

    def setup(self):
        use_tab = self.options['use_tabular']

        self.pyc_add_pnt('DESIGN', TurbojetCycle(use_tabular=use_tab))
        self.set_input_defaults('DESIGN.Nmech', 8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN', 0.4)

        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozz.Cv', 0.99)

        od_alts = self.options['od_alts']
        od_MNs = self.options['od_MNs']
        od_Fns = self.options['od_Fns']

        self.od_pts = [f'OD{i}' for i in range(len(od_alts))]
        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, TurbojetCycle(design=False, use_tabular=use_tab))
            self.set_input_defaults(f'{pt}.fc.MN', val=od_MNs[i])
            self.set_input_defaults(f'{pt}.fc.alt', od_alts[i], units='ft')
            self.set_input_defaults(f'{pt}.balance.Fn_target', od_Fns[i], units='lbf')

        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()
