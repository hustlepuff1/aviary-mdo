"""High-bypass turbofan cycle model using pyCycle.

Two-spool turbofan: fan -> splitter -> core(LPC->HPC->burner->HPT->LPT) + bypass.
Adapted from pyCycle example_cycles/high_bypass_turbofan.py.
"""

import openmdao.api as om
import pycycle.api as pyc


class TurbofanCycle(pyc.Cycle):
    """Two-spool high-bypass turbofan cycle.

    Options
    -------
    design : bool
        If True, design-point. If False, off-design.
    use_tabular : bool
        Thermo method (True=TABULAR, False=CEA).
    throttle_mode : str
        'T4' or 'percent_thrust'.
    """

    def initialize(self):
        self.options.declare('use_tabular', default=True, types=bool)
        self.options.declare('throttle_mode', default='T4',
                             values=['T4', 'percent_thrust'])
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

        # Core elements
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap,
                           bleed_names=[], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap,
                           map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap,
                           bleed_names=['cool1', 'cool2', 'cust'],
                           map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(
                           bleed_names=['cool3', 'cool4']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap,
                           bleed_names=['cool3', 'cool4'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('duct11', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap,
                           bleed_names=['cool1', 'cool2'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct13', pyc.Duct())
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        # Bypass elements
        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct())
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        # Shafts and performance
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        # Performance connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        # Shaft connections
        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')

        # Nozzle exhaust conditions
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        # Flow connections
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O', 'core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        # Cooling flow connections
        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('hpc.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        # Balances
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            self.promotes('balance', inputs=[('rhs:W', 'Fn_DES')])

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])

            balance.add_balance('lpt_PR', val=10.937, lower=1.001, upper=20,
                                eq_units='hp', rhs_val=0., res_ref=1e4)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=4.185, lower=1.001, upper=8,
                                eq_units='hp', rhs_val=0., res_ref=1e4)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')
        else:
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.3)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')
            self.promotes('balance', inputs=[('rhs:FAR', 'Fn_target')])

            balance.add_balance('W', val=168.0, units='lbm/s',
                                eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', val=5.0, lower=0.25, upper=12.0,
                                eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            balance.add_balance('LP_Nmech', val=4000., units='rpm',
                                lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:LP_Nmech')

            balance.add_balance('HP_Nmech', val=15000., units='rpm',
                                lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:HP_Nmech')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver()

        super().setup()


class MPTurbofan(pyc.MPCycle):
    """Multi-point high-bypass turbofan."""

    def initialize(self):
        self.options.declare('use_tabular', default=True, types=bool)
        super().initialize()

    def setup(self):
        use_tab = self.options['use_tabular']

        self.pyc_add_pnt('DESIGN', TurbofanCycle(use_tabular=use_tab))

        self.set_input_defaults('DESIGN.fan.PR', 1.685)
        self.set_input_defaults('DESIGN.fan.eff', 0.8948)
        self.set_input_defaults('DESIGN.lpc.PR', 1.935)
        self.set_input_defaults('DESIGN.lpc.eff', 0.9243)
        self.set_input_defaults('DESIGN.hpc.PR', 10.0)
        self.set_input_defaults('DESIGN.hpc.eff', 0.8707)
        self.set_input_defaults('DESIGN.hpt.eff', 0.8888)
        self.set_input_defaults('DESIGN.lpt.eff', 0.8996)
        self.set_input_defaults('DESIGN.LP_Nmech', 4666.1, units='rpm')
        self.set_input_defaults('DESIGN.HP_Nmech', 14705.7, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.625)
        self.set_input_defaults('DESIGN.fan.MN', 0.45)
        self.set_input_defaults('DESIGN.splitter.MN1', 0.45)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.45)
        self.set_input_defaults('DESIGN.splitter.BPR', 5.105)
        self.set_input_defaults('DESIGN.duct4.MN', 0.45)
        self.set_input_defaults('DESIGN.lpc.MN', 0.45)
        self.set_input_defaults('DESIGN.duct6.MN', 0.45)
        self.set_input_defaults('DESIGN.hpc.MN', 0.30)
        self.set_input_defaults('DESIGN.bld3.MN', 0.30)
        self.set_input_defaults('DESIGN.burner.MN', 0.10)
        self.set_input_defaults('DESIGN.hpt.MN', 0.30)
        self.set_input_defaults('DESIGN.duct11.MN', 0.30)
        self.set_input_defaults('DESIGN.lpt.MN', 0.35)
        self.set_input_defaults('DESIGN.duct13.MN', 0.25)
        self.set_input_defaults('DESIGN.byp_bld.MN', 0.45)
        self.set_input_defaults('DESIGN.duct15.MN', 0.45)

        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('core_nozz.Cv', 0.9999)
        self.pyc_add_cycle_param('byp_nozz.Cv', 0.9975)
        self.pyc_add_cycle_param('hpc.cool1:frac_W', 0.0422)
        self.pyc_add_cycle_param('hpc.cool2:frac_W', 0.0177)
        self.pyc_add_cycle_param('bld3.cool3:frac_W', 0.0510)
        self.pyc_add_cycle_param('bld3.cool4:frac_W', 0.0206)
        self.pyc_add_cycle_param('hpc.cust:frac_W', 0.0)
        self.pyc_add_cycle_param('hpt.cool3:frac_P', 1.0)
        self.pyc_add_cycle_param('hpt.cool4:frac_P', 0.0)
        self.pyc_add_cycle_param('lpt.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.cool2:frac_P', 0.0)
        self.pyc_add_cycle_param('byp_bld.bypBld:frac_W', 0.005)

        # Off-design: top-of-climb
        self.pyc_add_pnt('OD_TOC', TurbofanCycle(design=False,
                         use_tabular=use_tab))
        self.set_input_defaults('OD_TOC.fc.MN', 0.8)
        self.set_input_defaults('OD_TOC.fc.alt', 35000.0, units='ft')

        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('core_nozz.Throat:stat:area', 'balance.rhs:W')
        self.pyc_connect_des_od('byp_nozz.Throat:stat:area', 'balance.rhs:BPR')

        super().setup()
