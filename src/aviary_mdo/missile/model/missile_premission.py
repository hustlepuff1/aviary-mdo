"""
Missile pre-mission system.

Assembles all TMD discipline components into a single OpenMDAO Group
that computes missile sizing and performance in the pre-mission phase.

This is the equivalent of running the entire TMD spreadsheet:
geometry -> aero -> propulsion -> trajectory -> structures/warhead/radar/dynamics
"""

import openmdao.api as om

from aviary_mdo.missile.model.missile_aero import (
    MissileBodyDrag, MissileWingDrag, MissileNormalForce,
)
from aviary_mdo.missile.model.missile_propulsion import RocketMotor
from aviary_mdo.missile.model.missile_trajectory import MissileTrajectoryGroup
from aviary_mdo.missile.model.missile_structures import (
    SkinTemperature, MotorCaseStress, MissileWeightEstimation,
)


class MissilePreMission(om.Group):
    """
    Complete missile pre-mission sizing and synthesis.

    Computes all TMD outputs from the input geometry, propulsion,
    and flight condition parameters. Mirrors the TMD spreadsheet's
    Master Input -> individual sheets -> Master Output flow.
    """

    def initialize(self):
        self.options.declare('sustain_engine', default='rocket',
                            values=['rocket', 'ramjet'],
                            desc='Type of sustain engine')

    def setup(self):
        sustain_type = self.options['sustain_engine']

        # --- Propulsion: compute thrust and Isp ---
        self.add_subsystem('boost_motor', RocketMotor(),
                           promotes_inputs=[
                               ('Pc', 'boost_Pc'),
                               ('expansion_ratio', 'boost_expansion_ratio'),
                               ('fuel_type', 'boost_fuel_type'),
                               ('propellant_weight', 'W_boost_prop'),
                               ('burn_time', 'boost_burn_time'),
                               ('Pa', 'Pa'),
                           ])

        self.add_subsystem('sustain_motor', RocketMotor(),
                           promotes_inputs=[
                               ('Pc', 'sustain_Pc'),
                               ('expansion_ratio', 'sustain_expansion_ratio'),
                               ('fuel_type', 'sustain_fuel_type'),
                               ('propellant_weight', 'W_sustain_prop'),
                               ('burn_time', 'sustain_burn_time'),
                               ('Pa', 'Pa'),
                           ])

        # --- Aerodynamics at launch condition ---
        self.add_subsystem('body_drag_launch',
                           MissileBodyDrag(power_on=False),
                           promotes_inputs=[
                               ('mach', 'launch_mach'),
                               ('length', 'missile_length'),
                               ('diameter', 'missile_diameter'),
                               ('nose_length', 'nose_length'),
                               ('nose_bluntness', 'nose_bluntness'),
                               ('ref_area', 'ref_area'),
                               ('alt', 'launch_alt'),
                           ])

        self.add_subsystem('wing_drag_launch',
                           MissileWingDrag(),
                           promotes_inputs=[
                               ('mach', 'launch_mach'),
                               ('ref_area', 'ref_area'),
                               ('alt', 'launch_alt'),
                               ('planform_area', 'wing_area'),
                               ('thickness_to_chord', 'wing_tc'),
                               ('le_thickness_angle', 'wing_le_angle'),
                               ('aspect_ratio', 'wing_AR'),
                               ('taper_ratio', 'wing_taper'),
                               ('sweep', 'wing_sweep'),
                           ])

        self.add_subsystem('tail_drag_launch',
                           MissileWingDrag(),
                           promotes_inputs=[
                               ('mach', 'launch_mach'),
                               ('ref_area', 'ref_area'),
                               ('alt', 'launch_alt'),
                               ('aspect_ratio', 'tail_AR'),
                               ('taper_ratio', 'tail_taper'),
                           ])

        # Sum launch CD0
        self.add_subsystem('cd0_launch',
                           om.ExecComp(
                               'CD0 = CD_body + CD_wing + CD_tail',
                               CD0={'val': 0.3},
                               CD_body={'val': 0.0},
                               CD_wing={'val': 0.0},
                               CD_tail={'val': 0.0},
                           ))
        self.connect('body_drag_launch.CD_body_total', 'cd0_launch.CD_body')
        self.connect('wing_drag_launch.CD_surface_total', 'cd0_launch.CD_wing')
        self.connect('tail_drag_launch.CD_surface_total', 'cd0_launch.CD_tail')

        # --- Normal force at reference condition ---
        self.add_subsystem('normal_force',
                           MissileNormalForce(),
                           promotes_inputs=[
                               ('mach', 'launch_mach'),
                               ('body_diameter', 'missile_diameter'),
                               ('body_length', 'missile_length'),
                               ('nose_length', 'nose_length'),
                               ('ref_area', 'ref_area'),
                               ('cg_station', 'cg_station'),
                               ('num_wings', 'num_wings'),
                               ('wing_area', 'wing_area'),
                               ('wing_AR', 'wing_AR'),
                               ('wing_le_station', 'wing_le_station'),
                               ('wing_taper', 'wing_taper'),
                               ('num_tails', 'num_tails'),
                               ('tail_area', 'tail_area'),
                               ('tail_AR', 'tail_AR'),
                               ('tail_le_station', 'tail_le_station'),
                               ('tail_taper', 'tail_taper'),
                           ],
                           promotes_outputs=['CN_total', 'Xcp'])

        # --- Trajectory ---
        self.add_subsystem('trajectory',
                           MissileTrajectoryGroup(sustain_engine=sustain_type),
                           promotes_inputs=[('alt', 'launch_alt'),
                                            'ref_area'],
                           promotes_outputs=['range_total', 'time_total'])

        # Connect propulsion to trajectory
        self.connect('boost_motor.thrust', 'trajectory.boost.thrust_boost')
        self.connect('boost_motor.Isp', 'trajectory.boost.Isp_boost')
        self.connect('sustain_motor.thrust', 'trajectory.sustain.thrust_sustain')
        self.connect('sustain_motor.Isp', 'trajectory.sustain.Isp_sustain')
        self.connect('cd0_launch.CD0', 'trajectory.boost.CD0_launch')

        # Connect nozzle exit area for body drag
        self.connect('boost_motor.exit_area',
                     'body_drag_launch.nozzle_exit_area')

        # --- Structures ---
        self.add_subsystem('skin_temp',
                           SkinTemperature(),
                           promotes_inputs=[('alt', 'launch_alt')])
        self.connect('trajectory.boost.mach_boost_end', 'skin_temp.mach')

        self.add_subsystem('motor_case',
                           MotorCaseStress(),
                           promotes_inputs=[
                               ('missile_diameter', 'missile_diameter'),
                               ('chamber_pressure', 'boost_Pc'),
                           ])

        # Set default values for inputs that need them
        self.set_input_defaults('launch_mach', val=0.8)
        self.set_input_defaults('launch_alt', val=20000.0, units='ft')
        self.set_input_defaults('missile_length', val=143.9, units='inch')
        self.set_input_defaults('missile_diameter', val=8.0, units='inch')
        self.set_input_defaults('nose_length', val=19.2, units='inch')
        self.set_input_defaults('nose_bluntness', val=0.05)
        self.set_input_defaults('ref_area', val=50.27, units='inch**2')
        self.set_input_defaults('cg_station', val=76.2, units='inch')

        self.set_input_defaults('num_wings', val=2.0)
        self.set_input_defaults('wing_area', val=400.0, units='inch**2')
        self.set_input_defaults('wing_AR', val=2.82)
        self.set_input_defaults('wing_taper', val=0.175)
        self.set_input_defaults('wing_le_station', val=60.8, units='inch')
        self.set_input_defaults('wing_tc', val=0.044)
        self.set_input_defaults('wing_le_angle', val=10.0, units='deg')
        self.set_input_defaults('wing_sweep', val=45.0, units='deg')

        self.set_input_defaults('num_tails', val=2.0)
        self.set_input_defaults('tail_area', val=87.0, units='inch**2')
        self.set_input_defaults('tail_AR', val=2.59)
        self.set_input_defaults('tail_taper', val=0.0)
        self.set_input_defaults('tail_le_station', val=125.4, units='inch')

        self.set_input_defaults('boost_Pc', val=1769.0, units='lbf/inch**2')
        self.set_input_defaults('boost_expansion_ratio', val=6.0)
        self.set_input_defaults('boost_fuel_type', val=4.0)
        self.set_input_defaults('W_boost_prop', val=84.8, units='lbm')
        self.set_input_defaults('boost_burn_time', val=3.26, units='s')
        self.set_input_defaults('Pa', val=6.76, units='lbf/inch**2')

        self.set_input_defaults('sustain_Pc', val=301.0, units='lbf/inch**2')
        self.set_input_defaults('sustain_expansion_ratio', val=6.2)
        self.set_input_defaults('sustain_fuel_type', val=4.0)
        self.set_input_defaults('W_sustain_prop', val=48.2, units='lbm')
        self.set_input_defaults('sustain_burn_time', val=10.86, units='s')
