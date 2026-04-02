"""
Tactical Missile Design structures components.

Implements structural analysis from TMD spreadsheet Structure sheet (sheet 7):
  - Motor case stress (thin-wall pressure vessel hoop stress, MEOP)
  - Skin temperature (aerodynamic heating, recovery temperature)
  - Weight estimation (regression, CG tracking, moment of inertia)

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001.
"""

import numpy as np
import openmdao.api as om

from aviary_mdo.missile.model.missile_aero import _atmosphere


class MotorCaseStress(om.ExplicitComponent):
    """
    Motor case stress analysis using thin-wall pressure vessel theory.

    Computes MEOP (Maximum Expected Operating Pressure), hoop stress,
    required wall thickness, and motor case weight.
    """

    def setup(self):
        # Inputs
        self.add_input('chamber_pressure', val=2151.95, units='psi',
                       desc='Motor chamber pressure Pc')
        self.add_input('e_multiplier', val=1.103,
                       desc='3-sigma margin multiplier for MEOP')
        self.add_input('missile_diameter', val=8.0, units='inch',
                       desc='Missile outer diameter')
        self.add_input('case_length', val=60.0, units='inch',
                       desc='Motor case length')
        self.add_input('material_yield', val=180000.0, units='psi',
                       desc='Case material yield strength')
        self.add_input('material_density', val=0.283, units='lbm/inch**3',
                       desc='Case material density (steel ~0.283)')
        self.add_input('safety_factor', val=1.25,
                       desc='Structural safety factor')

        # Outputs
        self.add_output('MEOP', val=0.0, units='psi',
                        desc='Maximum Expected Operating Pressure')
        self.add_output('hoop_stress', val=0.0, units='psi',
                        desc='Hoop stress at required thickness')
        self.add_output('wall_thickness', val=0.0, units='inch',
                        desc='Required wall thickness')
        self.add_output('case_weight', val=0.0, units='lbm',
                        desc='Motor case weight')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Pc = inputs['chamber_pressure']
        e_mult = inputs['e_multiplier']
        d = inputs['missile_diameter']
        L_case = inputs['case_length']
        sigma_y = inputs['material_yield']
        rho_mat = inputs['material_density']
        SF = inputs['safety_factor']

        r = d / 2.0  # inner radius

        # MEOP = Pc * margin multiplier (TMD spreadsheet R17/R19)
        MEOP = Pc * e_mult
        outputs['MEOP'] = MEOP

        # Required wall thickness from thin-wall hoop stress:
        #   sigma = MEOP * r / t  =>  t = MEOP * r / sigma_allowable
        sigma_allow = sigma_y / SF
        t = MEOP * r / sigma_allow
        outputs['wall_thickness'] = t

        # Hoop stress at this thickness (should equal sigma_allow)
        outputs['hoop_stress'] = sigma_allow

        # Motor case weight: cylindrical shell + two end caps (simplified as flat)
        # Cylinder shell volume
        r_outer = r + t
        V_cyl = np.pi * (r_outer**2 - r**2) * L_case
        # Two hemispherical end caps approximated as flat disks of thickness t
        V_caps = 2.0 * np.pi * r_outer**2 * t
        V_total = V_cyl + V_caps
        outputs['case_weight'] = V_total * rho_mat


class SkinTemperature(om.ExplicitComponent):
    """
    Aerodynamic heating skin temperature calculation.

    Computes recovery (stagnation-like) temperature using the recovery factor
    approach from TMD. The recovery temperature is the equilibrium wall
    temperature for an insulated surface.

    T_recovery = T_atm * (1 + recovery_factor * (gamma - 1) / 2 * M^2)

    where gamma = 1.4 for air.
    """

    def setup(self):
        self.add_input('mach', val=2.0, desc='Freestream Mach number')
        self.add_input('alt', val=20000.0, units='ft', desc='Altitude')
        self.add_input('recovery_factor', val=0.9,
                       desc='Recovery factor (0.85-0.9 for turbulent BL)')

        self.add_output('T_freestream', val=0.0, units='degR',
                        desc='Freestream static temperature')
        self.add_output('T_recovery', val=0.0, units='degR',
                        desc='Recovery (adiabatic wall) temperature')
        self.add_output('T_recovery_F', val=0.0, units='degF',
                        desc='Recovery temperature in Fahrenheit')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['mach']
        alt = inputs['alt']
        r_f = inputs['recovery_factor']

        gamma = 1.4

        T_atm, _rho, _mu = _atmosphere(alt)

        outputs['T_freestream'] = T_atm

        # Recovery temperature
        T_rec = T_atm * (1.0 + r_f * (gamma - 1.0) / 2.0 * M**2)
        outputs['T_recovery'] = T_rec

        # Convert Rankine to Fahrenheit
        outputs['T_recovery_F'] = T_rec - 459.67


class MissileWeightEstimation(om.ExplicitComponent):
    """
    Missile weight estimation, CG tracking, and moment of inertia.

    Uses TMD spreadsheet Structure sheet regression and geometry for:
    - Total estimated weight from regression (R2: 391.28 lbm)
    - CG at launch and end-boost
    - Boost fuel length and CG
    - Moment of inertia for cylinder body and cone nose

    TMD spreadsheet validation (Structure sheet):
      R2:  estimated weight = 391.28 lbm
      R5:  CG at launch = 76.2 in
      R6:  CG at end boost = 73.08 in
      R8:  Iy_cylinder = 102.6 slugs-ft^2
      R12: Iy_nose = 0.061 slugs-ft^2
      R13: Boost fuel length = 20.32 in
      R14: Boost fuel CG = 91.48 in
    """

    def setup(self):
        # Geometry inputs
        self.add_input('missile_diameter', val=8.0, units='inch',
                       desc='Missile diameter')
        self.add_input('missile_length', val=143.9, units='inch',
                       desc='Total missile length')
        self.add_input('nose_length', val=19.2, units='inch',
                       desc='Nose cone length')

        # Weight inputs
        self.add_input('W_launch', val=500.0, units='lbm',
                       desc='Launch weight')
        self.add_input('W_payload', val=44.0, units='lbm',
                       desc='Payload (warhead) weight')
        self.add_input('W_guidance', val=30.0, units='lbm',
                       desc='Guidance section weight')
        self.add_input('W_structure', val=80.0, units='lbm',
                       desc='Structural weight (case, fins, etc.)')
        self.add_input('W_nose', val=17.42, units='lbm',
                       desc='Nose section weight (radome, seeker)')
        self.add_input('W_body', val=365.7, units='lbm',
                       desc='Cylindrical body weight for Iy calculation')

        # Propulsion inputs
        self.add_input('boost_fuel_weight', val=112.0, units='lbm',
                       desc='Boost phase fuel weight')
        self.add_input('sustain_fuel_weight', val=40.0, units='lbm',
                       desc='Sustain phase fuel weight')
        self.add_input('motor_length', val=60.0, units='inch',
                       desc='Motor case length')
        self.add_input('fuel_density', val=0.065, units='lbm/inch**3',
                       desc='Solid propellant density')
        self.add_input('fuel_outer_radius', val=5.195, units='inch',
                       desc='Fuel grain outer radius')

        # Section CG locations (from nose)
        self.add_input('cg_payload', val=25.0, units='inch',
                       desc='Payload CG station from nose')
        self.add_input('cg_guidance', val=40.0, units='inch',
                       desc='Guidance section CG from nose')
        self.add_input('cg_structure', val=80.0, units='inch',
                       desc='Structure CG from nose')
        self.add_input('motor_start_station', val=81.32, units='inch',
                       desc='Motor section start station from nose')

        # Outputs
        self.add_output('W_estimated', val=0.0, units='lbm',
                        desc='Estimated weight from regression')
        self.add_output('cg_launch', val=0.0, units='inch',
                        desc='CG at launch from nose')
        self.add_output('cg_end_boost', val=0.0, units='inch',
                        desc='CG at end of boost from nose')
        self.add_output('boost_fuel_length', val=0.0, units='inch',
                        desc='Boost fuel grain length')
        self.add_output('boost_fuel_cg', val=0.0, units='inch',
                        desc='Boost fuel CG from nose')
        self.add_output('sustain_fuel_length', val=0.0, units='inch',
                        desc='Sustain fuel grain length')
        self.add_output('sustain_fuel_cg', val=0.0, units='inch',
                        desc='Sustain fuel CG from nose')
        self.add_output('Iy_cylinder', val=0.0, units='slug*ft**2',
                        desc='Pitch moment of inertia of cylindrical body')
        self.add_output('Iy_nose', val=0.0, units='slug*ft**2',
                        desc='Pitch moment of inertia of nose cone')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        d = inputs['missile_diameter']
        L = inputs['missile_length']
        ln = inputs['nose_length']
        W_launch = inputs['W_launch']
        W_pay = inputs['W_payload']
        W_guid = inputs['W_guidance']
        W_struct = inputs['W_structure']
        W_nose = inputs['W_nose']
        W_body = inputs['W_body']
        W_boost = inputs['boost_fuel_weight']
        W_sust = inputs['sustain_fuel_weight']
        L_motor = inputs['motor_length']
        rho_fuel = inputs['fuel_density']
        r_fuel = inputs['fuel_outer_radius']
        cg_pay = inputs['cg_payload']
        cg_guid = inputs['cg_guidance']
        cg_struct = inputs['cg_structure']
        motor_start = inputs['motor_start_station']

        r = d / 2.0

        # --- Weight regression (TMD Structure sheet R2) ---
        # Fleeman regression: W_est = k * d^2.4 * (L_ft)^0.4
        # where d in inches, L_ft = L/12 in feet
        # k = 0.9852 calibrated to TMD spreadsheet (391.28 lbm for baseline)
        W_est = 0.9852 * d**2.4 * (L / 12.0)**0.4
        outputs['W_estimated'] = W_est

        # --- Fuel geometry ---
        # Fuel grain cross-section area (solid cylinder, no bore for simplicity)
        A_fuel = np.pi * r_fuel**2

        # Boost fuel length
        if A_fuel * rho_fuel > 0:
            L_boost_fuel = W_boost / (A_fuel * rho_fuel)
        else:
            L_boost_fuel = 0.0
        outputs['boost_fuel_length'] = L_boost_fuel

        # Sustain fuel length
        if A_fuel * rho_fuel > 0:
            L_sust_fuel = W_sust / (A_fuel * rho_fuel)
        else:
            L_sust_fuel = 0.0
        outputs['sustain_fuel_length'] = L_sust_fuel

        # Boost fuel CG: centered in its grain, placed at start of motor
        boost_fuel_start = motor_start
        boost_fuel_cg = boost_fuel_start + L_boost_fuel / 2.0
        outputs['boost_fuel_cg'] = boost_fuel_cg

        # Sustain fuel CG: aft of boost fuel
        sustain_fuel_start = boost_fuel_start + L_boost_fuel
        sustain_fuel_cg = sustain_fuel_start + L_sust_fuel / 2.0
        outputs['sustain_fuel_cg'] = sustain_fuel_cg

        # --- CG tracking ---
        # CG at launch: weighted average of all sections
        # "Other" weight is everything not explicitly accounted for
        W_other = W_launch - W_pay - W_guid - W_struct - W_boost - W_sust
        # Assume "other" CG is at missile mid-length
        cg_other = L / 2.0

        if W_launch > 0:
            cg_l = (W_pay * cg_pay + W_guid * cg_guid +
                    W_struct * cg_struct + W_boost * boost_fuel_cg +
                    W_sust * sustain_fuel_cg + W_other * cg_other) / W_launch
        else:
            cg_l = L / 2.0
        outputs['cg_launch'] = cg_l

        # CG at end of boost: boost fuel consumed
        W_end_boost = W_launch - W_boost
        if W_end_boost > 0:
            cg_eb = (W_pay * cg_pay + W_guid * cg_guid +
                     W_struct * cg_struct +
                     W_sust * sustain_fuel_cg +
                     W_other * cg_other) / W_end_boost
        else:
            cg_eb = L / 2.0
        outputs['cg_end_boost'] = cg_eb

        # --- Moment of inertia ---
        # Iy for uniform solid cylinder about its centroid (transverse/pitch):
        #   Iy = m * (3*r^2 + L^2) / 12
        L_cyl = L - ln  # cylindrical body length

        # Cylinder body mass from W_body input
        m_cyl = W_body / 32.174  # slugs

        L_cyl_ft = L_cyl / 12.0
        r_ft = r / 12.0
        Iy_cyl = m_cyl * (3.0 * r_ft**2 + L_cyl_ft**2) / 12.0
        outputs['Iy_cylinder'] = Iy_cyl

        # Nose cone Iy (solid cone about its centroid, transverse axis):
        #   Iy = m * (3*r^2/20 + 3*h^2/80)
        m_nose = W_nose / 32.174  # slugs
        ln_ft = ln / 12.0
        Iy_nose_centroid = m_nose * (3.0 * r_ft**2 / 20.0 + 3.0 * ln_ft**2 / 80.0)
        outputs['Iy_nose'] = Iy_nose_centroid


class MissileStructuresGroup(om.Group):
    """
    Complete missile structures group assembling motor case, skin temperature,
    and weight estimation components.
    """

    def setup(self):
        self.add_subsystem('motor_case', MotorCaseStress(),
                           promotes_inputs=['missile_diameter'],
                           promotes_outputs=['MEOP', 'wall_thickness',
                                             'case_weight'])

        self.add_subsystem('skin_temp', SkinTemperature(),
                           promotes_inputs=['mach', 'alt'],
                           promotes_outputs=['T_recovery', 'T_recovery_F'])

        self.add_subsystem('weight', MissileWeightEstimation(),
                           promotes_inputs=['missile_diameter',
                                            'missile_length',
                                            'nose_length'],
                           promotes_outputs=['W_estimated', 'cg_launch',
                                             'cg_end_boost',
                                             'Iy_cylinder', 'Iy_nose'])
