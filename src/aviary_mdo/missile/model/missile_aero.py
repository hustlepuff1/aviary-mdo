"""
Tactical Missile Design aerodynamics component.

Implements Fleeman's component drag buildup and normal force methods
from "Tactical Missile Design" (AIAA Education Series).

Drag buildup:
  CD0 = CD_body_friction + CD_body_wave + CD_wing_friction + CD_wing_wave
        + CD_tail_friction + CD_tail_wave + CD_base

Normal force:
  CN = CN_body + CN_wing + CN_tail

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001.
"""

import numpy as np
import openmdao.api as om


class MissileBodyDrag(om.ExplicitComponent):
    """
    Body zero-lift drag coefficient (friction + wave + base).

    Uses turbulent flat plate skin friction with compressibility correction,
    slender body wave drag, and base drag with power-on/off options.
    """

    def initialize(self):
        self.options.declare('power_on', default=True, types=bool,
                            desc='Whether engine is firing (affects base drag)')

    def setup(self):
        self.add_input('mach', val=2.0, desc='Freestream Mach number')
        self.add_input('length', val=143.9, units='inch',
                       desc='Missile body length')
        self.add_input('diameter', val=8.0, units='inch',
                       desc='Missile effective diameter')
        self.add_input('nose_length', val=19.2, units='inch',
                       desc='Nose length')
        self.add_input('nose_bluntness', val=0.05,
                       desc='Nose bluntness ratio (hemispherical dia / body dia)')
        self.add_input('nozzle_exit_area', val=14.52, units='inch**2',
                       desc='Nozzle exit area')
        self.add_input('ref_area', val=50.27, units='inch**2',
                       desc='Missile reference area')
        self.add_input('alt', val=20000.0, units='ft',
                       desc='Altitude for Reynolds number')

        self.add_output('CD_body_friction', val=0.0,
                        desc='Body skin friction drag coefficient')
        self.add_output('CD_body_wave', val=0.0,
                        desc='Body wave drag coefficient')
        self.add_output('CD_base', val=0.0,
                        desc='Base drag coefficient')
        self.add_output('CD_body_total', val=0.0,
                        desc='Total body drag coefficient')
        self.add_output('nose_fineness', val=0.0,
                        desc='Nose fineness ratio ln/d')
        self.add_output('body_fineness', val=0.0,
                        desc='Body fineness ratio l/d')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['mach']
        length = inputs['length']
        d = inputs['diameter']
        ln = inputs['nose_length']
        blunt = inputs['nose_bluntness']
        A_exit = inputs['nozzle_exit_area']
        S_ref = inputs['ref_area']
        alt = inputs['alt']

        fn = ln / d  # nose fineness ratio
        fb = length / d  # body fineness ratio
        outputs['nose_fineness'] = fn
        outputs['body_fineness'] = fb

        # Wetted area of body (cylinder + ogive nose approximation)
        r = d / 2.0
        S_body_wet = np.pi * d * (length - ln) + np.pi * r * np.sqrt(r**2 + ln**2)

        # --- Body friction drag ---
        # Reynolds number (using standard atmosphere approximation)
        # Kinematic viscosity from altitude (simplified Sutherland's law)
        T_atm, rho, mu = _atmosphere(alt)
        a_sound = np.sqrt(1.4 * 1716.49 * T_atm)  # ft/s
        V = M * a_sound
        length_ft = length / 12.0
        Re = rho * V * length_ft / mu

        # Turbulent flat plate skin friction (Schlichting)
        Cf = 0.455 / (np.log10(Re))**2.58

        # Compressibility correction (van Driest II approximation)
        Cf_comp = Cf / (1.0 + 0.144 * M**2)**0.65

        # Body friction drag coefficient
        CD_bf = Cf_comp * S_body_wet / S_ref
        outputs['CD_body_friction'] = CD_bf

        # --- Body wave drag ---
        # Nose wave drag for ogive at supersonic (Fleeman empirical)
        if M > 1.0:
            # Transonic/supersonic nose wave drag
            CD_nose_wave = _nose_wave_drag(M, fn, blunt)
        else:
            CD_nose_wave = 0.0

        outputs['CD_body_wave'] = CD_nose_wave

        # --- Base drag ---
        # Base area = reference area - nozzle exit area (if power on)
        A_base = S_ref
        if self.options['power_on']:
            A_base = max(S_ref - A_exit, 0.0)

        # Fleeman base drag correlation
        if M < 1.0:
            CD_base_ref = 0.12 + 0.13 * M**2
        else:
            CD_base_ref = 0.25 / M

        CD_base = CD_base_ref * A_base / S_ref
        outputs['CD_base'] = CD_base

        outputs['CD_body_total'] = CD_bf + CD_nose_wave + CD_base


class MissileWingDrag(om.ExplicitComponent):
    """
    Wing (or tail) zero-lift drag coefficient (friction + wave).

    Handles both wing and tail surfaces with the same equations.
    """

    def setup(self):
        self.add_input('mach', val=2.0, desc='Freestream Mach number')
        self.add_input('num_surfaces', val=2.0,
                       desc='Number of surfaces (wings or tails, per pair)')
        self.add_input('planform_area', val=400.0, units='inch**2',
                       desc='Total planform area of surfaces')
        self.add_input('thickness_to_chord', val=0.044,
                       desc='Maximum thickness-to-chord ratio')
        self.add_input('le_thickness_angle', val=10.0, units='deg',
                       desc='Leading edge thickness half-angle')
        self.add_input('aspect_ratio', val=2.82,
                       desc='Wing aspect ratio')
        self.add_input('taper_ratio', val=0.175,
                       desc='Wing taper ratio')
        self.add_input('ref_area', val=50.27, units='inch**2',
                       desc='Missile reference area')
        self.add_input('alt', val=20000.0, units='ft',
                       desc='Altitude')
        self.add_input('sweep', val=45.0, units='deg',
                       desc='Wing sweep angle')

        self.add_output('CD_surface_friction', val=0.0,
                        desc='Surface skin friction drag coefficient')
        self.add_output('CD_surface_wave', val=0.0,
                        desc='Surface wave drag coefficient')
        self.add_output('CD_surface_total', val=0.0,
                        desc='Total surface drag coefficient')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['mach']
        n_surf = inputs['num_surfaces']
        S_plan = inputs['planform_area']
        tc = inputs['thickness_to_chord']
        le_angle = inputs['le_thickness_angle'] * np.pi / 180.0
        AR = inputs['aspect_ratio']
        taper = inputs['taper_ratio']
        S_ref = inputs['ref_area']
        alt = inputs['alt']
        sweep = inputs['sweep'] * np.pi / 180.0

        # Wetted area (both sides of the surface)
        S_wet = 2.0 * n_surf * S_plan

        # Mean aerodynamic chord
        span = np.sqrt(AR * S_plan)
        c_root = 2.0 * S_plan / (span * (1.0 + taper))
        c_mac = c_root * (2.0 / 3.0) * (1.0 + taper + taper**2) / (1.0 + taper)

        # Reynolds number based on MAC
        T_atm, rho, mu = _atmosphere(alt)
        a_sound = np.sqrt(1.4 * 1716.49 * T_atm)
        V = M * a_sound
        c_mac_ft = c_mac / 12.0
        Re_mac = rho * V * c_mac_ft / mu

        # Skin friction
        if Re_mac > 0:
            Cf = 0.455 / (np.log10(max(Re_mac, 1e3)))**2.58
        else:
            Cf = 0.003

        Cf_comp = Cf / (1.0 + 0.144 * M**2)**0.65

        CD_fric = Cf_comp * S_wet / S_ref
        outputs['CD_surface_friction'] = CD_fric

        # Wave drag (supersonic only)
        if M > 1.0:
            # Supersonic wing wave drag (Fleeman: based on t/c and LE angle)
            # Ackeret theory corrected for sweep
            M_n = M * np.cos(sweep)  # normal Mach component
            if M_n > 1.0:
                beta = np.sqrt(M_n**2 - 1.0)
                CD_wave = (4.0 * (tc**2)) / beta * (S_plan * n_surf / S_ref)
            else:
                # Subsonic leading edge
                CD_wave = (tc**2) * (S_plan * n_surf / S_ref) * 0.5
        else:
            CD_wave = 0.0

        outputs['CD_surface_wave'] = CD_wave
        outputs['CD_surface_total'] = CD_fric + CD_wave


class MissileNormalForce(om.ExplicitComponent):
    """
    Total normal force coefficient CN(alpha) for the missile.

    Combines body, wing, and tail contributions:
    - Body: slender body theory + crossflow theory
    - Wing: linear wing theory (low alpha) + Newtonian (high alpha)
    - Tail: same as wing with downwash correction

    Also computes center of pressure Xcp.
    """

    def setup(self):
        self.add_input('alpha', val=10.0, units='deg',
                       desc='Angle of attack')
        self.add_input('mach', val=2.0, desc='Freestream Mach number')

        # Body geometry
        self.add_input('body_diameter', val=8.0, units='inch')
        self.add_input('body_length', val=143.9, units='inch')
        self.add_input('nose_length', val=19.2, units='inch')
        self.add_input('ref_area', val=50.27, units='inch**2')
        self.add_input('cg_station', val=76.2, units='inch')

        # Wing geometry
        self.add_input('num_wings', val=2.0)
        self.add_input('wing_area', val=400.0, units='inch**2')
        self.add_input('wing_AR', val=2.82)
        self.add_input('wing_le_station', val=60.8, units='inch')
        self.add_input('wing_taper', val=0.175)

        # Tail geometry
        self.add_input('num_tails', val=2.0)
        self.add_input('tail_area', val=87.0, units='inch**2')
        self.add_input('tail_AR', val=2.59)
        self.add_input('tail_le_station', val=125.4, units='inch')
        self.add_input('tail_taper', val=0.0)

        self.add_output('CN_body', val=0.0, desc='Body normal force coefficient')
        self.add_output('CN_wing', val=0.0, desc='Wing normal force coefficient')
        self.add_output('CN_tail', val=0.0, desc='Tail normal force coefficient')
        self.add_output('CN_total', val=0.0, desc='Total normal force coefficient')
        self.add_output('Xcp', val=0.0, units='inch',
                        desc='Center of pressure from nose')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        alpha_deg = inputs['alpha']
        alpha = alpha_deg * np.pi / 180.0
        M = inputs['mach']
        d = inputs['body_diameter']
        L = inputs['body_length']
        ln = inputs['nose_length']
        S_ref = inputs['ref_area']
        x_cg = inputs['cg_station']

        n_w = inputs['num_wings']
        S_w = inputs['wing_area']
        AR_w = inputs['wing_AR']
        x_w_le = inputs['wing_le_station']
        taper_w = inputs['wing_taper']

        n_t = inputs['num_tails']
        S_t = inputs['tail_area']
        AR_t = inputs['tail_AR']
        x_t_le = inputs['tail_le_station']
        taper_t = inputs['tail_taper']

        # --- Body normal force ---
        # Slender body theory: CN_body = 2 * alpha (per radian) * S_base/S_ref
        # Plus crossflow at higher alpha
        S_base = np.pi * (d / 2.0)**2
        CN_alpha_body = 2.0 * (S_base / S_ref)  # per radian

        # Crossflow drag contribution (empirical, Fleeman)
        Cd_crossflow = 1.2  # crossflow drag coefficient for cylinder
        S_planform_body = d * (L - ln)  # projected side area of cylindrical section
        CN_crossflow = Cd_crossflow * (S_planform_body / S_ref) * np.sin(alpha) * abs(np.sin(alpha))

        CN_b = CN_alpha_body * alpha + CN_crossflow
        outputs['CN_body'] = CN_b

        # Body Cp (slender body portion at 2/3 nose length, crossflow at body center)
        if abs(CN_b) > 1e-10:
            Xcp_body = ((CN_alpha_body * alpha) * (2.0 / 3.0 * ln)
                        + CN_crossflow * (ln + (L - ln) / 2.0)) / CN_b
        else:
            Xcp_body = 2.0 / 3.0 * ln

        # --- Wing normal force ---
        CN_w, Xcp_wing = _surface_normal_force(
            alpha, M, n_w, S_w, AR_w, x_w_le, taper_w, S_ref)
        outputs['CN_wing'] = CN_w

        # --- Tail normal force ---
        # Apply downwash factor (simplified: 0.5 for wing wake effect)
        alpha_tail = alpha * 0.8  # reduced effective alpha at tail
        CN_t, Xcp_tail = _surface_normal_force(
            alpha_tail, M, n_t, S_t, AR_t, x_t_le, taper_t, S_ref)
        outputs['CN_tail'] = CN_t

        # --- Total ---
        CN_total = CN_b + CN_w + CN_t
        outputs['CN_total'] = CN_total

        # Center of pressure
        if abs(CN_total) > 1e-10:
            Xcp = (CN_b * Xcp_body + CN_w * Xcp_wing + CN_t * Xcp_tail) / CN_total
        else:
            Xcp = x_cg
        outputs['Xcp'] = Xcp


class MissileAeroGroup(om.Group):
    """
    Complete missile aerodynamics group assembling drag and normal force.

    Outputs total CD0 and CN at a given flight condition.
    """

    def initialize(self):
        self.options.declare('power_on', default=True, types=bool)

    def setup(self):
        power_on = self.options['power_on']

        self.add_subsystem('body_drag',
                           MissileBodyDrag(power_on=power_on),
                           promotes_inputs=['mach', 'ref_area', 'alt'],
                           promotes_outputs=['CD_body_total',
                                             'nose_fineness', 'body_fineness'])

        self.add_subsystem('wing_drag',
                           MissileWingDrag(),
                           promotes_inputs=['mach', 'ref_area', 'alt'])

        self.add_subsystem('tail_drag',
                           MissileWingDrag(),
                           promotes_inputs=['mach', 'ref_area', 'alt'])

        self.add_subsystem('normal_force',
                           MissileNormalForce(),
                           promotes_inputs=['alpha', 'mach', 'ref_area'],
                           promotes_outputs=['CN_total', 'Xcp'])

        # Sum total drag
        self.add_subsystem('drag_sum',
                           om.ExecComp(
                               'CD0 = CD_body + CD_wing + CD_tail',
                               CD0={'val': 0.0},
                               CD_body={'val': 0.0},
                               CD_wing={'val': 0.0},
                               CD_tail={'val': 0.0},
                           ),
                           promotes_outputs=['CD0'])

        self.connect('CD_body_total', 'drag_sum.CD_body')
        self.connect('wing_drag.CD_surface_total', 'drag_sum.CD_wing')
        self.connect('tail_drag.CD_surface_total', 'drag_sum.CD_tail')


# ===== Helper functions =====

def _atmosphere(alt_ft):
    """
    Simplified 1976 US Standard Atmosphere.

    Parameters
    ----------
    alt_ft : float
        Altitude in feet.

    Returns
    -------
    T : float
        Temperature in Rankine.
    rho : float
        Density in slugs/ft^3.
    mu : float
        Dynamic viscosity in slugs/(ft*s).
    """
    alt_ft = max(alt_ft, 0.0)

    if alt_ft <= 36089.0:
        # Troposphere
        T = 518.67 - 0.003566 * alt_ft  # Rankine
        p = 2116.22 * (T / 518.67)**5.2561  # psf
    else:
        # Stratosphere (isothermal layer up to 65,617 ft)
        T = 389.97  # Rankine
        p = 2116.22 * 0.22336 * np.exp(-4.80614e-5 * (alt_ft - 36089.0))

    rho = p / (1716.49 * T)  # slugs/ft^3

    # Sutherland's law for air viscosity
    T_kelvin = T * 5.0 / 9.0
    mu_si = 1.458e-6 * T_kelvin**1.5 / (T_kelvin + 110.4)  # Pa*s
    mu = mu_si * 0.020885  # convert to slugs/(ft*s)

    return T, rho, mu


def _nose_wave_drag(M, fn, bluntness):
    """
    Nose wave drag coefficient (referenced to body cross-section area).

    Uses Fleeman's empirical correlation for ogive noses at supersonic Mach.

    Parameters
    ----------
    M : float
        Mach number (must be > 1.0).
    fn : float
        Nose fineness ratio (nose_length / diameter).
    bluntness : float
        Nose bluntness ratio.

    Returns
    -------
    CD_wave : float
        Nose wave drag coefficient (on ref area).
    """
    if M <= 1.0:
        return 0.0

    # Ogive nose wave drag (Fleeman approximation)
    # For pointed ogive: CD_wave ~ 1 / (fn^2) at low supersonic
    # Modified for bluntness
    beta = np.sqrt(M**2 - 1.0)

    # Van Dyke second-order theory for ogive
    if fn > 0.5:
        CD_wave_nose = (1.0 / (2.0 * fn**2)) * (1.0 + bluntness * 5.0)
    else:
        # Blunt nose approximation
        CD_wave_nose = 0.9 * (1.0 - 1.0 / (M**2))

    # Mach correction
    if M < 1.2:
        # Transonic correction (smooth transition)
        frac = (M - 1.0) / 0.2
        CD_wave_nose *= frac

    return CD_wave_nose


def _surface_normal_force(alpha, M, num_surfaces, S_plan, AR, x_le, taper, S_ref):
    """
    Normal force coefficient and center of pressure for a lifting surface.

    Uses linear wing theory for low alpha, blends to Newtonian at high alpha.

    Parameters
    ----------
    alpha : float
        Angle of attack in radians.
    M : float
        Mach number.
    num_surfaces : float
        Number of surface panels (e.g., 2 for cruciform wings).
    S_plan : float
        Total planform area of one pair (inch^2).
    AR : float
        Aspect ratio.
    x_le : float
        Leading edge station from nose (inch).
    taper : float
        Taper ratio.
    S_ref : float
        Reference area (inch^2).

    Returns
    -------
    CN : float
        Normal force coefficient.
    Xcp : float
        Center of pressure from nose (inch).
    """
    # CN_alpha for a single pair of surfaces (per radian)
    # Supersonic linear theory with aspect ratio correction
    if M > 1.0:
        beta = np.sqrt(M**2 - 1.0)
        # Supersonic: Ackeret with AR correction
        if AR * beta > 4.0:
            CN_alpha = 4.0 / beta  # 2D Ackeret
        else:
            # Low AR supersonic
            CN_alpha = 2.0 * np.pi * AR / (2.0 + np.sqrt(4.0 + (AR * beta)**2))
    else:
        # Subsonic: Helmbold equation
        beta_sub = np.sqrt(1.0 - M**2) if M < 1.0 else 0.01
        CN_alpha = 2.0 * np.pi * AR / (2.0 + np.sqrt(4.0 + (AR * beta_sub)**2))

    # Scale for number of surfaces and area ratio
    CN_alpha_total = CN_alpha * (num_surfaces * S_plan / (2.0 * S_ref))

    # Blend linear and Newtonian at high alpha
    CN_linear = CN_alpha_total * alpha

    # Newtonian: CN = 2 * sin^2(alpha) * cos(alpha) * S/S_ref
    CN_newton = 2.0 * np.sin(alpha)**2 * np.cos(alpha) * (num_surfaces * S_plan / (2.0 * S_ref))

    # Crossover around 15 deg
    alpha_blend = 15.0 * np.pi / 180.0
    if abs(alpha) < alpha_blend:
        CN = CN_linear
    else:
        # Smooth blend
        w = min((abs(alpha) - alpha_blend) / (15.0 * np.pi / 180.0), 1.0)
        CN = (1.0 - w) * CN_linear + w * CN_newton
        if alpha < 0:
            CN = -abs(CN)

    # Center of pressure: MAC quarter-chord location
    span = np.sqrt(AR * S_plan)
    c_root = 2.0 * S_plan / (span * (1.0 + taper)) if (span * (1.0 + taper)) > 0 else 1.0
    c_mac = c_root * (2.0 / 3.0) * (1.0 + taper + taper**2) / (1.0 + taper)

    # MAC LE offset from wing root LE
    if abs(1.0 + taper) > 0:
        y_mac = (span / 2.0) * (1.0 + 2.0 * taper) / (3.0 * (1.0 + taper))
    else:
        y_mac = span / 6.0

    # Approximate MAC LE station (assuming straight leading edge)
    # For trapezoidal wing, sweep of LE:
    if span > 0:
        sweep_le = np.arctan2(c_root * (1.0 - taper), span)
    else:
        sweep_le = 0.0
    x_mac_le = x_le + y_mac * np.tan(sweep_le)

    # CP at quarter-chord of MAC
    Xcp = x_mac_le + 0.25 * c_mac

    return CN, Xcp
