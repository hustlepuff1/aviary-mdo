"""
TMD-to-JSBSim XML translator.

Converts TMD missile design parameters into a JSBSim aircraft definition
XML file that can be simulated with JSBSim's 6DOF flight dynamics model.

The generated XML follows JSBSim's fdm_config format and includes:
- metrics (reference area, wingspan/diameter, chord)
- mass_balance (empty weight, CG, moments of inertia)
- aerodynamics (CD vs Mach table from TMD drag buildup, CN_alpha)
- propulsion (solid rocket motor with constant thrust)
- flight_control (no active control — ballistic flight)

Usage:
    from aviary_mdo.missile.jsbsim_bridge.tmd_to_jsbsim import (
        TMDToJSBSim, generate_reset_file
    )

    translator = TMDToJSBSim()
    translator.set_geometry(length=143.9, diameter=8.0, nose_length=19.2)
    translator.set_mass(launch_weight=500.0, cg_station=76.2, Iy=94.0)
    translator.set_aero_table(machs, cd0_values)
    translator.set_propulsion(thrust=7036.0, isp=271.0, burn_time=3.26,
                               propellant_weight=84.8)
    translator.write_xml('/path/to/missile.xml')
"""

import os
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


class TMDToJSBSim:
    """Translates TMD missile parameters into JSBSim XML format."""

    def __init__(self, name='tmd_missile'):
        self.name = name
        self.geometry = {}
        self.mass = {}
        self.aero = {}
        self.propulsion = {}

    def set_geometry(self, length=143.9, diameter=8.0, nose_length=19.2,
                     wing_area=400.0, num_wings=2):
        """Set missile geometry (all in inches)."""
        self.geometry = {
            'length': length,
            'diameter': diameter,
            'nose_length': nose_length,
            'wing_area': wing_area,
            'num_wings': num_wings,
            'ref_area': np.pi / 4.0 * diameter**2,  # body cross-section
        }

    def set_mass(self, launch_weight=500.0, empty_weight=367.0,
                 cg_station=76.2, Iy=94.0):
        """Set mass properties (weight in lbm, CG in inches, Iy in slug-ft^2)."""
        self.mass = {
            'launch_weight': launch_weight,
            'empty_weight': empty_weight,
            'cg_station': cg_station,
            'Ixx': 0.66,  # roll inertia (small for axisymmetric body)
            'Iyy': Iy,
            'Izz': Iy,  # symmetric
        }

    def set_aero_table(self, machs=None, cd0_values=None, cn_alpha=38.0):
        """
        Set aerodynamic data.

        Parameters
        ----------
        machs : array-like
            Mach numbers for CD table.
        cd0_values : array-like
            Zero-lift drag coefficient at each Mach.
        cn_alpha : float
            Normal force slope (per radian), referenced to body cross-section.
        """
        if machs is None:
            # Default CD0 vs Mach table (typical for Sparrow-class missile)
            machs = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05,
                     1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
            cd0_values = [0.28, 0.28, 0.29, 0.31, 0.33, 0.38, 0.42, 0.44,
                          0.45, 0.46, 0.44, 0.42, 0.40, 0.38, 0.34, 0.31]

        self.aero = {
            'machs': list(machs),
            'cd0': list(cd0_values),
            'cn_alpha': cn_alpha,
        }

    def set_propulsion(self, thrust=7036.0, isp=271.0, burn_time=3.26,
                       propellant_weight=84.8, sustain_thrust=1119.0,
                       sustain_isp=252.0, sustain_burn_time=10.86,
                       sustain_propellant_weight=48.2):
        """Set rocket motor parameters (thrust in lbf, Isp in sec, etc.)."""
        self.propulsion = {
            'boost_thrust': thrust,
            'boost_isp': isp,
            'boost_burn_time': burn_time,
            'boost_propellant': propellant_weight,
            'sustain_thrust': sustain_thrust,
            'sustain_isp': sustain_isp,
            'sustain_burn_time': sustain_burn_time,
            'sustain_propellant': sustain_propellant_weight,
        }

    def build_xml(self):
        """Build the JSBSim XML Element tree."""
        root = Element('fdm_config')
        root.set('name', self.name)
        root.set('version', '2.0')
        root.set('release', 'BETA')

        # File header
        header = SubElement(root, 'fileheader')
        SubElement(header, 'author').text = 'TMD-to-JSBSim Translator'
        SubElement(header, 'filecreationdate').text = '2026-03-28'
        SubElement(header, 'description').text = (
            f'Tactical missile generated from TMD parameters. '
            f'Length={self.geometry.get("length", 0):.1f} in, '
            f'Diameter={self.geometry.get("diameter", 0):.1f} in, '
            f'Weight={self.mass.get("launch_weight", 0):.0f} lbm'
        )

        # Metrics
        self._add_metrics(root)

        # Mass balance
        self._add_mass_balance(root)

        # Ground reactions (minimal — missile doesn't land)
        self._add_ground_reactions(root)

        # Propulsion
        self._add_propulsion(root)

        # Flight control (none — ballistic/guided)
        self._add_flight_control(root)

        # Aerodynamics
        self._add_aerodynamics(root)

        return root

    def write_xml(self, filepath):
        """Write the JSBSim XML to file."""
        root = self.build_xml()
        rough_string = tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent='  ')

        # Remove extra XML declaration if present
        lines = pretty.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0"?>'

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    def _add_metrics(self, root):
        geom = self.geometry
        d = geom.get('diameter', 8.0)
        ref_area = geom.get('ref_area', np.pi / 4.0 * d**2)
        ref_area_ft2 = ref_area / 144.0  # in^2 to ft^2

        metrics = SubElement(root, 'metrics')
        SubElement(metrics, 'wingarea', unit='FT2').text = f'{ref_area_ft2:.4f}'
        SubElement(metrics, 'wingspan', unit='FT').text = f'{d / 12.0:.4f}'
        SubElement(metrics, 'chord', unit='FT').text = f'{d / 12.0:.4f}'
        SubElement(metrics, 'htailarea', unit='FT2').text = '0'
        SubElement(metrics, 'htailarm', unit='FT').text = '0'
        SubElement(metrics, 'vtailarea', unit='FT2').text = '0'
        SubElement(metrics, 'vtailarm', unit='FT').text = '0'

        # Aero reference point at CG
        cg = self.mass.get('cg_station', geom.get('length', 143.9) / 2.0)
        aero_rp = SubElement(metrics, 'location', name='AERORP', unit='IN')
        SubElement(aero_rp, 'x').text = f'{cg:.1f}'
        SubElement(aero_rp, 'y').text = '0'
        SubElement(aero_rp, 'z').text = '0'

        eye = SubElement(metrics, 'location', name='EYEPOINT', unit='IN')
        SubElement(eye, 'x').text = '0'
        SubElement(eye, 'y').text = '0'
        SubElement(eye, 'z').text = '0'

        vrp = SubElement(metrics, 'location', name='VRP', unit='IN')
        SubElement(vrp, 'x').text = f'{cg:.1f}'
        SubElement(vrp, 'y').text = '0'
        SubElement(vrp, 'z').text = '0'

    def _add_mass_balance(self, root):
        mass = self.mass
        mb = SubElement(root, 'mass_balance')
        SubElement(mb, 'ixx', unit='SLUG*FT2').text = f'{mass.get("Ixx", 0.66):.2f}'
        SubElement(mb, 'iyy', unit='SLUG*FT2').text = f'{mass.get("Iyy", 94.0):.2f}'
        SubElement(mb, 'izz', unit='SLUG*FT2').text = f'{mass.get("Izz", 94.0):.2f}'
        SubElement(mb, 'emptywt', unit='LBS').text = f'{mass.get("empty_weight", 367.0):.1f}'

        cg = SubElement(mb, 'location', name='CG', unit='IN')
        SubElement(cg, 'x').text = f'{mass.get("cg_station", 76.2):.1f}'
        SubElement(cg, 'y').text = '0'
        SubElement(cg, 'z').text = '0'

    def _add_ground_reactions(self, root):
        gr = SubElement(root, 'ground_reactions')
        for name, x_pos in [('NOSE', 0), ('TAIL', self.geometry.get('length', 143.9))]:
            contact = SubElement(gr, 'contact', type='STRUCTURE', name=f'{name}_CONTACT')
            loc = SubElement(contact, 'location', unit='IN')
            SubElement(loc, 'x').text = f'{x_pos:.0f}'
            SubElement(loc, 'y').text = '0'
            SubElement(loc, 'z').text = '0'
            SubElement(contact, 'static_friction').text = '0'
            SubElement(contact, 'dynamic_friction').text = '0'
            SubElement(contact, 'spring_coeff', unit='LBS/FT').text = '10000'
            SubElement(contact, 'damping_coeff', unit='LBS/FT/SEC').text = '200000'
            SubElement(contact, 'brake_group').text = 'NONE'
            SubElement(contact, 'retractable').text = '0'

    def _add_propulsion(self, root):
        prop = self.propulsion
        propulsion = SubElement(root, 'propulsion')

        # Boost motor — modeled as a solid rocket
        engine = SubElement(propulsion, 'engine', file='boost_motor')
        engine.set('name', 'Boost Motor')
        loc = SubElement(engine, 'location', unit='IN')
        tail_x = self.geometry.get('length', 143.9)
        SubElement(loc, 'x').text = f'{tail_x:.0f}'
        SubElement(loc, 'y').text = '0'
        SubElement(loc, 'z').text = '0'
        SubElement(engine, 'feed').text = '0'

        # Propellant tank (boost)
        tank = SubElement(propulsion, 'tank', type='FUEL')
        tloc = SubElement(tank, 'location', unit='IN')
        SubElement(tloc, 'x').text = f'{tail_x - 30:.0f}'
        SubElement(tloc, 'y').text = '0'
        SubElement(tloc, 'z').text = '0'
        SubElement(tank, 'capacity', unit='LBS').text = (
            f'{prop.get("boost_propellant", 84.8) + prop.get("sustain_propellant", 48.2):.1f}')
        SubElement(tank, 'contents', unit='LBS').text = (
            f'{prop.get("boost_propellant", 84.8) + prop.get("sustain_propellant", 48.2):.1f}')

    def _add_flight_control(self, root):
        fcs = SubElement(root, 'flight_control', name=f'FCS: {self.name}')
        # No active control surfaces — ballistic flight
        # Could add guidance channels here in the future

    def _add_aerodynamics(self, root):
        aero_data = self.aero
        aero = SubElement(root, 'aerodynamics')

        # DRAG axis — CD vs Mach table
        drag_axis = SubElement(aero, 'axis', name='DRAG')
        cd_func = SubElement(drag_axis, 'function', name='aero/coefficient/CDmin')
        SubElement(cd_func, 'description').text = 'Drag_minimum from TMD buildup'
        product = SubElement(cd_func, 'product')
        SubElement(product, 'property').text = 'aero/qbar-psf'
        SubElement(product, 'property').text = 'metrics/Sw-sqft'

        table = SubElement(product, 'table')
        SubElement(table, 'independentVar').text = 'velocities/mach'
        td = SubElement(table, 'tableData')

        machs = aero_data.get('machs', [0.8, 1.0, 1.5, 2.0, 2.5])
        cd0s = aero_data.get('cd0', [0.31, 0.42, 0.44, 0.38, 0.34])
        lines = []
        for m, cd in zip(machs, cd0s):
            lines.append(f'{m:.4f}\t{cd:.4f}')
        td.text = '\n' + '\n'.join(lines) + '\n'

        # SIDE axis (zero for axisymmetric)
        side_axis = SubElement(aero, 'axis', name='SIDE')

        # LIFT axis — CN_alpha contribution (normal force as lift in body frame)
        lift_axis = SubElement(aero, 'axis', name='LIFT')
        cl_func = SubElement(lift_axis, 'function', name='aero/coefficient/CLalpha')
        SubElement(cl_func, 'description').text = 'Lift due to alpha from TMD normal force'
        prod2 = SubElement(cl_func, 'product')
        SubElement(prod2, 'property').text = 'aero/qbar-psf'
        SubElement(prod2, 'property').text = 'metrics/Sw-sqft'
        SubElement(prod2, 'property').text = 'aero/alpha-rad'
        cn_alpha = aero_data.get('cn_alpha', 38.0)
        # Convert CN_alpha (per radian) to CL_alpha considering ref area
        SubElement(prod2, 'value').text = f'{cn_alpha:.2f}'


def generate_reset_file(filepath, mach=0.8, altitude=20000.0, heading=0.0,
                        gamma=0.0):
    """
    Generate a JSBSim reset (initial condition) file.

    Parameters
    ----------
    filepath : str
        Path to write the reset XML.
    mach : float
        Initial Mach number.
    altitude : float
        Initial altitude in feet.
    heading : float
        Initial heading in degrees.
    gamma : float
        Initial flight path angle in degrees.
    """
    root = Element('initialize', name='reset00')

    SubElement(root, 'ubody', unit='MACH').text = f'{mach:.2f}'
    SubElement(root, 'vbody', unit='FT/SEC').text = '0.0'
    SubElement(root, 'wbody', unit='FT/SEC').text = '0.0'

    SubElement(root, 'altitude', unit='FT').text = f'{altitude:.0f}'
    SubElement(root, 'latitude', unit='DEG').text = '0.0'
    SubElement(root, 'longitude', unit='DEG').text = '0.0'

    SubElement(root, 'phi', unit='DEG').text = '0.0'
    SubElement(root, 'theta', unit='DEG').text = f'{gamma:.1f}'
    SubElement(root, 'psi', unit='DEG').text = f'{heading:.0f}'

    rough = tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent='  ')

    with open(filepath, 'w') as f:
        f.write(pretty)
