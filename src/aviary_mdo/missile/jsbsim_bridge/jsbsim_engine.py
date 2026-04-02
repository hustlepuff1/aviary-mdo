"""
JSBSim rocket engine definition generator.

Creates the engine XML file that JSBSim needs alongside the aircraft definition.
Uses JSBSim's 'rocket' engine type with constant thrust.
"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def generate_rocket_engine(filepath, thrust=7036.0, isp=271.0,
                           burn_time=3.26, propellant_weight=84.8):
    """
    Generate a JSBSim rocket engine XML file.

    Parameters
    ----------
    filepath : str
        Path to write the engine XML.
    thrust : float
        Motor thrust in lbf.
    isp : float
        Specific impulse in seconds.
    burn_time : float
        Burn time in seconds.
    propellant_weight : float
        Propellant weight in lbm.
    """
    root = Element('rocket_engine', name='boost_motor')

    SubElement(root, 'isp').text = f'{isp:.1f}'
    # JSBSim uses vacuum thrust
    SubElement(root, 'maxthrottle').text = '1.0'
    SubElement(root, 'minthrottle').text = '0.0'
    # Thrust is specified via propellant flow rate and Isp
    # mdot = thrust / (Isp * g0), g0 = 32.174 ft/s^2
    mdot = propellant_weight / burn_time  # lbm/s
    # JSBSim uses SFC in lbm/hr/lbf, but for rockets we use direct thrust
    SubElement(root, 'thrust', unit='LBS').text = f'{thrust:.1f}'

    rough = tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent='  ')

    with open(filepath, 'w') as f:
        f.write(pretty)
