"""
Extended metadata for the Aerothermodynamics subsystem.

Registers all aerothermo-specific variables with Aviary's metadata system
so they can be used with add_aviary_input/add_aviary_output.
"""

import aviary.api as av
from aviary_mdo.aerothermodynamics.aerothermo_variables import Aerothermo

ExtendedMetaData = av.CoreMetaData

# ===== Design Inputs =====

av.add_meta_data(Aerothermo.Design.NOSE_RADIUS, units='m',
                 desc='Vehicle nose radius of curvature', default_value=0.5,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.BODY_LENGTH, units='m',
                 desc='Vehicle body length', default_value=5.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.WETTED_AREA, units='m**2',
                 desc='Vehicle wetted area exposed to aerodynamic heating',
                 default_value=20.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.DESIGN_MACH, units=None,
                 desc='Design Mach number for aerothermal sizing',
                 default_value=5.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.DESIGN_ALTITUDE, units='m',
                 desc='Design altitude for aerothermal sizing',
                 default_value=30000.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.WALL_TEMPERATURE, units='K',
                 desc='Specified wall temperature for heat transfer',
                 default_value=300.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Design.TPS_TYPE, units=None,
                 desc='TPS material type (1=ablative, 2=ceramic, 3=metallic)',
                 default_value=1.0,
                 meta_data=ExtendedMetaData)

# ===== Outputs =====

av.add_meta_data(Aerothermo.Output.Q_STAG, units='W/m**2',
                 desc='Stagnation point heat flux', default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Output.T_WALL_EQ, units='K',
                 desc='Radiative equilibrium wall temperature',
                 default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Output.TPS_THICKNESS, units='m',
                 desc='Required TPS thickness', default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Output.TPS_MASS, units='kg',
                 desc='Total TPS mass', default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Aerothermo.Output.TOTAL_HEAT_LOAD, units='W',
                 desc='Total integrated heat load over vehicle surface',
                 default_value=0.0,
                 meta_data=ExtendedMetaData)
