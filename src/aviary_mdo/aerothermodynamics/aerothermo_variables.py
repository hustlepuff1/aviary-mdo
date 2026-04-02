"""
Variable hierarchy for the Aerothermodynamics subsystem.

Follows Aviary's variable naming convention. Organized by discipline
for stagnation heating, surface heat flux, and TPS sizing.
"""


class Aerothermo:
    """Aerothermodynamics design and sizing variables."""

    class Design:
        NOSE_RADIUS = 'aerothermo:design:nose_radius'
        BODY_LENGTH = 'aerothermo:design:body_length'
        WETTED_AREA = 'aerothermo:design:wetted_area'
        DESIGN_MACH = 'aerothermo:design:design_mach'
        DESIGN_ALTITUDE = 'aerothermo:design:design_altitude'
        WALL_TEMPERATURE = 'aerothermo:design:wall_temperature'
        TPS_TYPE = 'aerothermo:design:tps_type'

    class Output:
        Q_STAG = 'aerothermo:output:q_stag'
        T_WALL_EQ = 'aerothermo:output:t_wall_eq'
        TPS_THICKNESS = 'aerothermo:output:tps_thickness'
        TPS_MASS = 'aerothermo:output:tps_mass'
        TOTAL_HEAT_LOAD = 'aerothermo:output:total_heat_load'
