"""
Variable hierarchy for the Tactical Missile Design subsystem.

Follows Aviary's variable naming convention. Organized by discipline
matching the TMD spreadsheet structure.
"""


class Missile:
    """Missile-level (pre-mission / sizing) variables."""

    class Body:
        LENGTH = 'missile:body:length'
        DIAMETER_MAJOR = 'missile:body:diameter_major'
        DIAMETER_MINOR = 'missile:body:diameter_minor'
        NOSE_LENGTH = 'missile:body:nose_length'
        NOSE_BLUNTNESS = 'missile:body:nose_bluntness'
        FINENESS_RATIO = 'missile:body:fineness_ratio'
        REFERENCE_AREA = 'missile:body:reference_area'
        CG_STATION = 'missile:body:cg_station'

    class Wing:
        NUM_WINGS = 'missile:wing:num_wings'
        PLANFORM_AREA = 'missile:wing:planform_area'
        ASPECT_RATIO = 'missile:wing:aspect_ratio'
        TAPER_RATIO = 'missile:wing:taper_ratio'
        LE_STATION = 'missile:wing:le_station'
        THICKNESS_TO_CHORD = 'missile:wing:thickness_to_chord'
        LE_THICKNESS_ANGLE = 'missile:wing:le_thickness_angle'
        INCIDENT_ANGLE = 'missile:wing:incident_angle'

    class Tail:
        NUM_TAILS = 'missile:tail:num_tails'
        ASPECT_RATIO = 'missile:tail:aspect_ratio'
        TAPER_RATIO = 'missile:tail:taper_ratio'
        LE_STATION = 'missile:tail:le_station'
        THICKNESS_TO_CHORD = 'missile:tail:thickness_to_chord'
        LE_THICKNESS_ANGLE = 'missile:tail:le_thickness_angle'
        AREA = 'missile:tail:area'

    class Weight:
        LAUNCH = 'missile:weight:launch'
        BOOST_PROPELLANT = 'missile:weight:boost_propellant'
        CRUISE_PROPELLANT = 'missile:weight:cruise_propellant'
        EJECTABLES = 'missile:weight:ejectables'
        EMPTY = 'missile:weight:empty'

    class Propulsion:
        # Boost motor
        BOOST_ENGINE_TYPE = 'missile:propulsion:boost_engine_type'
        BOOST_FUEL_TYPE = 'missile:propulsion:boost_fuel_type'
        BOOST_CHAMBER_PRESSURE = 'missile:propulsion:boost_chamber_pressure'
        BOOST_EXPANSION_RATIO = 'missile:propulsion:boost_expansion_ratio'
        BOOST_BURN_TIME = 'missile:propulsion:boost_burn_time'
        BOOST_THRUST = 'missile:propulsion:boost_thrust'
        BOOST_ISP = 'missile:propulsion:boost_isp'
        BOOST_MASS_FLOW = 'missile:propulsion:boost_mass_flow'
        BOOST_THROAT_AREA = 'missile:propulsion:boost_throat_area'
        BOOST_EXIT_AREA = 'missile:propulsion:boost_exit_area'

        # Sustain motor
        SUSTAIN_ENGINE_TYPE = 'missile:propulsion:sustain_engine_type'
        SUSTAIN_FUEL_TYPE = 'missile:propulsion:sustain_fuel_type'
        SUSTAIN_CHAMBER_PRESSURE = 'missile:propulsion:sustain_chamber_pressure'
        SUSTAIN_EXPANSION_RATIO = 'missile:propulsion:sustain_expansion_ratio'
        SUSTAIN_BURN_TIME = 'missile:propulsion:sustain_burn_time'
        SUSTAIN_THRUST = 'missile:propulsion:sustain_thrust'
        SUSTAIN_ISP = 'missile:propulsion:sustain_isp'

        # Ramjet-specific
        FUEL_AIR_RATIO = 'missile:propulsion:fuel_air_ratio'
        MAX_COMBUSTOR_TEMP = 'missile:propulsion:max_combustor_temp'

    class Flight:
        LAUNCH_MACH = 'missile:flight:launch_mach'
        LAUNCH_ALTITUDE = 'missile:flight:launch_altitude'
        TERMINAL_MACH = 'missile:flight:terminal_mach'


class DynamicMissile:
    """Time-varying mission variables."""

    MACH = 'dynamic:missile:mach'
    VELOCITY = 'dynamic:missile:velocity'
    ALTITUDE = 'dynamic:missile:altitude'
    WEIGHT = 'dynamic:missile:weight'
    RANGE = 'dynamic:missile:range'
    TIME = 'dynamic:missile:time'
    CD0 = 'dynamic:missile:cd0'
    CN = 'dynamic:missile:cn'
    THRUST = 'dynamic:missile:thrust'
    ISP = 'dynamic:missile:isp'
    DRAG = 'dynamic:missile:drag'
