"""
Variable hierarchy for the RocketPy 6-DOF missile subsystem.

Follows Aviary's variable naming convention. Variables map to RocketPy
constructor parameters and flight simulation outputs.
"""


class Missile6DOF:
    """Missile 6-DOF pre-mission / sizing variables."""

    class Body:
        LENGTH = 'missile_6dof:body:length'
        RADIUS = 'missile_6dof:body:radius'
        MASS = 'missile_6dof:body:mass'
        INERTIA_I = 'missile_6dof:body:inertia_i'
        INERTIA_Z = 'missile_6dof:body:inertia_z'
        CG_WITHOUT_MOTOR = 'missile_6dof:body:cg_without_motor'

    class NoseCone:
        LENGTH = 'missile_6dof:nose:length'
        KIND = 'missile_6dof:nose:kind'
        BASE_RADIUS = 'missile_6dof:nose:base_radius'
        BLUFFNESS = 'missile_6dof:nose:bluffness'

    class Fins:
        NUMBER = 'missile_6dof:fins:number'
        ROOT_CHORD = 'missile_6dof:fins:root_chord'
        TIP_CHORD = 'missile_6dof:fins:tip_chord'
        SPAN = 'missile_6dof:fins:span'
        POSITION = 'missile_6dof:fins:position'
        CANT_ANGLE = 'missile_6dof:fins:cant_angle'
        SWEEP_LENGTH = 'missile_6dof:fins:sweep_length'

    class Tail:
        TOP_RADIUS = 'missile_6dof:tail:top_radius'
        BOTTOM_RADIUS = 'missile_6dof:tail:bottom_radius'
        LENGTH = 'missile_6dof:tail:length'
        POSITION = 'missile_6dof:tail:position'

    class Motor:
        THRUST_SOURCE = 'missile_6dof:motor:thrust_source'
        BURN_TIME = 'missile_6dof:motor:burn_time'
        DRY_MASS = 'missile_6dof:motor:dry_mass'
        DRY_INERTIA_I = 'missile_6dof:motor:dry_inertia_i'
        DRY_INERTIA_Z = 'missile_6dof:motor:dry_inertia_z'
        CENTER_OF_DRY_MASS = 'missile_6dof:motor:center_of_dry_mass'
        NOZZLE_RADIUS = 'missile_6dof:motor:nozzle_radius'
        NOZZLE_POSITION = 'missile_6dof:motor:nozzle_position'

    class SolidGrain:
        PROPELLANT_DENSITY = 'missile_6dof:grain:propellant_density'
        NUMBER = 'missile_6dof:grain:number'
        OUTER_RADIUS = 'missile_6dof:grain:outer_radius'
        INNER_RADIUS = 'missile_6dof:grain:inner_radius'
        HEIGHT = 'missile_6dof:grain:height'

    class Environment:
        LATITUDE = 'missile_6dof:env:latitude'
        LONGITUDE = 'missile_6dof:env:longitude'
        ELEVATION = 'missile_6dof:env:elevation'

    class Launch:
        RAIL_LENGTH = 'missile_6dof:launch:rail_length'
        INCLINATION = 'missile_6dof:launch:inclination'
        HEADING = 'missile_6dof:launch:heading'

    class Atmosphere:
        ALTITUDE = 'missile_6dof:atm:altitude'
        DENSITY = 'missile_6dof:atm:density'
        TEMPERATURE = 'missile_6dof:atm:temperature'
        PRESSURE = 'missile_6dof:atm:pressure'
        SPEED_OF_SOUND = 'missile_6dof:atm:speed_of_sound'
        DYNAMIC_VISCOSITY = 'missile_6dof:atm:dynamic_viscosity'


class DynamicMissile6DOF:
    """Time-varying 6-DOF flight outputs."""

    # Trajectory outputs
    APOGEE = 'dynamic:missile_6dof:apogee'
    APOGEE_TIME = 'dynamic:missile_6dof:apogee_time'
    MAX_SPEED = 'dynamic:missile_6dof:max_speed'
    MAX_MACH = 'dynamic:missile_6dof:max_mach'
    MAX_ACCELERATION = 'dynamic:missile_6dof:max_acceleration'
    MAX_DYNAMIC_PRESSURE = 'dynamic:missile_6dof:max_dynamic_pressure'
    FLIGHT_TIME = 'dynamic:missile_6dof:flight_time'
    IMPACT_VELOCITY = 'dynamic:missile_6dof:impact_velocity'
    RANGE_X = 'dynamic:missile_6dof:range_x'
    RANGE_Y = 'dynamic:missile_6dof:range_y'
    RANGE_TOTAL = 'dynamic:missile_6dof:range_total'

    # Motor performance outputs
    TOTAL_IMPULSE = 'dynamic:missile_6dof:total_impulse'
    MAX_THRUST = 'dynamic:missile_6dof:max_thrust'
    AVERAGE_THRUST = 'dynamic:missile_6dof:average_thrust'
    AVERAGE_ISP = 'dynamic:missile_6dof:average_isp'
    PROPELLANT_MASS = 'dynamic:missile_6dof:propellant_mass'

    # Stability outputs
    STATIC_MARGIN_AT_IGNITION = 'dynamic:missile_6dof:static_margin_ignition'
    STATIC_MARGIN_AT_BURNOUT = 'dynamic:missile_6dof:static_margin_burnout'
    CP_POSITION = 'dynamic:missile_6dof:cp_position'
    CG_POSITION = 'dynamic:missile_6dof:cg_position'

    # Aero outputs
    POWER_ON_CD = 'dynamic:missile_6dof:power_on_cd'
    POWER_OFF_CD = 'dynamic:missile_6dof:power_off_cd'
