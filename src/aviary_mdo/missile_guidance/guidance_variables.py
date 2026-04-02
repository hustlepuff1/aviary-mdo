"""
Variable hierarchy for the missile guidance subsystem.

Follows Aviary's variable naming convention. Covers engagement
geometry, guidance law parameters, autopilot/seeker models, and
engagement performance outputs.

Reference equations from:
- Zarchan, "Tactical and Strategic Missile Guidance"
- Fleeman, "Tactical Missile Design" (Ch. 8 Guidance)
- Engelsman, "Advanced Topics in Missile Guidance" (Technion)
"""


class Guidance:
    """Missile guidance pre-mission / sizing variables."""

    class Missile:
        POSITION_X = 'guidance:missile:position_x'
        POSITION_Y = 'guidance:missile:position_y'
        VELOCITY = 'guidance:missile:velocity'
        HEADING = 'guidance:missile:heading'
        MAX_ACCELERATION = 'guidance:missile:max_acceleration'
        DRAG_COEFFICIENT = 'guidance:missile:drag_coefficient'
        REFERENCE_AREA = 'guidance:missile:reference_area'
        MASS = 'guidance:missile:mass'
        THRUST = 'guidance:missile:thrust'
        BURN_TIME = 'guidance:missile:burn_time'

    class Target:
        POSITION_X = 'guidance:target:position_x'
        POSITION_Y = 'guidance:target:position_y'
        VELOCITY = 'guidance:target:velocity'
        HEADING = 'guidance:target:heading'
        ACCELERATION_MAX = 'guidance:target:acceleration_max'
        MANEUVER_TIME = 'guidance:target:maneuver_time'

    class Navigation:
        LAW = 'guidance:nav:law'
        RATIO = 'guidance:nav:ratio'

    class Autopilot:
        TIME_CONSTANT = 'guidance:autopilot:time_constant'
        RATE_LIMIT = 'guidance:autopilot:rate_limit'
        DEFLECTION_LIMIT = 'guidance:autopilot:deflection_limit'

    class Seeker:
        NOISE_SIGMA = 'guidance:seeker:noise_sigma'
        GIMBAL_LIMIT = 'guidance:seeker:gimbal_limit'
        SAMPLE_RATE = 'guidance:seeker:sample_rate'

    class Engagement:
        INITIAL_RANGE = 'guidance:engagement:initial_range'
        CLOSING_VELOCITY = 'guidance:engagement:closing_velocity'
        LOS_ANGLE = 'guidance:engagement:los_angle'
        LOS_RATE = 'guidance:engagement:los_rate'
        TIME_TO_GO = 'guidance:engagement:time_to_go'
        HEADING_ERROR = 'guidance:engagement:heading_error'

    class Environment:
        ALTITUDE = 'guidance:env:altitude'
        DENSITY = 'guidance:env:density'


class DynamicGuidance:
    """Engagement performance outputs."""

    MISS_DISTANCE = 'dynamic:guidance:miss_distance'
    MAX_LATERAL_ACCEL = 'dynamic:guidance:max_lateral_accel'
    MAX_COMMANDED_ACCEL = 'dynamic:guidance:max_commanded_accel'
    TIME_TO_INTERCEPT = 'dynamic:guidance:time_to_intercept'
    ZERO_EFFORT_MISS = 'dynamic:guidance:zero_effort_miss'
    DIVERT_REQUIREMENT = 'dynamic:guidance:divert_requirement'
    CLOSEST_APPROACH = 'dynamic:guidance:closest_approach'
    REQUIRED_LOAD_FACTOR = 'dynamic:guidance:required_load_factor'
    SINGLE_SHOT_PK = 'dynamic:guidance:single_shot_pk'
    INTERCEPT_ACHIEVED = 'dynamic:guidance:intercept_achieved'
    FINAL_RANGE = 'dynamic:guidance:final_range'
    FLIGHT_TIME = 'dynamic:guidance:flight_time'
