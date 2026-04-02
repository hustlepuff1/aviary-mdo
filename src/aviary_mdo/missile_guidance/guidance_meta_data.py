"""
Extended metadata for the missile guidance subsystem.

Registers all guidance-specific variables with Aviary's metadata system.
Default values are representative of an SM-2 class surface-to-air missile
engagement against a maneuvering aircraft target.
"""

import aviary.api as av
from aviary_mdo.missile_guidance.guidance_variables import (
    Guidance, DynamicGuidance
)

ExtendedMetaDataGuidance = av.CoreMetaData

# ===== Missile State =====

av.add_meta_data(Guidance.Missile.POSITION_X, units='m',
                 desc='Missile initial X position', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.POSITION_Y, units='m',
                 desc='Missile initial Y position', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.VELOCITY, units='m/s',
                 desc='Missile initial velocity', default_value=1000.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.HEADING, units='deg',
                 desc='Missile initial heading (0=East, 90=North)', default_value=45.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.MAX_ACCELERATION, units='m/s**2',
                 desc='Maximum lateral acceleration capability', default_value=300.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.DRAG_COEFFICIENT, units=None,
                 desc='Missile zero-lift drag coefficient', default_value=0.4,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.REFERENCE_AREA, units='m**2',
                 desc='Missile reference area', default_value=0.0324,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.MASS, units='kg',
                 desc='Missile mass at engagement start', default_value=700.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.THRUST, units='N',
                 desc='Missile thrust during sustain (0 if coasting)', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Missile.BURN_TIME, units='s',
                 desc='Remaining sustain burn time', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Target State =====

av.add_meta_data(Guidance.Target.POSITION_X, units='m',
                 desc='Target initial X position', default_value=15000.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Target.POSITION_Y, units='m',
                 desc='Target initial Y position', default_value=15000.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Target.VELOCITY, units='m/s',
                 desc='Target velocity', default_value=300.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Target.HEADING, units='deg',
                 desc='Target heading (0=East, 90=North)', default_value=180.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Target.ACCELERATION_MAX, units='m/s**2',
                 desc='Max target maneuver acceleration', default_value=50.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Target.MANEUVER_TIME, units='s',
                 desc='Time after engagement start when target begins maneuvering',
                 default_value=5.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Navigation Law =====

av.add_meta_data(Guidance.Navigation.LAW, units=None,
                 desc='Guidance law: PN, APN, TPN, OGL', default_value='PN',
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Navigation.RATIO, units=None,
                 desc='Navigation ratio N-prime (3-5 typical)', default_value=4.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Autopilot =====

av.add_meta_data(Guidance.Autopilot.TIME_CONSTANT, units='s',
                 desc='Autopilot + actuator combined time constant',
                 default_value=0.2,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Autopilot.RATE_LIMIT, units='deg/s',
                 desc='Fin deflection rate limit', default_value=200.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Autopilot.DEFLECTION_LIMIT, units='deg',
                 desc='Max fin deflection angle', default_value=25.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Seeker =====

av.add_meta_data(Guidance.Seeker.NOISE_SIGMA, units='rad/s',
                 desc='LOS rate measurement noise (1-sigma)', default_value=1e-4,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Seeker.GIMBAL_LIMIT, units='deg',
                 desc='Seeker gimbal angle limit', default_value=60.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Seeker.SAMPLE_RATE, units='Hz',
                 desc='Seeker measurement rate', default_value=50.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Engagement Geometry (computed) =====

av.add_meta_data(Guidance.Engagement.INITIAL_RANGE, units='m',
                 desc='Initial missile-to-target range', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Engagement.CLOSING_VELOCITY, units='m/s',
                 desc='Closing velocity (positive when closing)', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Engagement.LOS_ANGLE, units='rad',
                 desc='Initial line-of-sight angle', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Engagement.LOS_RATE, units='rad/s',
                 desc='Initial LOS rate', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Engagement.TIME_TO_GO, units='s',
                 desc='Estimated time to intercept', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Engagement.HEADING_ERROR, units='deg',
                 desc='Initial heading error to collision course', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Environment =====

av.add_meta_data(Guidance.Environment.ALTITUDE, units='m',
                 desc='Engagement altitude', default_value=6096.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(Guidance.Environment.DENSITY, units='kg/m**3',
                 desc='Air density at engagement altitude', default_value=0.66,
                 meta_data=ExtendedMetaDataGuidance)

# ===== Dynamic Outputs (Engagement Results) =====

av.add_meta_data(DynamicGuidance.MISS_DISTANCE, units='m',
                 desc='Closest approach distance (miss distance)', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.MAX_LATERAL_ACCEL, units='m/s**2',
                 desc='Maximum achieved lateral acceleration', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.MAX_COMMANDED_ACCEL, units='m/s**2',
                 desc='Maximum commanded acceleration (may exceed capability)',
                 default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.TIME_TO_INTERCEPT, units='s',
                 desc='Actual time to closest approach', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.ZERO_EFFORT_MISS, units='m',
                 desc='Zero effort miss distance at engagement start',
                 default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.DIVERT_REQUIREMENT, units='m/s',
                 desc='Total velocity change from guidance commands',
                 default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.CLOSEST_APPROACH, units='m',
                 desc='Minimum range during engagement', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.REQUIRED_LOAD_FACTOR, units=None,
                 desc='Peak load factor required (g-units)', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.SINGLE_SHOT_PK, units=None,
                 desc='Single-shot probability of kill (0-1)', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.INTERCEPT_ACHIEVED, units=None,
                 desc='Whether intercept was achieved (1=yes, 0=no)',
                 default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.FINAL_RANGE, units='m',
                 desc='Range at simulation end', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)

av.add_meta_data(DynamicGuidance.FLIGHT_TIME, units='s',
                 desc='Total engagement flight time', default_value=0.0,
                 meta_data=ExtendedMetaDataGuidance)
