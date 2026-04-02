"""
Extended metadata for the RocketPy 6-DOF missile subsystem.

Registers all 6-DOF-specific variables with Aviary's metadata system.
"""

import aviary.api as av
from aviary_mdo.missile_6dof.missile_6dof_variables import (
    Missile6DOF, DynamicMissile6DOF
)

ExtendedMetaData6DOF = av.CoreMetaData

# ===== Body Geometry =====

av.add_meta_data(Missile6DOF.Body.LENGTH, units='m',
                 desc='Total missile body length', default_value=3.658,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Body.RADIUS, units='m',
                 desc='Body outer radius', default_value=0.1016,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Body.MASS, units='kg',
                 desc='Dry airframe mass (without motor)', default_value=100.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Body.INERTIA_I, units='kg*m**2',
                 desc='Lateral moment of inertia (without motor)', default_value=50.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Body.INERTIA_Z, units='kg*m**2',
                 desc='Axial moment of inertia (without motor)', default_value=0.5,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Body.CG_WITHOUT_MOTOR, units='m',
                 desc='CG position from nose without motor', default_value=1.83,
                 meta_data=ExtendedMetaData6DOF)

# ===== Nose Cone =====

av.add_meta_data(Missile6DOF.NoseCone.LENGTH, units='m',
                 desc='Nose cone length', default_value=0.488,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.NoseCone.KIND, units=None,
                 desc='Nose cone shape: ogive, conical, elliptical, tangent, von_karman, parabolic',
                 default_value='ogive',
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.NoseCone.BASE_RADIUS, units='m',
                 desc='Nose cone base radius', default_value=0.1016,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.NoseCone.BLUFFNESS, units=None,
                 desc='Nose bluffness ratio (0=sharp, 1=flat)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

# ===== Fins =====

av.add_meta_data(Missile6DOF.Fins.NUMBER, units=None,
                 desc='Number of fin panels', default_value=4,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.ROOT_CHORD, units='m',
                 desc='Fin root chord length', default_value=0.254,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.TIP_CHORD, units='m',
                 desc='Fin tip chord length', default_value=0.076,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.SPAN, units='m',
                 desc='Fin semi-span (each panel)', default_value=0.127,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.POSITION, units='m',
                 desc='Fin root leading edge position from nose', default_value=3.2,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.CANT_ANGLE, units='deg',
                 desc='Fin cant angle', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Fins.SWEEP_LENGTH, units='m',
                 desc='Fin sweep length (LE root to LE tip)', default_value=0.127,
                 meta_data=ExtendedMetaData6DOF)

# ===== Tail =====

av.add_meta_data(Missile6DOF.Tail.TOP_RADIUS, units='m',
                 desc='Tail transition top radius', default_value=0.1016,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Tail.BOTTOM_RADIUS, units='m',
                 desc='Tail transition bottom radius', default_value=0.0762,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Tail.LENGTH, units='m',
                 desc='Tail transition length', default_value=0.1,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Tail.POSITION, units='m',
                 desc='Tail transition position from nose', default_value=3.558,
                 meta_data=ExtendedMetaData6DOF)

# ===== Motor =====

av.add_meta_data(Missile6DOF.Motor.BURN_TIME, units='s',
                 desc='Total motor burn time', default_value=3.26,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.DRY_MASS, units='kg',
                 desc='Motor dry mass (casing, nozzle)', default_value=10.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.DRY_INERTIA_I, units='kg*m**2',
                 desc='Motor dry lateral inertia', default_value=5.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.DRY_INERTIA_Z, units='kg*m**2',
                 desc='Motor dry axial inertia', default_value=0.1,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.CENTER_OF_DRY_MASS, units='m',
                 desc='Motor dry CG from motor origin', default_value=0.5,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.NOZZLE_RADIUS, units='m',
                 desc='Nozzle exit radius', default_value=0.0508,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Motor.NOZZLE_POSITION, units='m',
                 desc='Nozzle position from rocket origin (negative = aft)',
                 default_value=-1.0,
                 meta_data=ExtendedMetaData6DOF)

# ===== Solid Grain =====

av.add_meta_data(Missile6DOF.SolidGrain.PROPELLANT_DENSITY, units='kg/m**3',
                 desc='Solid propellant density', default_value=1750.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.SolidGrain.NUMBER, units=None,
                 desc='Number of grain segments', default_value=4,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.SolidGrain.OUTER_RADIUS, units='m',
                 desc='Grain outer radius', default_value=0.0889,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.SolidGrain.INNER_RADIUS, units='m',
                 desc='Grain initial inner radius (port)', default_value=0.0254,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.SolidGrain.HEIGHT, units='m',
                 desc='Height of each grain segment', default_value=0.254,
                 meta_data=ExtendedMetaData6DOF)

# ===== Environment =====

av.add_meta_data(Missile6DOF.Environment.LATITUDE, units='deg',
                 desc='Launch latitude', default_value=32.99,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Environment.LONGITUDE, units='deg',
                 desc='Launch longitude', default_value=-106.97,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Environment.ELEVATION, units='m',
                 desc='Launch site elevation above MSL', default_value=1400.0,
                 meta_data=ExtendedMetaData6DOF)

# ===== Launch Conditions =====

av.add_meta_data(Missile6DOF.Launch.RAIL_LENGTH, units='m',
                 desc='Launch rail length', default_value=5.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Launch.INCLINATION, units='deg',
                 desc='Rail inclination from horizontal (90=vertical)',
                 default_value=85.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Launch.HEADING, units='deg',
                 desc='Launch heading (0=North, 90=East)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

# ===== Atmosphere Outputs =====

av.add_meta_data(Missile6DOF.Atmosphere.ALTITUDE, units='m',
                 desc='Evaluation altitude', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Atmosphere.DENSITY, units='kg/m**3',
                 desc='Air density', default_value=1.225,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Atmosphere.TEMPERATURE, units='K',
                 desc='Air temperature', default_value=288.15,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Atmosphere.PRESSURE, units='Pa',
                 desc='Air pressure', default_value=101325.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Atmosphere.SPEED_OF_SOUND, units='m/s',
                 desc='Speed of sound', default_value=340.3,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(Missile6DOF.Atmosphere.DYNAMIC_VISCOSITY, units='Pa*s',
                 desc='Dynamic viscosity', default_value=1.789e-5,
                 meta_data=ExtendedMetaData6DOF)

# ===== Dynamic Outputs (Flight Results) =====

av.add_meta_data(DynamicMissile6DOF.APOGEE, units='m',
                 desc='Maximum altitude reached', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.APOGEE_TIME, units='s',
                 desc='Time of apogee', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.MAX_SPEED, units='m/s',
                 desc='Maximum speed during flight', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.MAX_MACH, units=None,
                 desc='Maximum Mach number', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.MAX_ACCELERATION, units='m/s**2',
                 desc='Maximum acceleration', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.MAX_DYNAMIC_PRESSURE, units='Pa',
                 desc='Maximum dynamic pressure', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.FLIGHT_TIME, units='s',
                 desc='Total flight time', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.IMPACT_VELOCITY, units='m/s',
                 desc='Velocity at impact', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.RANGE_X, units='m',
                 desc='Downrange distance (x)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.RANGE_Y, units='m',
                 desc='Crossrange distance (y)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.RANGE_TOTAL, units='m',
                 desc='Total ground range', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.TOTAL_IMPULSE, units='N*s',
                 desc='Motor total impulse', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.MAX_THRUST, units='N',
                 desc='Motor maximum thrust', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.AVERAGE_THRUST, units='N',
                 desc='Motor average thrust', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.AVERAGE_ISP, units='s',
                 desc='Motor average specific impulse', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.PROPELLANT_MASS, units='kg',
                 desc='Total propellant mass', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.STATIC_MARGIN_AT_IGNITION, units=None,
                 desc='Static margin at ignition (calibers)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.STATIC_MARGIN_AT_BURNOUT, units=None,
                 desc='Static margin at burnout (calibers)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.CP_POSITION, units='m',
                 desc='Center of pressure from nose', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.CG_POSITION, units='m',
                 desc='Center of gravity from nose', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.POWER_ON_CD, units=None,
                 desc='Drag coefficient at max Mach (power on)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)

av.add_meta_data(DynamicMissile6DOF.POWER_OFF_CD, units=None,
                 desc='Drag coefficient at max Mach (power off)', default_value=0.0,
                 meta_data=ExtendedMetaData6DOF)
