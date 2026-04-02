"""
Extended metadata for the Tactical Missile Design subsystem.

Registers all missile-specific variables with Aviary's metadata system
so they can be used with add_aviary_input/add_aviary_output.
"""

import aviary.api as av
from aviary_mdo.missile.missile_variables import Missile, DynamicMissile

ExtendedMetaData = av.CoreMetaData

# ===== Body Geometry =====

av.add_meta_data(Missile.Body.LENGTH, units='inch',
                 desc='Total missile body length', default_value=143.9,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.DIAMETER_MAJOR, units='inch',
                 desc='Missile body major diameter', default_value=8.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.DIAMETER_MINOR, units='inch',
                 desc='Missile body minor diameter', default_value=8.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.NOSE_LENGTH, units='inch',
                 desc='Nose section length', default_value=19.2,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.NOSE_BLUNTNESS, units=None,
                 desc='Nose bluntness ratio', default_value=0.05,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.FINENESS_RATIO, units=None,
                 desc='Body fineness ratio l/d', default_value=18.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.REFERENCE_AREA, units='inch**2',
                 desc='Missile reference area', default_value=50.27,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Body.CG_STATION, units='inch',
                 desc='Missile center of gravity from nose', default_value=76.2,
                 meta_data=ExtendedMetaData)

# ===== Wing Geometry =====

av.add_meta_data(Missile.Wing.NUM_WINGS, units=None,
                 desc='Number of wing panels', default_value=2.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.PLANFORM_AREA, units='inch**2',
                 desc='Total wing planform area', default_value=400.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.ASPECT_RATIO, units=None,
                 desc='Wing aspect ratio', default_value=2.82,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.TAPER_RATIO, units=None,
                 desc='Wing taper ratio', default_value=0.175,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.LE_STATION, units='inch',
                 desc='Wing leading edge station from nose', default_value=60.8,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.THICKNESS_TO_CHORD, units=None,
                 desc='Wing max thickness-to-chord ratio', default_value=0.044,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.LE_THICKNESS_ANGLE, units='deg',
                 desc='Wing LE thickness half-angle', default_value=10.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Wing.INCIDENT_ANGLE, units='deg',
                 desc='Wing incidence angle', default_value=0.0,
                 meta_data=ExtendedMetaData)

# ===== Tail Geometry =====

av.add_meta_data(Missile.Tail.NUM_TAILS, units=None,
                 desc='Number of tail panels', default_value=2.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.ASPECT_RATIO, units=None,
                 desc='Tail aspect ratio', default_value=2.59,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.TAPER_RATIO, units=None,
                 desc='Tail taper ratio', default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.LE_STATION, units='inch',
                 desc='Tail leading edge station from nose', default_value=125.4,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.THICKNESS_TO_CHORD, units=None,
                 desc='Tail max thickness-to-chord ratio', default_value=0.027,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.LE_THICKNESS_ANGLE, units='deg',
                 desc='Tail LE thickness half-angle', default_value=6.2,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Tail.AREA, units='inch**2',
                 desc='Total tail area (computed or input)', default_value=87.0,
                 meta_data=ExtendedMetaData)

# ===== Weight =====

av.add_meta_data(Missile.Weight.LAUNCH, units='lbm',
                 desc='Missile launch weight', default_value=500.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Weight.BOOST_PROPELLANT, units='lbm',
                 desc='Boost propellant weight', default_value=84.8,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Weight.CRUISE_PROPELLANT, units='lbm',
                 desc='Cruise/sustain propellant weight', default_value=48.2,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Weight.EJECTABLES, units='lbm',
                 desc='Weight of booster ejectables', default_value=0.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Weight.EMPTY, units='lbm',
                 desc='Missile empty weight', default_value=367.0,
                 meta_data=ExtendedMetaData)

# ===== Propulsion =====

av.add_meta_data(Missile.Propulsion.BOOST_CHAMBER_PRESSURE, units='lbf/inch**2',
                 desc='Boost motor chamber pressure', default_value=1769.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.BOOST_EXPANSION_RATIO, units=None,
                 desc='Boost nozzle expansion ratio', default_value=6.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.BOOST_BURN_TIME, units='s',
                 desc='Boost motor burn time', default_value=3.26,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.BOOST_FUEL_TYPE, units=None,
                 desc='Boost fuel type (1-5 solid)', default_value=4.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.BOOST_THRUST, units='lbf',
                 desc='Boost motor thrust (computed)', default_value=7036.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.BOOST_ISP, units='s',
                 desc='Boost motor specific impulse (computed)', default_value=271.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.SUSTAIN_CHAMBER_PRESSURE, units='lbf/inch**2',
                 desc='Sustain motor chamber pressure', default_value=301.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.SUSTAIN_EXPANSION_RATIO, units=None,
                 desc='Sustain nozzle expansion ratio', default_value=6.2,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.SUSTAIN_BURN_TIME, units='s',
                 desc='Sustain motor burn time', default_value=10.86,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.SUSTAIN_FUEL_TYPE, units=None,
                 desc='Sustain fuel type', default_value=4.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.SUSTAIN_ENGINE_TYPE, units=None,
                 desc='Sustain engine type (1=rocket, 2=ramjet)', default_value=1.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.FUEL_AIR_RATIO, units=None,
                 desc='Ramjet fuel-to-air ratio', default_value=0.06,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Propulsion.MAX_COMBUSTOR_TEMP, units='degR',
                 desc='Max combustor temperature (ramjet)', default_value=4500.0,
                 meta_data=ExtendedMetaData)

# ===== Flight Conditions =====

av.add_meta_data(Missile.Flight.LAUNCH_MACH, units=None,
                 desc='Launch Mach number', default_value=0.8,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Flight.LAUNCH_ALTITUDE, units='ft',
                 desc='Launch altitude', default_value=20000.0,
                 meta_data=ExtendedMetaData)

av.add_meta_data(Missile.Flight.TERMINAL_MACH, units=None,
                 desc='Flight termination Mach number', default_value=1.5,
                 meta_data=ExtendedMetaData)
