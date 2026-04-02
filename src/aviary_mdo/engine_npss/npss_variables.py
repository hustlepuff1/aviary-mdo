import aviary.api as av

AviaryAircraft = av.Aircraft
AviaryDynamic = av.Dynamic


class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy for NPSS model."""

    class Engine(AviaryAircraft.Engine):
        DESIGN_MACH = 'aircraft:engine:design_mach'
        DESIGN_ALTITUDE = 'aircraft:engine:design_alt'
        DESIGN_MASS_FLOW = 'aircraft:engine:design_mass_flow'
        DESIGN_NET_THRUST = 'aircraft:engine:design_net_thrust'


class Dynamic(AviaryDynamic):
    """Dynamics data hierarchy for NPSS model."""

    class Engine:
        SHAFT_MECH_SPEED = 'dynamic:engine:shaft_mech_speed'
