"""
Variable hierarchy for the Missile DATCOM subsystem.

Follows Aviary's variable naming convention.
"""


class DATCOM:
    """Missile DATCOM aerodynamic variables."""

    class Flight:
        MACH = 'datcom:flight:mach'
        ALPHA = 'datcom:flight:alpha'
        REYNOLDS = 'datcom:flight:reynolds'
        ALTITUDE = 'datcom:flight:altitude'

    class Body:
        LNOSE = 'datcom:body:lnose'
        DNOSE = 'datcom:body:dnose'
        LCENTR = 'datcom:body:lcentr'
        DEXIT = 'datcom:body:dexit'
        LAFT = 'datcom:body:laft'
        DAFT = 'datcom:body:daft'

    class Reference:
        XCG = 'datcom:reference:xcg'
        SREF = 'datcom:reference:sref'
        LREF = 'datcom:reference:lref'

    class Aero:
        CN = 'datcom:aero:CN'
        CM = 'datcom:aero:CM'
        CA = 'datcom:aero:CA'
        CY = 'datcom:aero:CY'
        CLN = 'datcom:aero:CLN'
        CLL = 'datcom:aero:CLL'
        CL = 'datcom:aero:CL'
        CD = 'datcom:aero:CD'
