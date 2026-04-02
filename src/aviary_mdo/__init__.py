"""
aviary-mdo: Multidisciplinary Design Optimization modules for NASA Aviary.

External subsystem libraries that extend Aviary with:
- Tactical Missile Design (TMD) — Fleeman analytical sizing
- Aerothermodynamics — stagnation heating, TPS sizing
- RocketPy 6-DOF — flight simulation, Barrowman aero, grain regression
- Missile Guidance — PN/APN/OGL guidance, 3-DOF engagement simulation
- Missile DATCOM — semi-empirical aero prediction (USAF Missile DATCOM)
- NPSS Engine — community NPSS tabular engine model
- pyCycle Engine — thermodynamic cycle analysis (turbojet, turbofan, turboshaft)
- Community Aircraft — 11 aircraft models (A380, B757, C-17, C-5, etc.)

Usage:
    from aviary_mdo.missile import MissileBuilder
    from aviary_mdo.aerothermodynamics import AerothermoBuilder
    from aviary_mdo.missile_6dof import Missile6DOFBuilder
    from aviary_mdo.missile_guidance import GuidanceBuilder
    from aviary_mdo.datcom import DATCOMBuilder
    from aviary_mdo.engine_npss import NPSSTabularEngineBuilder
    from aviary_mdo.engine_pycycle import PyCycleEngineBuilder

Built on NASA Aviary (https://github.com/OpenMDAO/Aviary).
"""

__version__ = '0.1.0'
