"""
Pre-mission Group for the RocketPy 6-DOF missile subsystem.

Assembles all discipline components into a single OpenMDAO Group
that runs during Aviary's pre-mission phase.  Currently this is
just the RocketPyFlightComp; separate atmosphere, motor, and aero
components can be added here later for finer-grained optimization.
"""

import openmdao.api as om

from aviary_mdo.missile_6dof.model.rocketpy_flight import (
    RocketPyFlightComp,
)


class Missile6DOFPreMission(om.Group):
    """Pre-mission group for 6-DOF missile analysis using RocketPy."""

    def setup(self):
        self.add_subsystem(
            'flight_6dof',
            RocketPyFlightComp(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
