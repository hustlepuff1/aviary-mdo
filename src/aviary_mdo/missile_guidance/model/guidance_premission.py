"""
Pre-mission group for missile guidance analysis.

Assembles the guided engagement simulation into an OpenMDAO Group
that can be integrated with Aviary's SubsystemBuilder pattern.
"""

import openmdao.api as om

from aviary_mdo.missile_guidance.model.engagement_geometry import EngagementGeometry
from aviary_mdo.missile_guidance.model.proportional_navigation import ProNavGuidance
from aviary_mdo.missile_guidance.model.actuator_dynamics import ActuatorDynamics
from aviary_mdo.missile_guidance.model.seeker_model import SeekerModel
from aviary_mdo.missile_guidance.model.guided_engagement_sim import GuidedEngagementSim


class GuidancePremission(om.Group):
    """Pre-mission group for missile guidance analysis.

    Contains both analytical sizing tools and the full engagement simulation.
    The analytical components (engagement geometry, PN, actuator, seeker) run
    fast for MDO. The engagement sim validates with a 3-DOF trajectory.
    """

    def setup(self):
        # Full engagement simulation (primary output source)
        self.add_subsystem('engagement_sim', GuidedEngagementSim(),
                           promotes_inputs=['*'], promotes_outputs=['*'])
