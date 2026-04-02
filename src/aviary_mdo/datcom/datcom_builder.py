"""
DATCOMBuilder — Aviary SubsystemBuilder for Missile DATCOM aerodynamics.

Integrates the USAF Missile DATCOM semi-empirical aerodynamic prediction
code into Aviary's problem structure following the SubsystemBuilder pattern.

Usage:
    from aviary_mdo.datcom import DATCOMBuilder

    datcom = DATCOMBuilder(
        name='datcom',
        body_config={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25},
        n_alpha=8
    )

    # As Aviary external subsystem:
    prob.load_inputs(..., external_subsystems=[datcom])

    # Or standalone:
    prob = om.Problem()
    prob.model.add_subsystem('datcom', datcom.build_pre_mission(aviary_inputs))
    prob.setup()
    prob.run_model()

Reference: Vukelich, S.R. et al., "Missile DATCOM," AFRL-VA-WP-TR-1998-3009.
"""

import openmdao.api as om
from aviary.subsystems.subsystem_builder import SubsystemBuilder

from aviary_mdo.datcom.model.datcom_aero import DATCOMAero


class DATCOMBuilder(SubsystemBuilder):
    """
    Aviary SubsystemBuilder for Missile DATCOM aerodynamics.

    Wraps the Fortran-based DATCOM code as an OpenMDAO component for
    semi-empirical aerodynamic coefficient prediction across subsonic
    through hypersonic speed regimes.

    Parameters
    ----------
    name : str
        Name of the subsystem (default: 'datcom').
    body_config : dict, optional
        Body geometry for DATCOM (lnose, dnose, lcentr, dexit, etc.).
    fin_configs : list of dict, optional
        Up to 4 fin set definitions.
    n_alpha : int
        Number of angle-of-attack points (default 8).
    datcom_dir : str, optional
        Path to Missile-Datcom installation.
    exe_path : str, optional
        Explicit path to misdat executable.
    """

    default_name = 'datcom'

    def __init__(self, name=None, body_config=None, fin_configs=None,
                 n_alpha=8, datcom_dir=None, exe_path=None):
        self.body_config = body_config
        self.fin_configs = fin_configs
        self.n_alpha = n_alpha
        self.datcom_dir = datcom_dir
        self.exe_path = exe_path
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs=None, **kwargs):
        """
        Build the pre-mission DATCOM aerodynamics group.

        Returns an OpenMDAO Group containing the DATCOMAero component.

        Parameters
        ----------
        aviary_inputs : AviaryValues or dict, optional
            Input values (currently unused; geometry comes from options).

        Returns
        -------
        group : om.Group
            OpenMDAO Group wrapping DATCOMAero.
        """
        group = om.Group()
        group.add_subsystem('aero', DATCOMAero(
            n_alpha=self.n_alpha,
            body_config=self.body_config,
            fin_configs=self.fin_configs,
            datcom_dir=self.datcom_dir,
            exe_path=self.exe_path,
        ), promotes=['*'])
        return group

    def get_mass_names(self):
        return []

    def get_outputs(self):
        return []
