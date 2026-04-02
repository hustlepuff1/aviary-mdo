"""OpenMDAO ExplicitComponent wrapper for Missile DATCOM.

Enables Missile DATCOM in multidisciplinary design optimization workflows.
Calls the Fortran executable per compute(), parses the output, and returns
aerodynamic coefficients as OpenMDAO outputs.
"""

import numpy as np
import openmdao.api as om

from .datcom_wrapper import MissileDATCOM


class DATCOMAero(om.ExplicitComponent):
    """OpenMDAO component wrapping Missile DATCOM.

    Options
    -------
    n_alpha : int
        Number of angle-of-attack points (default 8).
    body_config : dict
        Body geometry configuration passed to DATCOM.
    fin_configs : list of dict
        Fin geometry configurations (up to 4 fin sets).
    controls : list of str
        Control cards (default ['PART']).
    datcom_dir : str or None
        Path to Missile-Datcom installation (default: auto-detect).
    exe_path : str or None
        Explicit path to misdat executable (overrides datcom_dir).
    """

    def initialize(self):
        self.options.declare('n_alpha', default=8, types=int)
        self.options.declare('body_config', default=None, types=(dict, type(None)))
        self.options.declare('fin_configs', default=None, types=(list, type(None)))
        self.options.declare('controls', default=['PART'], types=list)
        self.options.declare('datcom_dir', default=None)
        self.options.declare('exe_path', default=None)

        self._datcom = None

    def setup(self):
        n = self.options['n_alpha']

        # Inputs
        self.add_input('mach', val=1.5, desc='Mach number')
        self.add_input('alpha', val=np.zeros(n), units='deg',
                       desc='Angles of attack')
        self.add_input('reynolds', val=1e6, desc='Reynolds number per foot')
        self.add_input('xcg', val=0.0, units='ft',
                       desc='Moment center x-location')

        # Optional geometry inputs (override body_config)
        self.add_input('lnose', val=0.0, units='ft',
                       desc='Nose length')
        self.add_input('dnose', val=0.0, units='ft',
                       desc='Nose diameter')
        self.add_input('lcentr', val=0.0, units='ft',
                       desc='Centerbody length')

        # Outputs — aerodynamic coefficients at each alpha
        self.add_output('CN', val=np.zeros(n), desc='Normal force coefficient')
        self.add_output('CM', val=np.zeros(n), desc='Pitching moment coefficient')
        self.add_output('CA', val=np.zeros(n), desc='Axial force coefficient')
        self.add_output('CY', val=np.zeros(n), desc='Side force coefficient')
        self.add_output('CLN', val=np.zeros(n), desc='Yawing moment coefficient')
        self.add_output('CLL', val=np.zeros(n), desc='Rolling moment coefficient')
        self.add_output('CL', val=np.zeros(n), desc='Lift coefficient')
        self.add_output('CD', val=np.zeros(n), desc='Drag coefficient')

        self._datcom = MissileDATCOM(
            datcom_dir=self.options['datcom_dir'],
            exe_path=self.options['exe_path']
        )

    def compute(self, inputs, outputs):
        mach = inputs['mach'].item()
        alpha = list(inputs['alpha'])
        reynolds = inputs['reynolds'].item()
        xcg = inputs['xcg'].item()

        # Build body config from inputs or options
        body = self.options['body_config']
        if body is None:
            lnose = float(inputs['lnose'])
            dnose = float(inputs['dnose'])
            lcentr = float(inputs['lcentr'])
            if lnose > 0 and dnose > 0:
                body = {'lnose': lnose, 'dnose': dnose, 'lcentr': lcentr}

        results = self._datcom.run(
            mach=[mach],
            alpha=alpha,
            reynolds=reynolds,
            xcg=xcg,
            body=body,
            fins=self.options['fin_configs'],
            controls=self.options['controls']
        )

        # Extract Case 1 results
        n = self.options['n_alpha']
        case = results.get(1, {})
        for coeff in ['CN', 'CM', 'CA', 'CY', 'CLN', 'CLL', 'CL', 'CD']:
            if coeff in case:
                data = np.array(case[coeff])
                outputs[coeff] = data[:n]
            else:
                outputs[coeff] = np.zeros(n)
