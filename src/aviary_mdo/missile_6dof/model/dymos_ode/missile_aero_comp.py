"""Aerodynamic force computation for missile trajectory ODE.

Vectorized over num_nodes. Supports complex-step partials.
Uses Fleeman-style drag buildup: CD0 (zero-lift) + CDi (induced).
"""

import numpy as np
import openmdao.api as om


class MissileAeroComp(om.ExplicitComponent):
    """Compute drag and lift forces from flight conditions and geometry.

    This is a simplified aero model suitable for trajectory optimization.
    CD = CD0 + k * CN_alpha^2 * alpha^2  (induced drag from angle of attack)
    L = q * S * CN_alpha * alpha
    D = q * S * CD

    For supersonic missiles, CN_alpha ≈ 2/rad (slender body theory).
    CD0 is treated as an input (can come from TMD or DATCOM).
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('velocity', val=np.ones(nn), units='m/s')
        self.add_input('density', val=1.225 * np.ones(nn), units='kg/m**3')
        self.add_input('speed_of_sound', val=340.3 * np.ones(nn), units='m/s')
        self.add_input('alpha', val=np.zeros(nn), units='rad',
                        desc='Angle of attack')
        self.add_input('S_ref', val=0.0324, units='m**2',
                        desc='Reference area (body cross-section)')
        self.add_input('CD0', val=0.3, desc='Zero-lift drag coefficient')
        self.add_input('CN_alpha', val=2.0, units='1/rad',
                        desc='Normal force slope')

        # Outputs
        self.add_output('drag', val=np.zeros(nn), units='N')
        self.add_output('lift', val=np.zeros(nn), units='N')
        self.add_output('mach', val=np.zeros(nn))
        self.add_output('q_inf', val=np.zeros(nn), units='Pa',
                         desc='Dynamic pressure')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        # Diagonal partials (node-wise)
        self.declare_partials('drag', ['velocity', 'density', 'alpha'], rows=ar, cols=ar)
        self.declare_partials('drag', ['S_ref', 'CD0', 'CN_alpha'])
        self.declare_partials('lift', ['velocity', 'density', 'alpha'], rows=ar, cols=ar)
        self.declare_partials('lift', ['S_ref', 'CN_alpha'])
        self.declare_partials('mach', ['velocity', 'speed_of_sound'], rows=ar, cols=ar)
        self.declare_partials('q_inf', ['velocity', 'density'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        V = inputs['velocity']
        rho = inputs['density']
        a = inputs['speed_of_sound']
        alpha = inputs['alpha']
        S = inputs['S_ref']
        CD0 = inputs['CD0']
        CN_a = inputs['CN_alpha']

        q = 0.5 * rho * V ** 2
        # Induced drag factor: k = 1/(pi*AR*e), approximate for missile as 0.5
        k_ind = 0.5
        CN = CN_a * alpha
        CD = CD0 + k_ind * CN ** 2

        outputs['drag'] = q * S * CD
        outputs['lift'] = q * S * CN
        outputs['mach'] = V / a
        outputs['q_inf'] = q

    def compute_partials(self, inputs, J):
        V = inputs['velocity']
        rho = inputs['density']
        a = inputs['speed_of_sound']
        alpha = inputs['alpha']
        S = inputs['S_ref']
        CD0 = inputs['CD0']
        CN_a = inputs['CN_alpha']

        q = 0.5 * rho * V ** 2
        k_ind = 0.5
        CN = CN_a * alpha
        CD = CD0 + k_ind * CN ** 2

        dq_dV = rho * V
        dq_drho = 0.5 * V ** 2

        dCN_dalpha = CN_a
        dCN_dCNa = alpha
        dCD_dalpha = 2 * k_ind * CN * dCN_dalpha
        dCD_dCNa = 2 * k_ind * CN * dCN_dCNa

        # Drag partials
        J['drag', 'velocity'] = dq_dV * S * CD
        J['drag', 'density'] = dq_drho * S * CD
        J['drag', 'alpha'] = q * S * dCD_dalpha
        J['drag', 'S_ref'] = q * CD
        J['drag', 'CD0'] = q * S
        J['drag', 'CN_alpha'] = q * S * dCD_dCNa

        # Lift partials
        J['lift', 'velocity'] = dq_dV * S * CN
        J['lift', 'density'] = dq_drho * S * CN
        J['lift', 'alpha'] = q * S * dCN_dalpha
        J['lift', 'S_ref'] = q * CN
        J['lift', 'CN_alpha'] = q * S * dCN_dCNa

        # Mach partials
        J['mach', 'velocity'] = 1.0 / a
        J['mach', 'speed_of_sound'] = -V / a ** 2

        # q_inf partials
        J['q_inf', 'velocity'] = dq_dV
        J['q_inf', 'density'] = dq_drho
