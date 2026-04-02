"""NPSS design engine group for Aviary integration.

Wraps the NPSS external code as an OpenMDAO ExternalCodeComp for
engine design-point analysis, then provides training data for
off-design mission interpolation.

Requires NPSS to be installed (runnpss on PATH).
"""

import subprocess

import numpy as np
import openmdao.api as om
from openmdao.utils.file_wrap import FileParser
from pathlib import Path

from aviary_mdo.engine_npss.npss_variables import Aircraft


# Path to the NPSS model data within this package
_NPSS_MODEL_DIR = Path(__file__).parent


class NPSSExternalCodeComp(om.ExternalCodeComp):
    """Component that wraps NPSS engine model."""

    def initialize(self):
        self.options.declare(
            'vec_size',
            default=72,
            types=int,
            desc='number of points in NPSS model deck',
        )
        self.options.declare(
            'npss_model_dir',
            default=None,
            desc='Path to NPSS model directory (defaults to package bundled model)',
        )

    def setup(self):
        vec_size = self.options['vec_size']
        self.add_input('Alt_DES', val=0.0, units='ft', desc='design altitude')
        self.add_input('MN_DES', val=0.0, units=None, desc='design Mach number')
        self.add_input('W_DES', val=240.0, units='lbm/s', desc='design mass flow')

        self.add_output(
            'Fn_SLS', val=1.0, units='lbf',
            desc='net thrust at sea-level-static conditions')
        self.add_output(
            'Wf_training_data', val=np.ones(vec_size), units='lbm/s',
            desc='fuel flow training data')
        self.add_output(
            'thrust_training_data', val=np.ones(vec_size), units='lbf',
            desc='thrust training data')
        self.add_output(
            'thrustmax_training_data', val=np.ones(vec_size), units='lbf',
            desc='maximum thrust training data')

        model_dir = self.options['npss_model_dir'] or _NPSS_MODEL_DIR

        self.input_file = str(Path(model_dir) / 'Design_files' / 'input.int')
        self.output_file = str(Path(model_dir) / 'Design_files' / 'output.int')

        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]

        run_location = str(Path(model_dir) / 'turbojet.run')
        engine_location = str(Path(model_dir))

        run_command = ['runnpss', run_location, '-D ENG_PATH=' + engine_location]
        self.options['command'] = run_command

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        Alt_DES = inputs['Alt_DES']
        MN_DES = inputs['MN_DES']
        W_DES = inputs['W_DES']
        vec_size = self.options['vec_size']

        with open(self.input_file, 'w') as input_file:
            input_file.write(
                'Alt_DES  = %.16f ;\nMN_DES  = %.16f ;\nW_DES  = %.16f ;'
                % (Alt_DES, MN_DES, W_DES))

        super().compute(inputs, outputs)

        parser = FileParser()
        parser.set_file(self.output_file)

        parser.mark_anchor('Fn_SLS')
        Fn_SLS = float(parser.transfer_var(0, 3))
        outputs['Fn_SLS'] = Fn_SLS

        parser.mark_anchor('Wf_training_data')
        parser.set_delimiters(', ')
        Wf_raw = parser.transfer_array(0, 4, 0, vec_size * 3)
        Wf_snip = [a for a in Wf_raw if a.replace('.', '').isnumeric()]
        outputs['Wf_training_data'] = np.array(Wf_snip, float)

        parser.mark_anchor('thrust_training_data')
        parser.set_delimiters(', ')
        thrust_raw = parser.transfer_array(0, 4, 0, vec_size * 3)
        thrust_snip = [a for a in thrust_raw if a.replace('.', '').isnumeric()]
        outputs['thrust_training_data'] = np.array(thrust_snip, float)

        parser.mark_anchor('thrustmax_training_data')
        parser.set_delimiters(', ')
        thrustmax_raw = parser.transfer_array(0, 4, 0, vec_size * 3)
        thrustmax_snip = [a for a in thrustmax_raw if a.replace('.', '').isnumeric()]
        outputs['thrustmax_training_data'] = np.array(thrustmax_snip, float)


class DesignEngineGroup(om.Group):
    """Group containing NPSSExternalCodeComp and negative fuel flow calc."""

    def initialize(self):
        self.options.declare(
            'vec_size', default=72, types=int,
            desc='number of points in NPSS model deck')

    def setup(self):
        vec_size = self.options['vec_size']
        self.add_subsystem(
            'DESIGN',
            NPSSExternalCodeComp(vec_size=vec_size),
            promotes_inputs=[('W_DES', Aircraft.Engine.DESIGN_MASS_FLOW)],
            promotes_outputs=[
                ('Fn_SLS', Aircraft.Engine.SCALED_SLS_THRUST),
                ('thrust_training_data', 'Fn_train'),
                ('thrustmax_training_data', 'Fn_max_train'),
                ('Wf_training_data', 'Wf_td'),
            ],
        )

        self.add_subsystem(
            'negative_fuel_rate',
            om.ExecComp(
                'y=-x',
                x={'val': np.ones(vec_size), 'units': 'lbm/s'},
                y={'val': np.ones(vec_size), 'units': 'lbm/s'},
                has_diag_partials=True,
            ),
            promotes_inputs=[('x', 'Wf_td')],
            promotes_outputs=[('y', 'Wf_inv_train')],
        )

    def configure(self):
        self.set_input_defaults(Aircraft.Engine.DESIGN_MASS_FLOW, 240.0, units='lbm/s')
        self.set_input_defaults('DESIGN.Alt_DES', 0.0, units='ft')
        self.set_input_defaults('DESIGN.MN_DES', 0.0)
        super().configure()
