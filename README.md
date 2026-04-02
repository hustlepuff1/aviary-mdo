# aviary-mdo

Multidisciplinary Design Optimization modules for [NASA Aviary](https://github.com/OpenMDAO/Aviary). Extends Aviary with missile design, aerothermodynamics, flight simulation, guidance, and propulsion cycle analysis.

Built on [OpenMDAO](https://openmdao.org/) and [Dymos](https://github.com/OpenMDAO/dymos) for gradient-based trajectory optimization.

> **Note:** This entire codebase -- all 8 modules, 229 tests, Dymos ODE, Fortran integration, and this README -- was developed using [Claude Code](https://claude.ai/code) (Anthropic's AI coding agent). The human provided the domain expertise, reference materials, and architectural direction; Claude Code wrote the implementation, tests, and documentation.

## Modules

| Module | Builder | Description |
|--------|---------|-------------|
| `aviary_mdo.missile` | `MissileBuilder` | Tactical Missile Design (Fleeman TMD) -- 7-discipline analytical sizing |
| `aviary_mdo.aerothermodynamics` | `AerothermoBuilder` | Stagnation heating (Fay-Riddell, Sutton-Graves), TPS mass estimation |
| `aviary_mdo.missile_6dof` | `Missile6DOFBuilder` | RocketPy 6-DOF trajectory + Dymos-compatible 3-DOF ODE with analytic partials |
| `aviary_mdo.missile_guidance` | `GuidanceBuilder` | PN/APN/OGL guidance laws, 3-DOF engagement simulation |
| `aviary_mdo.datcom` | `DATCOMBuilder` | USAF Missile DATCOM semi-empirical aerodynamic prediction |
| `aviary_mdo.engine_npss` | `NPSSTabularEngineBuilder` | NPSS tabular engine model (from Aviary Community) |
| `aviary_mdo.engine_pycycle` | `PyCycleEngineBuilder` | pyCycle thermodynamic cycle analysis (turbojet, turbofan, turboshaft) |

Additionally, 11 community aircraft models are included under `aviary_mdo.examples.aircraft`.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/aviary-mdo.git
cd aviary-mdo
pip install -e ".[all]"
```

### Optional dependencies

The `[all]` extra installs Aviary, RocketPy, JSBSim, and pyCycle. You can also install subsets:

```bash
pip install -e ".[aviary]"      # Aviary only
pip install -e ".[rocketpy]"    # RocketPy only
pip install -e ".[pycycle]"     # pyCycle only
pip install -e ".[test]"        # All deps + pytest
```

### Missile DATCOM

The DATCOM module requires a compiled Fortran executable (not bundled). See the [Missile DATCOM setup](#missile-datcom-setup) section below.

## Quick start

```python
import openmdao.api as om
from aviary_mdo.missile import MissileBuilder

# TMD sizing
missile = MissileBuilder(name='missile', sustain_engine='rocket')
prob = om.Problem()
prob.model.add_subsystem('missile', missile.build_pre_mission(aviary_inputs=None))
prob.setup()
prob.run_model()
```

```python
# Dymos trajectory optimization with analytic gradients
import dymos as dm
from aviary_mdo.missile_6dof.model.dymos_ode import MissileODE

phase = dm.Phase(
    ode_class=MissileODE,
    ode_init_kwargs={'S_ref': 0.0324, 'CD0': 0.3},
    transcription=dm.GaussLobatto(num_segments=20, order=3),
)
```

```python
# pyCycle engine analysis
from aviary_mdo.engine_pycycle import PyCycleEngineBuilder

engine = PyCycleEngineBuilder(cycle_type='turbojet')
```

```python
# DATCOM semi-empirical aerodynamics
from aviary_mdo.datcom import DATCOMBuilder

datcom = DATCOMBuilder(
    body_config={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25},
    n_alpha=8,
)
```

## Testing

```bash
# All tests (229 tests)
python -m pytest src/ -v

# Individual modules
python -m pytest src/aviary_mdo/missile/test/ -v            # TMD: 99 tests
python -m pytest src/aviary_mdo/aerothermodynamics/test/ -v  # Aerothermo: 57 tests
python -m pytest src/aviary_mdo/missile_6dof/test/ -v        # 6-DOF: 38 tests
python -m pytest src/aviary_mdo/missile_guidance/test/ -v    # Guidance: 14 tests
python -m pytest src/aviary_mdo/datcom/test/ -v              # DATCOM: 8 tests
python -m pytest src/aviary_mdo/engine_npss/test/ -v         # NPSS: 6 tests
python -m pytest src/aviary_mdo/engine_pycycle/test/ -v      # pyCycle: 7 tests
```

## Architecture

All modules follow Aviary's `SubsystemBuilder` pattern and can be used as external subsystems in `AviaryProblem`:

```python
import aviary.api as av

prob = av.AviaryProblem()
prob.load_inputs('aircraft.csv', phase_info)
prob.load_external_subsystems([MissileBuilder(), AerothermoBuilder()])
prob.check_and_preprocess_inputs()
prob.build_model()
prob.setup()
prob.run_aviary_problem()
```

### Trajectory optimization

The `missile_6dof` module provides two trajectory models:

- **`MissileODE`** (Dymos-compatible) -- 3-DOF point-mass with analytic partials. Use inside Dymos phases for gradient-based trajectory optimization.
- **`RocketPyFlightComp`** (high-fidelity) -- Full 6-DOF via RocketPy with finite-difference partials. Use for post-design verification.

This dual-model architecture follows the standard practice of optimizing with a differentiable model and validating with a higher-fidelity simulation.

## Missile DATCOM setup

The DATCOM module wraps the USAF Missile DATCOM Fortran code. You need to:

1. Obtain the Missile DATCOM source code
2. Compile it:
   ```bash
   cd "Source Code-Missile DATCOM"
   gfortran -std=legacy -w *.f -o ../misdat
   # or with Intel Fortran:
   ifx -fixed -w *.f -o ../misdat
   ```
3. Set the path (defaults to `~/MDO/Missile-Datcom/`):
   ```bash
   export MISSILE_DATCOM_DIR=/path/to/Missile-Datcom
   ```

## References

- Fleeman, E., *Tactical Missile Design*, AIAA Education Series, 2001.
- Fay, J.A. and Riddell, F.R., "Theory of Stagnation Point Heat Transfer in Dissociated Air," *Journal of the Aerospace Sciences*, 1958.
- Sutton, K. and Graves, R.A., "A General Stagnation-Point Convective-Heating Equation for Arbitrary Gas Mixtures," NASA TR R-376, 1971.
- Zarchan, P., *Tactical and Strategic Missile Guidance*, AIAA, 7th ed.
- Hendricks, E.S. and Gray, J.S., "pyCycle: A Tool for Efficient Optimization of Gas Turbine Engine Cycles," *Aerospace*, vol. 6, no. 87, 2019.
- Vukelich, S.R. et al., "Missile DATCOM," AFRL-VA-WP-TR-1998-3009.
- Gratz, J. et al., "Aviary: An Open-Source Multidisciplinary Design, Analysis, and Optimization Tool for Modeling Aircraft," AIAA 2024.

## Validation

Validation against published reference data is in progress. Current status:

| Module | Validation status |
|--------|------------------|
| TMD (Track 1) | Validated against Fleeman TMD spreadsheet (Sparrow MRAAM) |
| Aerothermo (Track 2) | Pending -- Fay-Riddell (1958), Sutton-Graves (1971) |
| 6-DOF (Track 3) | Pending -- cross-validation against TMD |
| Guidance (Track 4) | Pending -- Zarchan textbook standard cases |
| DATCOM (Track 5) | Pending -- DATCOM Users Guide validation cases |
| pyCycle (Track 7) | Turbojet design point converges to target thrust/TSFC |

Quantitative validation results will be added here as they are completed.

## Built with Claude Code

This project was developed entirely using [Claude Code](https://claude.ai/code), Anthropic's agentic coding tool. ~17,000 lines of Python across 128 files, compiled Fortran via Intel ifx, integrated 5 external tools (Aviary, RocketPy, JSBSim, pyCycle, Missile DATCOM), and hand-coded analytic Jacobians for Dymos compatibility -- all through conversational AI-assisted development.

The human researcher provided:
- Domain expertise in aerospace engineering and MDO
- Reference materials (Fleeman, Fay-Riddell, Zarchan, DATCOM manuals)
- Architectural direction and validation requirements
- Review and acceptance of all implementation decisions

Claude Code provided:
- All source code, tests, and documentation
- OpenMDAO/Aviary integration patterns
- Fortran compilation and Python wrapper development
- Analytic partial derivative derivations and implementation

## License

MIT
