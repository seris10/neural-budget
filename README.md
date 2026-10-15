# Neural Budget

Work in progress; development and validation are ongoing.

Python reproduction and parameter analysis of the cerebral-cortex ATP budget in Howarth, Gleeson and Attwell (2012). The calculation covers glutamate recycling, postsynaptic receptors, presynaptic calcium handling and vesicle cycling, action potentials, and resting potentials. Housekeeping is an additional specified allowance.

## Install and run

Python 3.13 is the tested baseline; package metadata requires Python 3.11 or later. Run from the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt .
.venv/bin/neural-budget reproduce --out reports/example
.venv/bin/neural-budget scenario src/neural_budget/scenarios/hodgkin_overlap_factor.toml
.venv/bin/neural-budget validate
.venv/bin/python -m pytest -W error
```

Open `reports/example/report.html` to inspect the generated calculation. Packaged numerical reference values and formulas are sufficient to run offline after dependency installation. Results use ATP molecules per neuron per second and micromoles ATP per gram per minute.

## Methods and outputs

The reference retains the expressions, rounded constants, and accounting conventions of the workbook's cerebral-cortex sheet. Twenty-five cached workbook outputs are compared at relative tolerance 10⁻¹². Twelve printed paper values use their displayed numerical precision; the approximate sleep-versus-running reduction uses a five-percentage-point tolerance.

Seven TOML scenarios change parameters or accounting conventions. Sensitivity analysis varies 14 parameters individually and evaluates a two-level factorial design for firing rate, sodium overlap, and release probability. The bounds have no assigned coverage probability.

Outputs include `reference_reproduction.csv`, `component_budget.csv`, `parameter_sources.csv`, `scenario_results.csv`, `spreadsheet_findings.csv`, `sensitivity.csv`, `tornado.csv`, `factorial_effects.csv`, `limiting_cases.csv`, `excluded.csv`, and `report.html`.

`neural_budget.synapse.resource_state` implements the optional mean-field resource equation `dR/dt = (1-R)/tau - U*f*R`. `depleted_budget` substitutes effective release `U*R` while retaining the reference action-potential, resting-potential, and per-vesicle costs. Recovery time is an illustrative control; experimental ATP trajectories are not fitted.

## Verification and limits

Tests cover dimensional and parameter contracts, reference calculations, accounting limits, command-line output, and independent ODE checks of synaptic depletion. The workbook re-extraction test skips unless the pinned XLS is present in `data/raw`. To run that check, obtain the source identified by [data_manifest.json](data_manifest.json), verify its recorded SHA-256, and rerun the tests. `tools/extract_reference.py` and `tools/xls_formulas.py` implement extraction.

Agreement with workbook and paper values establishes numerical reproduction. It does not establish empirical ATP-consumption accuracy. The model combines estimates from different species and preparations; a joint physiological uncertainty distribution is unspecified. It treats cortical neurons as glutamatergic, uses mean dimensions, and excludes separate inhibitory populations, cortical layers, white matter, and cerebellar calculations. The reproduced budget has no temperature dependence.

## Sources and licensing

Code uses [BSD-3-Clause](LICENSE). Cite Howarth C, Gleeson P, Attwell D (2012), *Updated energy budgets for neural computation in the neocortex and cerebellum*, [doi:10.1038/jcbfm.2012.35](https://doi.org/10.1038/jcbfm.2012.35).

The source workbook permits investigation of parameter changes and adaptation to other cell types or regions with acknowledgment of the paper. Extracted values/formulas retain their attribution and source locations. The original XLS is not redistributed. Acquisition details, the archived source URL, checksums, and transformations are recorded in [data_manifest.json](data_manifest.json). Installed dependencies retain their own licenses.

The related [Model Atlas](https://github.com/seris10/model-atlas) visualizes the ATP accounting at [atlas.relay70.dev](https://atlas.relay70.dev/#neural).
