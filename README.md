# JEVPFN

Can label-free Jev features extracted from text improve tabular prediction?
TabPFN-3.5 is the main model; the proposed benchmark also includes other foundation models
and conventional baselines on the **20 core MulTaBench TEXT datasets**.

**Current phase: exploration and design. No paid calls or model experiments are enabled.**
The public repository is [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).

## Workflow

Exploration → local feature pilot and full feature creation → experiment 0 on VSC →
predictive experiments. Iterative experiment proposals are discussed in chat; the disabled
YAML configurations record the current machine-readable design, with unresolved choices explicit.
The [Literature Review](docs/LITERATURE_REVIEW.md) retains verified research and interface findings.

Jev reads each text column separately and all available text columns jointly, with fixed
questions filled from task metadata. **Non-text features and true row targets never enter Jev.**
Skip empty inputs and store missing numeric features. Identical complete requests reuse a
cached response; future experiments join different combinations of those saved features.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](notebooks/exploration/01_data_exploration.ipynb) | All-dataset visual overview; text length, missingness, uniqueness and exact reuse in counts and percentages. |
| [02_jev_input_design](notebooks/exploration/02_jev_input_design.ipynb) | Request diagrams, task-specific output shapes, examples, temporary mock cache and workload charts. |
| [03_feature_creation](notebooks/feature_creation/03_feature_creation.ipynb) | Feature dimensions, intended API bodies, reusable tables and budget charts. No API calls. |

All logic is in `src/`. The inherited synthetic example remains separate. Every notebook
ends with a full printed report containing all displayed tables, previews, plotted values and
captions. Source notebooks have cleared outputs; generated reports remain local and ignored.

## Local environment and notebooks

Open the inner `JEVPFN/` directory containing this README and `pyproject.toml`.
For a fresh checkout, run in PowerShell:

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
.\.venv\Scripts\python.exe -m src.data.prepare
```

The existing local environment is already set up. Its prompt/kernel is **JEVPFN**; the
`.venv` folder is its location. Activation is optional. Launch Jupyter or run everything:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Select **Python 3.12 (JEVPFN)** in Jupyter/VS Code. The runner automatically uses available CPUs, capped by the number of notebooks and any
Slurm CPU allocation. It uses one numerical-library thread per notebook to avoid oversubscription,
saves logs and preserves reports from other notebooks during partial reruns.
`--only exploration/01_data_exploration` selects one notebook (a unique bare name also works);
`--summaries-only` rebuilds saved reports. Discovery includes every phase subfolder.

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --offline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.jev.review
```

The last command recalculates the duplicate-aware offline budget; it sends nothing.
All dependencies are declared in `pyproject.toml`. Optional `jev`, `models`, `baselines`,
`evaluation`, `tabicl`, `tabdpt` and `tabstar` extras are for later implementation. Notebook setup
installs no API/model extras; development tests include the lightweight scoring dependencies.
GPU/model environments still require VSC validation; see
[the VSC guide](docs/VSC.md). Newly catalogued LimiX-2, Causilo, EXAONE Tabular, TabFM and
additional text comparators are not installed by these extras. Their source/checkpoint pins
and separate environments must be validated before model execution. AutoGluon is excluded.
Mitra-v2 is deferred because its verified integration requires AutoGluon; its checkpoint
namespace does not imply an independent supported runtime.

## Durable method constraints

The main method uses deterministic questions and no generative question writer. Only task
names, class vocabulary, column names and text values enter Jev. Row targets, class frequencies,
target summaries, correlations and model errors are excluded. Binary orientation currently
uses canonical class order; domain meaning must be checked before freezing requests.

Binary inputs use Noul probabilities; multiclass inputs use Choice class-probability vectors;
regression inputs use Score on the agreed nine levels −4 to +4. The intermediate regression
descriptions still need review. A probability-weighted direction is not a prediction in target
units. Full responses/vectors are retained so selecting numeric feature views later is free.

For t nonempty text columns, a row has t separate inputs and one joint input before exact
reuse. Both means joining those two cached representations, not another API call. Identical
complete requests share their first stored response, irrespective of whether repeated API
execution would return bit-identical values. Changes to the model, question or text policy
produce different request identities. Empty inputs become missing numeric features.

Durable artifacts live under `data/jev_cache/features/<artifact_id>/`: `features.parquet`
contains stable `csv:N` row IDs and numeric columns, `requests.parquet` maps rows to request
keys/missing status, and `manifest.json` pins data, configuration, code and model provenance.
Targets stay separate. The future `responses.sqlite3` is a transactional response cache;
the notebooks use only temporary mock databases and do not create a paid-response database.

Five outer CV folds and validation-only binary F1 threshold selection are agreed. All fitted
preprocessing, tuning and calibration must remain inside training partitions. The exact splits,
inner validation, multiclass/regression metrics, output views and compute budgets remain disabled
proposals in `config/experiment_1/protocol.yaml`. No test performance may guide Jev feature design.

`src/evaluation/metrics.py` scores supplied predictions for all three task types without
fitting models. The protocol lists all recorded probability, decision, calibration, ranking
and regression metrics. Undefined metrics are null with a reason; invalid probability arrays
are rejected. Binary decisions are recorded at 0.5 and at a separately supplied validation
threshold. Multiclass Brier uses the summed-class convention (range 0–2); binary Brier uses
the positive-class convention (range 0–1). ECE is a 15-bin diagnostic, not a primary score.

`src/evaluation/records.py` records separate fit/prediction/preprocessing/inner-fit timings,
including failed attempts, and saves immutable records with the original predictions and
their checksum. CUDA timing requires synchronization. Hardware, context size and memory
measurements come from future model adapters; unavailable measurements remain null. No
benchmark runner or model adapter is enabled by these utilities. Install `.[evaluation]`
for scoring-only environments; the development extra includes it for offline tests.

## Files and outputs

```text
config/
  exploration/          dataset catalog and audit
  feature_creation/     local feature preparation
  experiment_0/         VSC debugging
  experiment_1/         main comparison, shared CV proposal and model catalog
  experiment_2/         original-text comparison proposal
  experiment_3/         Jev output-representation proposal
notebooks/
  exploration/          data audit, request inspection and inherited synthetic example
  feature_creation/     local feature-build preparation
  experiment_N/         future notebooks when those experiments are implemented
src/                    reusable data, Jev, cluster, plotting and utility code
tests/                  deterministic checks
data/raw/               20 numbered folders, files directly inside each folder
data/jev_cache/         future durable responses and feature tables, created when needed
output_JEVPFN/
  All Results.md        complete reports from every notebook
  Captions.md           all figure captions
  figures/
    exploration/<notebook>/
    feature_creation/<notebook>/
    experiment_0/<notebook>/ ...
  exploration/          reports/, logs/, manifests/, results/
  feature_creation/     same layout
  experiment_0/ ...     same layout, created when used
docs/                   Literature Review, data definitions, VSC setup and maintenance records
scripts/slurm/          prepared preflights; no jobs submitted
tfm-library/            pinned read-only literature submodule
```

Each phase also has `All Results.md` and `Captions.md`. Notebook figure folders contain PDFs,
`Captions.md` and a figure manifest. No empty future-output folders are created.
The small `.reports.lock` file coordinates parallel report writers; the OS releases it after
a crash. Windows sharing errors are retried. If another application persistently blocks a
report, its previous content and a named `.pending` recovery copy are preserved.

On VSC, small outputs use `$VSC_DATA/JEVPFN/output_JEVPFN/`, including the shared `figures/`
folder. Large tables use `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/`.
Raw data, paid caches and weights also live under that project-storage `JEVPFN/` directory.
A project-storage error stops the run rather than filling personal DATA.

## GitHub and provenance

VS Code tracks local Git changes; they appear on GitHub only after committing and pushing.
The repository is already connected to the public GitHub URL above. **The owner handles all
commits and pushes; agents must not stage, commit or push.** For the owner's future code updates:

```powershell
git status
git add <specific-code-or-config-files>
git commit -m "Describe the change"
git push origin main
```

On VSC, initially clone with `git clone --recurse-submodules https://github.com/andreasgoethals/JEVPFN.git`;
subsequent updates use `git pull --ff-only` and `git submodule update --init --recursive`.
Transfer data and completed feature artifacts separately; Git does not sync them.

The [data-source notes](docs/DATA_SOURCES.md) document official definitions and immutable hashes.
Raw data and paid caches are protected inputs, never cleanup targets. The old empty-folder
cleanup command is disabled after the dataset-loss incident. Credentials, `.env*`, data,
model weights and generated reports are ignored. No credentials are needed for the current notebooks.

Based on [Andreas' research template](https://github.com/andreasgoethals/0.-Template).
Useful source, style, test and Git-history conventions are retained; project-specific phases,
numbered raw folders and shared figures replace the original generic layout.
