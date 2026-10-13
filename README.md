# JEVPFN

Can label-free Jev features extracted from text improve tabular prediction?
Exploration covers the **20 core MulTaBench TEXT datasets**. Feature creation and predictive
experiments select **18**, excluding Zomato Restaurants for extreme text length/context
overflow and extraction cost, and Spotify Genres for its 114-class representation size. TabPFN-3.5 is the main predictor. The comparison
asks whether Jev features improve each fixed predictor on identical folds; ranking prediction
engines is outside the research objective. The focused candidate set adds TabPFN-3, CatBoost
and logistic/linear regression. Other catalogued engines are deferred; TabM, Google TabFM
and AutoGluon are excluded.

**Current phase: exploration and design. No paid calls or model experiments are enabled.**
The public repository is [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).

## Workflow

Exploration → question preparation and review → local feature pilot and full feature creation → experiment 0 on VSC →
predictive experiments. Iterative experiment proposals are discussed in chat; the disabled
YAML configurations record the current machine-readable design, with unresolved choices explicit.
The [Literature Review](docs/LITERATURE_REVIEW.md) retains verified research and interface findings.

Jev reads each text column separately and all available text columns jointly, with fixed
questions filled from task metadata. **Non-text features and true row targets never enter Jev.**
Skip empty inputs and store missing numeric features. Identical complete requests reuse a
cached response; future experiments join different combinations of those saved features.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](notebooks/exploration/01_data_exploration.ipynb) | All-dataset visual overview; text length, missingness, uniqueness, exact reuse, p99/max long-text boxplots and longest-cell examples. |
| [02_jev_input_design](notebooks/exploration/02_jev_input_design.ipynb) | All 18 applicable questions, exact per-column/joint previews for six providers, shared question bank and numeric output contracts. |
| [01_prepare_questions](notebooks/question_preparation/01_prepare_questions.ipynb) | Source-backed meanings for all 18 tasks and 80 text columns; raw class mappings, saved questions and one review JSON per dataset. |
| [03_feature_creation](notebooks/feature_creation/03_feature_creation.ipynb) | Selected-dataset feature counts, duplicate-aware costs, throughput and one-file-per-dataset storage. No API calls. |

All logic is in `src/`. Every notebook
ends with a full printed report containing all displayed tables, previews, plotted values and
captions. The runner saves executed outputs back into the source notebooks; generated reports remain local and ignored.

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
prints each cell start/completion and a heartbeat during long cells, and saves rich outputs back into each `.ipynb`. Reopen or reload an already-open VS Code notebook after execution. The previous notebook is backed up as `_previous.ipynb` beside its figures. Existing historical
backups remain ignored; rebuildable audit/budget intermediates use temporary storage and are removed after the run. Errors save partial outputs; edits made during execution are preserved and the executed copy is recovered separately.
`--only exploration/01_data_exploration` selects one notebook (a unique bare name also works);
`--summaries-only` rebuilds saved reports. Discovery includes every phase subfolder.

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --offline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.jev.review
```

The last command recalculates the duplicate-aware offline budget; it sends nothing.
The complete local preparation check is:

```powershell
.\.venv\Scripts\python.exe -m src.cluster.preflight --local --verify-data --tests
```

It verifies selected dataset hashes/schema/CV feasibility and runs synthetic contracts, saving
an immutable report and test log in `output_JEVPFN/experiment_0/`. Real provider, checkpoint,
CUDA, cluster allocation and full-fold resource checks remain explicitly pending; local success
does not establish full Experiment 0 acceptance.

`python -m src.jev.pilot` prepares a compatibility manifest from first/middle/last and
longest-text rows of all 18 selected datasets, for six disabled providers using the saved `documented`
questions. It records no raw text or targets and makes no API calls or model loads. The manifest
and preparation log use `output_JEVPFN/feature_creation/manifests/pilots/` and `logs/`.
On VSC the full trace manifest uses PROJECT; its small `.summary.json` index and log use DATA.
`config/decision_pilot.yaml` and `config/question_design.yaml` are review-only preparation;
neither enables extraction. Notebook 01 reports main numeric feature counts and multiclass
expansion; Notebook 02 displays the proposed shorter questions beside the accepted wording.
All dependencies are declared in `pyproject.toml`. Optional `jev`, `models`, `baselines`,
`evaluation`, `tabicl`, `tabdpt` and `tabstar` extras are for later implementation. Notebook setup
installs no API/model extras; development tests include the lightweight scoring dependencies.
GPU/model environments still require VSC validation; see
[the VSC guide](docs/VSC.md). Deferred LimiX-2, Causilo, EXAONE Tabular and additional text
comparators are interface references, not part of the focused main comparison. Do not install
their environments as part of notebook setup. AutoGluon is excluded.
Mitra-v2 is deferred because its verified integration requires AutoGluon; its checkpoint
namespace does not imply an independent supported runtime.

## Durable method constraints

The main method uses deterministic questions and no generative question writer. Only task
names, class vocabulary, column names and text values enter Jev. Row targets, class frequencies,
target summaries, correlations and model errors are excluded. Binary orientation uses canonical class order: fraud=1, toxicity=True and Kickstarter=Successful. Coded labels need an authoritative description where the target name alone is insufficient; this is metadata verification, not a new target definition.

Binary inputs use Noul probabilities; multiclass inputs use Choice class-probability vectors;
regression inputs use Score on the agreed nine levels −4 to +4. The accepted `directional-v2`
rubric distinguishes indirect hints, qualified support, clear direct evidence and explicit
supported comparisons in either direction; zero favors neither direction. It still needs
label-blind pilot checks. A probability-weighted direction is not a prediction in target
units. The agreed main numeric features are one positive-class probability for binary tasks, all K class probabilities for multiclass tasks, and the expected directional score for regression. Full responses/vectors and confidence are retained for optional later ablations without more calls. Jev is pinned to `jev-1.13.0` in config; a future version creates a separate cache identity.

For t nonempty text columns, a row has t separate inputs and one joint input before exact
reuse. Both means joining those two cached representations, not another API call. Identical
separate and joint requests on a dataset with one text column produce one numeric block;
the combined feature set does not duplicate that block. Identical
complete requests share their first stored response, irrespective of whether repeated API
execution would return bit-identical values. Changes to the model, question or text policy
produce different request identities. Empty inputs become missing numeric features.

The question-preparation notebook freezes exact questions/options for all 18 datasets into
`data/decision_inputs/questions/<bank_id>/<dataset_id>.json`, with a complete
`<bank_id>.json` bank beside that directory. One readable JSON per dataset is also exported to
`output_JEVPFN/question_preparation/questions/<bank_id>/`. The originals are inputs to later
extraction, so they survive output cleanup. `python -m src.jev.preparation` performs the same
offline preparation without running a notebook.

`config/dataset_semantics.yaml` stores sourced dataset/target meanings, typed raw-class mappings
and all 80 official text-field definitions. Three common templates in `src/jev/semantics.py`
create the `documented-task-v1` wording; no dataset-specific question writer or label statistics
are involved. Unknown definitions stay explicit. The original CSVs, target names and class values
are unchanged. `documented` is the current inspection variant; earlier `accepted` and
`proposed_direct` variants remain available, and no variant enables paid inference. The content hash detects changed wording,
class order, rubric or documentary context. Accepted and proposed questions carry separate
status/version fields; saving does not approve a proposal. No row text, targets or credentials
enter this shared file. All six preview adapters consume the same saved semantics. Proposed
wording explicitly says **the dataset row described by the supplied text**; a row ID is retained
only for tracing, not sent to the model. A new question bank preserves older versions.

Permanent extraction uses **one file per dataset**: `data/jev_cache/<dataset_id>.sqlite3`.
`responses` stores exact request/settings, raw response, token usage and UTC timestamp;
`origins` maps all source rows sharing a response. `feature_artifacts` contains Zstandard-compressed
Parquet tables as SQLite blobs: one numeric table and one row/request mapping per immutable
representation/version, with an embedded manifest and checksums. `dataset_identity` prevents
writing a different dataset into the file. Different Jev/template versions coexist without
replacing old results. No real dataset cache is created until extraction is approved.

`bind_provider` prepares separate provider/checkpoint cache identities while preserving the
shared text and question. The one-dataset-file layout can hold multiple providers, but current
response decoding and spending ledger are **Jev-specific**. Jev materialization now accepts
`question_bank=bank, question_variant="documented"` and records both identities. Before
multi-provider extraction, implement/test each response adapter and namespace spending by
provider/campaign. The notebook
shows exact constructed calls, not live-tested integrations. Hosted aliases without an
immutable revision remain unresolved; they must not be described as pinned weights.

The identity is **dataset ID + CSV/metadata fingerprint + `csv:N`**, where N is the zero-based
position in the original CSV before splitting/shuffling. Joins use these IDs; changed data,
missing IDs and duplicate IDs are rejected. Targets remain separate. Full probability vectors
remain available even when an experiment uses only the agreed main features.

`ResponseCache.for_dataset(dataset_id)` opens the dataset file. `materialize_cached_block`
accepts verified cached responses only; `save_block` publishes the numeric tables transactionally
in the same dataset file. `load_block(path, artifact_id)` selects a specific representation/version;
ambiguous selection fails. Legacy Parquet folders remain readable. Notebook mocks use temporary
files. SQLite may create a transient journal while writing; transfer only after writers close.
Feature-build logs go to `output_JEVPFN/feature_creation/logs/builds/`; compact completion indexes
go to `output_JEVPFN/feature_creation/manifests/features/`. On VSC those logs/indexes use DATA;
the dataset files use PROJECT. `src/jev/extraction.py` prepares bounded dispatch, permanent
in-flight reservations, measured usage and spending stops. Its SDK transport requires explicit
network opt-in and completed readiness checks; all live switches remain false. Account rates,
a verified billing-token upper bound, a reviewed context validator and approved per-dataset
spending allocations are mandatory. Dataset allocations together cannot exceed the campaign
limit. Ambiguous failures retain their reservation and require review, avoiding automatic
duplicate billing. Only known rate-limit rejections receive bounded retries. No live behavior
has been tested. Materialization batches response lookups; final feature tables still require
memory proportional to rows times retained features and need a scale test before full extraction.

The agreed split is **five-fold shuffled stratified CV for binary and multiclass classification**,
and **five-fold shuffled KFold for regression**, shared across all feature combinations/models.
Time and entity grouping are outside this benchmark's scope. Binary thresholds maximize F1
using three-fold out-of-fold predictions inside the outer training fold, followed by refitting
on the full outer training fold. Thresholds are frozen before outer-test prediction. All fitted
preprocessing stays inside the relevant inner/outer training partition.

Use all outer-training rows, with no arbitrary cap, HPO, optional model fine-tuning or extra
ensembling. Pin standard library settings, random seeds, versions and resource settings.
Record any model-internal context limit; an unsupported full-data run is an explicit failure
to review, never silent subsampling. Compare per-dataset Jev improvements within each model;
no model-ranking significance campaign is required. All sensible metrics are recorded now; the headline metric can be selected later, before reviewing comparative results. Text encoding for original-text comparators still needs agreement. Execution remains disabled in the configs.
No test performance may guide Jev feature design.

Evaluation requires explicit resolved estimator settings and feature artifact IDs (empty for
baselines). Run IDs include the feature values, dtypes and original row order, preventing
different feature versions from sharing output identities. Each fold releases its estimator
before constructing the next one; actual peak GPU memory still needs a runtime measurement.

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
model adapter is enabled by these utilities. Install `.[evaluation]`
for scoring-only environments; the development extra includes it for offline tests.
`src/evaluation/protocol.py` prepares shared splits, inner-OOF F1 threshold selection,
full outer-training refits and saving both outer-test and inner predictions. Execution requires
an explicit opt-in and a fresh estimator factory containing all fold-local preprocessing.
Tests exercise fake estimators only. Production model/checkpoint adapters still need validation.
Adapters must save both outer-test predictions and binary inner-OOF predictions,
including row IDs, targets, full class probabilities or continuous predictions, split identity
and feature-artifact identity. `save_evaluation(run_id=..., ...)` routes each prediction Parquet
and full metric record to project storage and a compact summary plus event log to DATA.

`feature_inventory(path)` lists embedded artifacts without reading their large feature tables.
`load_feature_selection(path, selector, representations)` selects exact provider/model/version,
question-bank and question-variant identities; missing or ambiguous selections fail.
`assemble_feature_set(..., selector=selector)` prevents accidental mixing across extractors or
question recipes. These are storage/selection contracts, not working alternate-model decoders.

`src/evaluation/features.py` implements the cheap text-statistics baseline: per-column
character/word counts and missingness, plus row totals. It also supplies the proposed negative
control that permutes whole Jev vectors independently within training/test partitions.
`assemble_features(..., view="main")` uses only the agreed main features; explicit views expose
retained distributions/confidence for ablations. Neither utility reads targets or calls an API.
The owner approved preparing six disabled extractor candidates. Their interface references live in
`config/extractors.yaml`. The proposed comparison is in `config/experiment_4/decision_models.yaml`; execution remains
disabled and the catalog does not promise working account access; alternative provider adapters and live compatibility pilots remain required.

## Files and outputs

```text
config/
  audit.yaml            local exploration settings
  datasets.yaml         pinned dataset catalog
  feature_creation.yaml local feature preparation
  question_design.yaml  versioned wording candidates
  dataset_semantics.yaml source-backed target, class and text-field definitions
  question_preparation.yaml offline preparation settings
  decision_pilot.yaml   disabled compatibility pilot
  extractors.yaml       six disabled provider/checkpoint candidates
  template_example.yaml inherited synthetic example settings
  experiment_0/         VSC debugging
  experiment_1/         main comparison, shared CV proposal and model catalog
  experiment_2/         original-text comparison proposal
  experiment_3/         Jev output-representation proposal
  experiment_4/         decision-extractor comparison proposal
notebooks/
  exploration/          all-20 audit and readable request inspection
  question_preparation/ source meanings, questions and per-dataset exports
  feature_creation/     local feature-build preparation
  experiment_N/         future notebooks when those experiments are implemented
src/                    reusable data, Jev, cluster, plotting and utility code
tests/                  deterministic checks
data/raw/               20 numbered folders, files directly inside each folder
data/decision_inputs/   immutable shared question catalogs; no row targets or text
data/jev_cache/         one permanent dataset SQLite file (responses + feature tables)
output_JEVPFN/
  All Results.md        complete reports from every notebook
  figures/
    Captions.md         aggregate captions beside the figures
    exploration/<notebook>/
    feature_creation/<notebook>/
    experiment_0/<notebook>/ ...
  question_preparation/ one readable question JSON per dataset and bank version
  feature_creation/     logs/ and compact manifests/, created by actual feature builds
  experiment_0/ ...     real experiment logs/, manifests/, results/, created when used
docs/                   Literature Review, data definitions, VSC setup and maintenance records
scripts/slurm/          prepared preflights; no jobs submitted
tfm-library/            pinned read-only literature submodule
```

Exploration and feature-design notebooks retain their visual outputs under the figures tree.
Actual local feature builds create the requested feature_creation/logs and manifests folders. Their
figures, per-notebook reports/configurations and run records are under `figures/<phase>/<notebook>/`;
`All Results.md` stays directly under `output_JEVPFN/`; aggregate captions are in
`output_JEVPFN/figures/Captions.md`. Rebuildable audit tables, request
previews and budget calculations use temporary storage shared during one runner invocation.
They are not permanent research artifacts; all displayed measurements are retained in the notebooks and full report. All notebooks always run
locally and use these local paths, independently of VSC storage environment variables. An
optional `JEVPFN_LOCAL_STORAGE_ROOT` can place these local artifacts on another local disk.
Only real experiment folders exist as config subdirectories. Config files for preparation
live directly under `config/`.

The small `.reports.lock` file coordinates parallel report writers; the OS releases it after
a crash. Windows sharing errors are retried. A persistently blocked report keeps its previous
content and a named `.pending` recovery copy.

For experiment jobs on VSC, compact manifests and bounded operational logs use
`$VSC_DATA/JEVPFN/output_JEVPFN/experiment_N/`. Individual predictions, full metric tables,
feature matrices and large results use `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/experiment_N/`.
Raw data, permanent paid caches/features and weights also use the project-storage `JEVPFN/`
directory. Shared question catalogs use that same project tier. Large writes never fall back to limited DATA. Locally both experiment storage tiers
collapse into the repository. Notebook figures and reports always remain local.

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
