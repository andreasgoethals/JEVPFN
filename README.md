# JEVPFN

Can label-free Jev features extracted from text improve tabular prediction?
Exploration covers the **20 core MulTaBench TEXT datasets**. Feature creation and predictive
experiments select **18**, excluding Zomato Restaurants for extreme text length/context
overflow and extraction cost, and Spotify Genres for its 114-class representation size.
The comparison measures decision-model text features within each fixed predictor on identical
folds. The selected predictors are **CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo
Tabular and logistic/linear regression**. The original-text comparators are **TabSTAR and
ConTextTab**. AutoGluon, TabM and Google TabFM are excluded; other catalog entries are references.

**Current phase: Experiment 0A preparation and offline testing. No paid calls or GPU/model experiments are enabled.**
The public repository is [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).

## Workflow

Exploration → frozen question/text preparation → Experiment 0A (exact tokens and small VSC compatibility pilots) → full feature creation → Experiment 0B (complete-feature checks and predictor tests) →
predictive experiments. Both test stages share `3. experiment_0/`; feature creation retains its existing numbered folder. Iterative experiment proposals are discussed in chat; the disabled
YAML configurations record the current machine-readable design, with unresolved choices explicit.
The [Literature Review](docs/LITERATURE_REVIEW.md) retains verified research and interface findings.

Jev reads each text column separately and all available text columns jointly, with fixed
questions filled from task metadata. **Non-text features and true row targets never enter Jev.**
Skip empty inputs and store missing numeric features. Identical complete requests reuse a
cached response; future experiments join different combinations of those saved features.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](<notebooks/0. exploration/01_data_exploration.ipynb>) | All-dataset visual overview; text length, missingness, uniqueness, exact reuse, p99/max long-text boxplots and longest-cell examples. |
| [02_jev_input_design](<notebooks/0. exploration/02_jev_input_design.ipynb>) | All 18 applicable questions, exact per-column/joint previews for six providers, shared question bank and numeric output contracts. |
| [01_prepare_questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) | Source-backed meanings for all 18 tasks and 80 text columns; raw class mappings, saved questions and one SQLite input file per dataset, exact text reuse and first-row previews. |
| [03_feature_creation](<notebooks/2. feature_creation/03_feature_creation.ipynb>) | Selected-dataset feature counts, saved exact tokenizer audits, duplicate-aware costs, throughput and one-file-per-dataset storage. No API calls. |
| [04_decision_feature_comparison](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>) | Saved-feature coverage, agreement counts, probability/score differences, distributions, duplicate sensitivity and distance from the other models. No inference or target labels. |

All logic is in `src/`. Every notebook
ends with a full printed report containing all displayed tables, previews, plotted values and
captions. The runner saves executed outputs back into the source notebooks; generated reports remain local and ignored.

The comparison notebook reads the permanent dataset SQLite files. It verifies model/artifact
versions, row alignment, class order and identical semantic input hashes before comparing
models. If several versions exist, select artifact IDs in
`config/2. feature_creation/feature_comparison.yaml`; it never guesses the newest one.
Pairwise comparisons become available as features arrive; consensus rankings require the
complete configured roster. Reports separate all-row and exact-input weighting and average
datasets equally. Distance from the other models is a disagreement diagnostic, not accuracy.
With no real features, the notebook shows pending coverage only. Its local CSV tables and
source manifest sit beside its PDFs; the permanent feature cache is read-only.

Notebook 03's historical character screen is an approximation. Experiment 0A now counts
the complete published input formats with the pinned Liquid, Strands and Perplexity tokenizers:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[token-audit]"
.\.venv\Scripts\python.exe -m src.cluster.token_audit --stage-tokenizers
```

This downloads about 58 MB of tokenizer/source files, **no weights**, then counts every unique
nonempty input in all 18 selected datasets. Subsequent `python -m src.cluster.token_audit`
runs are offline and resume committed batches. Per-input counts use PROJECT on VSC; compact
summaries/logs use DATA under `output_JEVPFN/3. experiment_0/`, with `before_extraction`
subfolders. No raw rows are removed. Jev, Clef and Microsoft remain explicitly unverified:
their complete hosted tokenization is not available for this offline check. A base model's
tokenizer is not a substitute. Every self-hosted inference also checks the runtime's reported
token count against the audited format. No silent truncation is allowed.
The 14-10-2026 audit counted all 1,698,127 unique prepared inputs for each of the three open
models and found no overflows. Notebook 03 reads those saved results only when the input
file identities still match; the earlier character warnings do not justify deleting rows.

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
`--only 01_data_exploration` selects one notebook (a unique bare name also works);
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
an immutable report and test log in `output_JEVPFN/3. experiment_0/`. Real provider, checkpoint,
CUDA, cluster allocation and full-fold resource checks remain explicitly pending; local success
does not establish full Experiment 0 acceptance.

`python -m src.jev.pilot` prepares a compatibility manifest from first/middle/last and
longest-text rows of all 18 selected datasets, for six disabled providers using the saved `documented`
questions. It records no raw text or targets and makes no API calls or model loads. The manifest
and preparation log use `output_JEVPFN/2. feature_creation/manifests/pilots/` and `logs/`.
On VSC the full trace manifest uses PROJECT; its small `.summary.json` index and log use DATA.
`config/2. feature_creation/decision_pilot.yaml` and `config/1. question_preparation/question_design.yaml` are review-only preparation;
neither enables extraction. Notebook 01 reports main numeric feature counts and multiclass
expansion; Notebook 02 displays the proposed shorter questions beside the accepted wording.
Dependencies are declared in `pyproject.toml`; notebook setup installs no API/model extras.
Optional predictor extras are `models` (TabPFN), `baselines`, `tabicl`, `limix`, `kumo`,
`tabstar` and `contexttab`; `jev`, `liquid` and `strands` support later extractor environments.
Perplexity supplies checkpoint-specific requirements. Custom runtimes use isolated environments,
staged checkpoints and hashes; do not install all extras into the notebook environment.
[The VSC guide](docs/VSC.md) records the interfaces and setup sequence. No GPU runtime has
been exercised here. Causilo, Mitra-v2, EXAONE and TabDPT remain outside the selected roster.

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

Question preparation writes **exactly one SQLite input file per selected dataset**, directly in
`data/decision_inputs/<dataset_id>.sqlite3`. The workflow no longer exports question copies under output.
Run it locally from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.jev.preparation --preview BIN_TEXT_FAKE_JOB_POSTING --row 0
```

The first command verifies the pinned files, freezes all 18 task questions and groups every
row's per-column/joint inputs. Existing completed versions are reused. `--preview` prints the
exact saved first-row inputs without inference; if several input versions exist, supply the
`--input-set-id` printed in the preparation manifest. The preparation notebook performs the
same work and displays the measured grouping counts and every first-row request.

`question_banks` stores each dataset's immutable question versions. `input_sets` identifies
the question version, original-data fingerprint, missing-text rule and column slots.
`unique_inputs` stores each distinct named text group once, with its complete payload hash.
`row_map` links every original `csv:N` row/slot to that input; NULL means missing text, hence
no call and a missing numeric feature. Exact repeats in the same column reuse one input;
identical joint inputs do too. A joint input containing only one available field can reuse
its per-column input. Different field names, nonempty whitespace, question meanings or task
context remain distinct. No truncation, target values, non-text values or target statistics
enter preparation. Changed data/wording creates a version inside the same file.

Questions are deterministic templates filled from `config/1. question_preparation/dataset_semantics.yaml`:
source-backed target descriptions, typed raw-class mappings and all 80 official text-field
definitions. Unverified facts remain explicit. `documented` is the current review variant;
older `accepted` and `proposed_direct` recipes remain traceable. Questions do not depend on a
row's true target. All rows of a dataset share a question; the supplied text identifies the row.
`src.jev.inputs.iter_prepared_inputs` streams each unique payload once; `input_origins` expands
its response back to every source row and representation. A payload hash is provider-neutral;
a paid response key additionally includes provider/model revision and inference settings.

Preparation logs and compact manifests go to `output_JEVPFN/1. question_preparation/`.
Hosted inference is planned locally; self-hosted models need a validated VSC GPU environment
and a small compatibility pilot first. Open weights remove per-token API charges, not GPU
resource costs or licence requirements. All six candidates remain disabled.

Permanent extraction uses **one file per dataset**: `data/jev_cache/<dataset_id>.sqlite3`.
`responses` stores exact request/settings, raw response, token usage and UTC timestamp;
`origins` maps all source rows sharing a response. `feature_artifacts` contains Zstandard-compressed
Parquet tables as SQLite blobs: one numeric table and one row/request mapping per immutable
representation/version, with an embedded manifest and checksums. `dataset_identity` prevents
writing a different dataset into the file. Different Jev/template versions coexist without
replacing old results. No real dataset cache is created until extraction is approved.

`bind_provider` prepares separate provider/checkpoint cache identities while preserving the
shared text and question. Response decoding and feature materialization now support Jev and
the three self-hosted candidates; the latter also bind checkpoint, renderer and installed-runtime
identities. `src.jev.self_hosted` implements the guarded serial pilot/full-extraction paths,
cache recovery and permanent dataset features. Synthetic tests do not establish actual GPU
compatibility. Clef/Microsoft response adapters and separate hosted-provider spending accounting
remain pending. Materialization accepts `question_bank=bank, question_variant="documented"`
and records both identities. The notebook
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
Feature-build logs go to `output_JEVPFN/2. feature_creation/logs/builds/`; compact completion indexes
go to `output_JEVPFN/2. feature_creation/manifests/features/`. On VSC those logs/indexes use DATA;
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

Use all outer-training rows, with no arbitrary cap, HPO or optional extra adaptation.
Pin standard library settings, random seeds, versions and resource settings.
Record any model-internal context limit; an unsupported full-data run is an explicit failure
to review, never silent subsampling. Compare per-dataset Jev improvements within each model;
no model-ranking significance campaign is required. All sensible metrics are recorded now;
the headline metric can be selected before reviewing comparative results. TabSTAR's standard
supervised adaptation/internal validation and ConTextTab's pretrained text encoder are part
of those methods. Their checkpoint and encoder pins still need a runtime pilot. Execution
remains disabled in the configs.
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
Tests exercise fake estimators only. Source-checked adapters are implemented in
`src/evaluation/models.py`; real checkpoint/environment validation remains pending.
Adapters must save both outer-test predictions and binary inner-OOF predictions,
including row IDs, targets, full class probabilities or continuous predictions, split identity
and feature-artifact identity. `save_evaluation(run_id=..., ...)` routes each prediction Parquet
and full metric record to project storage and a compact summary plus event log to DATA.

`feature_inventory(path)` lists embedded artifacts without reading their large feature tables.
`load_feature_selection(path, selector, representations)` selects exact provider/model/version,
question-bank and question-variant identities; missing or ambiguous selections fail.
`assemble_feature_set(..., selector=selector)` prevents accidental mixing across extractors or
question recipes. Real model compatibility remains subject to the GPU pilot.

`src/evaluation/features.py` implements the cheap text-statistics baseline: per-column
character/word counts and missingness, plus row totals. It also supplies the proposed negative
control that permutes whole Jev vectors independently within training/test partitions.
`assemble_features(..., view="main")` uses only the agreed main features; explicit views expose
retained distributions/confidence for ablations. Neither utility reads targets or calls an API.
The owner approved six disabled extractors. References live in
`config/2. feature_creation/extractors.yaml`; their paired comparison uses experiment 1.
Experiment 4 is a later analysis of those outputs, requiring no duplicate fits.
Alternative response decoders, separate spending ledgers and live pilots remain required.

## Implemented benchmark contract

Each of the six predictors receives these four configurations for each selected extractor:

1. All original non-text columns, including categorical columns.
2. Those columns plus the joint text features.
3. Those columns plus the per-column text features.
4. Those columns plus both representations.

The first configuration is fitted once per predictor/dataset and shared across extractor
comparisons. Exact duplicate columns within the combined representation are removed; named
arms still have separate fits. TabSTAR and ConTextTab each receive one configuration: every
original feature, including raw text, without decision-model features. Their preprocessing
and TabSTAR's required training see only the relevant training partition. The same five
outer folds and binary inner-OOF threshold procedure apply to both comparisons.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.campaign
.\.venv\Scripts\python.exe -m src.jev.context
```

Both commands are offline. The first lists the plan (2,088 dataset/model/feature/extractor
combinations, 10,440 outer fits if all six extractors pass, plus binary inner fits). The second
screens every frozen unique input and saves `2. feature_creation/manifests/context_screen.json`.
Character-based screening is not exact tokenization or dispatch approval. It includes task
definitions, questions and options, but not provider-internal formatting overhead.

`campaign --execute` requires exactly one selected arm, enabled configuration, accepted
experiment-0 evidence, installed-package pins, verified checkpoint files and completed exact
feature selectors. None is enabled by the planning command. Errors or resource limits must
be recorded rather than addressed by silently sampling rows. ConTextTab uses one complete
training context instead of its default 8,192-row sampling; full-fold memory remains untested.
Kumo rejects its default recipe's 500-column overflow instead of silently selecting features.
TabSTAR uses its default internal validation, so all outer-training rows are supplied, but
some are reserved for stopping decisions rather than gradient updates. Timings reflect each
interface: deferred context computation in Kumo/LimiX occurs during prediction.

## Files and outputs

```text
config/
  0. exploration/         audit, pinned datasets, inherited example settings
  1. question_preparation/ definitions, templates, offline preparation
  2. feature_creation/    disabled extractors and compatibility pilot
  3. experiment_0/        debugging
  4. experiment_1/        main comparison, models and CV protocol
  5. experiment_2/        original-text comparison
  6. experiment_3/        retained-output ablations
  7. experiment_4/        decision-model comparison
notebooks/
  0. exploration/         01_data_exploration, 02_jev_input_design
  1. question_preparation/ 01_prepare_questions
  2. feature_creation/    03_feature_creation
src/                     reusable Python code
scripts/slurm/           disabled-by-default runtime preflights
tests/                  deterministic checks
data/raw/               all 20 original numbered datasets
data/decision_inputs/   18 dataset SQLite files: questions + grouped text + row links
data/jev_cache/         later responses + numeric features, one file per dataset
output_JEVPFN/
  All Results.md
  figures/
    Captions.md
    0. exploration/<notebook>/
    1. question_preparation/<notebook>/
    2. feature_creation/<notebook>/
  1. question_preparation/ logs and compact preparation manifests
  2. feature_creation/    extraction logs and manifests
  3. experiment_0/        debugging logs and manifests
  4. experiment_1/ ...    created when an experiment writes real output
docs/                   five durable uppercase documents
tfm-library/            read-only literature submodule
```

Exploration and feature-design notebooks retain their visual outputs under the figures tree.
Actual local feature builds create `2. feature_creation/logs` and manifests folders. Their
figures, per-notebook reports/configurations and run records are under `figures/<phase>/<notebook>/`;
`All Results.md` stays directly under `output_JEVPFN/`; aggregate captions are in
`output_JEVPFN/figures/Captions.md`. Rebuildable audit tables, request
previews and budget calculations use temporary storage shared during one runner invocation.
They are not permanent research artifacts; all displayed measurements are retained in the notebooks and full report. All notebooks always run
locally and use these local paths, independently of VSC storage environment variables. An
optional `JEVPFN_LOCAL_STORAGE_ROOT` can place these local artifacts on another local disk.
Config, notebook and output phase folders share the same numbers. Logical config IDs such as
`feature_creation` remain valid in code and resolve to the numbered folder. Exploration creates
figures/reports only; empty experiment notebook/output folders are not pre-created.

The small `.reports.lock` file coordinates parallel report writers; the OS releases it after
a crash. Windows sharing errors are retried. A persistently blocked report keeps its previous
content and a named `.pending` recovery copy.

For experiment jobs on VSC, compact manifests and bounded operational logs use
`$VSC_DATA/JEVPFN/output_JEVPFN/<number>. experiment_N/`. Individual predictions, full metric tables,
feature matrices and large results use `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<number>. experiment_N/`.
Raw data, permanent paid caches/features and weights also use the project-storage `JEVPFN/`
directory. Prepared input files use that same project tier. Large writes never fall back to limited DATA. Locally both experiment storage tiers
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
