# JEVPFN

Research software for testing whether **label-free probabilistic text features from decision
models improve tabular prediction**. Explore all 20 core MulTaBench TEXT datasets; use 18
for extraction and evaluation. Zomato Restaurants and Spotify Genres remain available locally
for exploration but are excluded from experiments for text burden and 114-class feature size.

**Current stage: Experiment 0A.** The VSC CPU preflight and five tokenizer audits passed on
the previous question version. Version 2 removes provenance URLs from model inputs and requires
a fresh token audit; real GPU compatibility remains unverified. Paid extraction and predictive
experiments remain disabled. Run evidence is recorded in [AGENTS_MEMORY](docs/AGENTS_MEMORY.md).
The owner handles all Git commits and pushes to [GitHub](https://github.com/andreasgoethals/JEVPFN).

## Research contract

Each extractor reads mechanically documented task metadata and either one text column or
all available text columns jointly. Questions are fixed templates; there is no generative
question writer. **Non-text features, current row targets, class frequencies, target
statistics and model feedback never enter extraction.** Documentary class-code meanings
retain their sources and original class vocabulary.

| Task | Main numeric features per text input |
|---|---|
| Binary classification | One probability that the text supports the fixed positive class over the other class. |
| Multiclass classification | All K class probabilities in the original class order. |
| Regression | Probability-weighted direction on the fixed nine levels −4 to +4. |

The binary feature answers an evidence question; it is not established as a calibrated target
probability. The regression score describes directional evidence, not target units. Its rubric distinguishes
indirect hints, qualified support, clear direct evidence and explicit supported comparisons;
zero favors neither direction. Full responses, probability vectors and available confidence
are retained for inspection and later ablations.

Empty text inputs are skipped and yield missing numeric features. Exact equality includes
task context, question/options, named verbatim text and provider/runtime settings. Identical
requests reuse their first verified response. Separate and joint inputs containing the same
single field share one response; their identical numeric columns are included only once.

## Selected models and evaluation

| Decision extractor | Execution |
|---|---|
| Jev 1.13.0 | Hosted TypeSafe API, locally. |
| Liquid d1-3B | Open weights on VSC. |
| Strands Decider 2B | Open weights on VSC. |
| Perplexity Decider v1.1-27B | Open weights on VSC. |
| Cloudflare Clef-flash | Open weights on VSC. |
| Intern-Decision-4B | Open weights on VSC. |

Microsoft-Decision-1 is excluded by owner decision because of access/throughput constraints.
`config/2. feature_creation/extractors.yaml` maintains checkpoint/version pins, limits and
price references. Open weights have no vendor token fee; licences and GPU credits still apply.
Actual model compatibility and complete coverage must be demonstrated by Experiment 0A.

The six main predictors are **CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular,
and logistic/linear regression**. Each receives four configurations:

1. All original non-text columns, including categorical columns.
2. Those columns plus joint decision features.
3. Those columns plus per-column decision features.
4. Those columns plus both representations.

The non-text baseline is fitted once per predictor/dataset and reused across extractor
comparisons. Named augmented arms remain separate fits. **TabSTAR and ConTextTab** are the
two original-text baselines: they receive all original features, including text, without
extracted features. Cheap character/word/missingness features and partition-local permutation
controls are implemented for separate ablations.
`python -m src.evaluation.campaign --include-controls` includes those Experiment 3 arms and
retained-probability views. On one-text-column datasets, separate/joint/both contain identical
information; exclude them from representation contrasts, while retaining their named fits.

Use shared five-fold shuffled stratified CV for binary/multiclass classification and shuffled
KFold for regression. Binary thresholds maximize F1 from three inner out-of-fold splits,
then the estimator is refitted on the complete outer-training fold. Thresholds are frozen
before outer-test prediction. All fitted preprocessing stays inside its training partition.
Use all training rows, fixed library settings, no HPO and no silent context/feature sampling.
TabSTAR's required task adaptation is recorded as part of its method.

Record full outer-test and binary inner-OOF predictions, all implemented probability/decision
or regression metrics, undefined-metric reasons, timings, resource use, folds and feature
identities. Headline metric selection remains deferred until before comparative predictive results are
reviewed. Exact conventions and runtime settings live in `config/4. experiment_1/`.
Full-fold feasibility, checkpoint loading and native-text encoders remain unverified on VSC.

## Workflow

Exploration → question preparation → Experiment 0A → full feature creation → saved-feature
analysis and Experiment 0B → predictive experiments. The folder numbers organize files;
Experiment 0A deliberately runs before the numbered feature-creation phase is executed.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](<notebooks/0. exploration/01_data_exploration.ipynb>) | All 20 datasets: text length, missingness, exact reuse, long text and feature dimensions. |
| [02_jev_input_design](<notebooks/0. exploration/02_jev_input_design.ipynb>) | Task examples and complete six-extractor request previews for all 18 selected datasets. |
| [01_prepare_questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) | Source-backed meanings, frozen questions, grouped inputs and original-row links. |
| [03_feature_creation](<notebooks/2. feature_creation/03_feature_creation.ipynb>) | Exact saved tokenizer audits, duplicate-aware workload, cost scenarios and storage readiness. |
| [04_decision_feature_comparison](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>) | Saved-feature coverage, agreement, distributions and distance from other extractors. |

All notebooks run locally. They perform no inference or model training. Reusable logic is in
`src/`; every final cell prints the complete recorded analysis. Feature comparison verifies
row alignment, matching input hashes, class order and artifact versions; disagreement is
not evidence of predictive accuracy. Ambiguous versions require explicit artifact selection.

## Local setup and commands

Open this inner repository folder. Create Python 3.12 with the prompt **JEVPFN**:

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Existing environments do not need recreation. Select **Python 3.12 (JEVPFN)** in Jupyter/VS Code.
The no-argument runner executes notebooks in parallel using available CPUs, prints progress,
and saves rich outputs back into each source `.ipynb`. Reload an already-open notebook after
execution. Partial failures preserve completed outputs and a recovery copy; concurrent edits
are not overwritten. `--only 01_data_exploration` selects a notebook;
`--summaries-only` rebuilds saved aggregate reports.

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.cluster.preflight --local --verify-data
.\.venv\Scripts\python.exe -m src.evaluation.campaign
```

The campaign command lists the planned arms without fitting models. Dependency extras in
`pyproject.toml` keep notebook, CPU-tokenizer and model environments separate. Do not install
all model extras into the notebook environment. [VSC](docs/VSC.md) gives the cluster commands.

## Durable inputs and outputs

- `data/raw/`: all 20 numbered dataset folders, each with CSV, metadata and download provenance.
- `data/decision_inputs/<dataset_id>.sqlite3`: exactly 18 files containing versioned questions,
  unique named text groups and every original row/representation link.
- `data/jev_cache/<dataset_id>.sqlite3`: permanent provider-specific responses and numeric features.
- `output_JEVPFN/figures/<number>. phase/<notebook>/`: PDF figures and local analysis records.
- `output_JEVPFN/figures/Captions.md` and `output_JEVPFN/All Results.md`: aggregate notebook reports.
- `output_JEVPFN/<number>. phase/`: operational logs/manifests and experiment results.

Configurations use `0. exploration`, `1. question_preparation`, `2. feature_creation`,
`3. experiment_0`, then `4. experiment_1`, etc. Notebook/output folders are created when used.
Exploration writes only figures/reports. Temporary audit tables and budget intermediates are
recomputed during a notebook run and discarded; the notebooks and report retain their values.
Raw datasets, prepared inputs and permanent features are never temporary cleanup targets.
`python -m src.utils.clean_run` previews cleanup; `--clean` removes only rebuildable figures,
reports and legacy question JSON copies. Logs, manifests, token audits, predictions and transfer
bundles/backups are preserved. It never recursively deletes a data directory.
For a completely fresh VSC test run, `python -m src.cluster.fresh_start` previews moving
both output trees to dated `archived_output_JEVPFN/` folders on their existing storage
tiers. `--apply --writers-stopped` performs those reversible moves and recreates the
Experiment 0 log directory. It refuses queued/running JEVPFN jobs; stop manual writers too.
Data, questions, features, weights and environments remain in place.

Original row identity is **dataset ID + source-data fingerprint + csv:N**, where N is the
zero-based CSV position before any split/shuffle. `row_map` links each row/slot to a unique
input; NULL denotes empty text. Features join by these IDs, with duplicate/missing IDs rejected.
Provider/checkpoint/settings enter response keys. SQLite stores raw responses, UTC timestamps,
usage and origins, plus immutable compressed-Parquet feature tables, manifests and checksums.
Each provider/version/representation can be selected independently in downstream experiments.
An explicit active input-set ID selects the current question version; historical sets remain
in the same 18 files. Bulk feature reads do not add execution origins. Feature identities include
the complete runtime-settings hash and relevant transformation code, not unrelated plotting code.

On VSC, **PROJECT** holds raw/prepared inputs, permanent responses/features, checkpoints,
CUDA environments, exact token databases, predictions and large results. **DATA** holds the
checkout, small CPU environment, compact manifests and logs. Both trees use `JEVPFN/` and
`output_JEVPFN/<number>. phase/`. Local execution uses local storage; notebook paths remain
local even if VSC variables are present. Large VSC writes fail rather than fall back to DATA.

Use `python -m src.cluster.transfer export` on VSC, download its bundle, then
`python -m src.cluster.transfer import <bundle.zip>` locally. Imports validate checksums,
back up existing dataset caches and merge provider/version records transactionally. Never
replace a populated cache wholesale with SCP. [Transfer instructions](docs/VSC.md#transfer-results).
Question uploads use the separate `src.cluster.input_transfer pack` / `install` commands:
exactly the selected 18 input files, byte-preserving copies, checksums and remote backups.

## Execution readiness and references

The bounded GPU contract job tests all three tasks, a 30-class vector, missing/duplicate inputs,
cache recovery, feature-table round trips and real first/longest inputs from every dataset.
Seven known-answer/polarity/option-order probes supplement deterministic decoder tests.
It plans at most 74 forwards per provider, with a hard ceiling of 100. A passing report requires
review; it does not authorize full extraction or establish predictive-model compatibility.
A separate two-node CPU storage test verifies PROJECT lock exclusion and SQLite
snapshot/restore on the actual mount; local tests cannot establish that behavior.

Still required before full extraction: real contracts/robustness and throughput tests for
selected extractors, Jev account/billing/context checks, an approved spending limit, and
verified storage quota/backup. Before evaluation: complete cache transfer/coverage, predictor
checkpoints and environments, full-fold resource checks and pretraining-overlap review.
Failures remain explicit; text is never silently truncated or replaced by missing values.
Self-hosted failures persist by request identity. The optional `--continue-on-error` records/skips
them but leaves the campaign incomplete; `--retry-failed` explicitly retries them. Provider jobs
rotate datasets and wait for busy stores, retaining one writer per dataset.

Future execution entry points are implemented and remain guarded: `src.jev.extraction` for the
local Jev pilot/full run; `src.jev.self_hosted` for open weights; `src.cluster.predictor_contracts`
for bounded real predictor checks; and `src.evaluation.campaign --profile-fold` for one complete
outer fold in Experiment 0B. CV resumes checksummed complete folds and saves warnings, sampled
process RSS and CUDA peaks. `src.evaluation.results --save` summarizes complete five-fold arms
and paired feature differences, without treating unfinished arms as complete. Exact setup and
acceptance requirements are in [VSC](docs/VSC.md).

Documentation has distinct purposes: [DATA_SOURCES](docs/DATA_SOURCES.md) records dataset
provenance/semantics; [LITERATURE_REVIEW](docs/LITERATURE_REVIEW.md) records published evidence;
[VSC](docs/VSC.md) documents operations; [AGENTS_MEMORY](docs/AGENTS_MEMORY.md) records run/failure
evidence; [CHANGELOG](docs/CHANGELOG.md) records changes. Proposed plans stay in chat.

Source code is MIT-licensed; datasets and weights retain their own licences. Cite the software
using [CITATION.cff](CITATION.cff). Data, weights, outputs and credentials are Git-ignored.
Git transfers code only; the owner publishes it. The repository retains conventions from
[Andreas' research template](https://github.com/andreasgoethals/0.-Template).
