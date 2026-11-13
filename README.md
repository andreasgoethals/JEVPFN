# JEVPFN

JEVPFN compares **task-conditioned decision features, frozen embeddings, supervised
target-aware embeddings and native-text systems** on 18 selected MulTaBench datasets.
The study asks how these representations help fixed tabular predictors on shared
cross-validation splits. Decision features use fixed task definitions and row text
without fitting to dataset labels; TAR is a separate supervised comparison.

## Project status

**Last recorded evidence: 12 November 2026.** Experiment 0A was accepted for eleven
open-weight extractors and three frozen encoders. The local Jev 1.14 first-100-row pilot
also passed. Eight open-weight providers and all 54 frozen encoder/dataset pairs are
complete; all 54 frozen-PCA preparation jobs passed. Kev has two unfinished datasets,
Nimble one, and Liquid LFM3 eighteen. Scoped staging recovery and independent 0B/TAR
preparation are authorized. These are dated observations, not live job status.
Full benchmark execution still requires complete feature checks and reviewed 0B evidence.
Fastino GLiDE-27B was added to the local planned roster on 13 November. It requires its
own token audit, sealed runtime, B200 contracts and dataset pilot; the active VSC campaign
remains on its recorded eleven-open-model scope until a safe handover.
Local Jev extraction is authorized within the existing EUR 5 allowance; full coverage
will need additional funding. [Run evidence](docs/AGENTS_MEMORY.md) ·
[VSC workflow](docs/VSC.md#46-experiment-0b)

## Benchmark design

| Experiment | Comparison | Configurations |
|---|---|---:|
| 1 | A shared structured baseline versus decision features from thirteen extractors, using text jointly, per column, or both | 4,320 |
| 2 | Three frozen encoders with joint PCA64 or separate PCA32 per text column, compared with the same structured baseline | 648 |
| 3 | E5-small-v2 TAR (LoRA), with the same two PCA recipes and six predictors | 216 |
| 4 | Three native-text systems on the original features | 54 |

One configuration is a dataset × predictor × representation. The grid contains **5,238
configurations**, with **26,190 outer fits** and **12,555 binary inner fits**. The structured
baseline includes non-text features and the same text-availability flags as the decision
and embedding arms. The experiments use fixed settings and prespecified recipes.

- **Six structured predictors** — CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo
  Tabular, and logistic/linear regression. LimiX-2 preprocessing is restricted to training rows.
- **Three frozen encoders** — E5-small-v2, E5-large-v2, and Qwen3-Embedding-0.6B.
  Joint PCA yields 64 embedding features total. Separate PCA yields 32 per text column.
- **Three native-text references** — TabSTAR v1.1, ConTextTab, and AutoGluon-Multimodal.
  Experiment 4 compares complete systems descriptively.

[Predictor settings](<config/4. experiment_1/models.yaml>) ·
[Embedding grid](<config/5. experiment_2/embeddings.yaml>) ·
[Model interfaces and prior work](docs/LITERATURE_REVIEW.md)

The full [experiment guide](docs/EXPERIMENTS.md) explains the grid arithmetic, feature
schemas, missing inputs, training boundaries, metrics and acceptance gates. It is the
single reference for the current scientific design. [TAR](docs/TAR.md) records exact
upstream provenance and declared differences; the [literature review](docs/LITERATURE_REVIEW.md)
explains the model choices and limits of the contribution.

## Local setup

From the repository root in Windows PowerShell, create a Python 3.12 environment once.
These dependencies support local inspection and tests; model runtimes are separate.

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks,control]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Prepare the pinned data and questions, audit the inputs, and run the notebooks.

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.jev.question_audit
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
.\.venv\Scripts\python.exe -m pytest -q
```

Check `$LASTEXITCODE` after each command and stop on failure. These commands do not launch
paid API calls or VSC inference. Preparation reuses verified downloads.
`src.jev.preparation --rebuild` regenerates the question stores transactionally while
preserving paid responses and features. For the full shared package lock and the
Windows/Linux interpreter difference, see [CPU environment parity](docs/VSC.md#cpu-environment-parity).

## Workflow and outputs

Prepare inputs, qualify them and the runtimes in **0A**, create features, then run **0B**
on the complete stores and predictor runtimes. Experiments 1–4 require reviewed 0B evidence.
The [VSC runbook](docs/VSC.md) gives commands and acceptance criteria. The
[operational record](docs/AGENTS_MEMORY.md) preserves dated receipts and recovery history.

| Location | Contents |
|---|---|
| `data/raw/` | Pinned source datasets and provenance |
| `data/decision_inputs/` | One versioned question SQLite store per selected dataset |
| `data/jev_cache/` | Permanent response, accounting and feature stores, plus Jev diagnostic ledgers |
| `data/decision_features/` | Verified portable snapshots of complete models for local analysis |
| `data/text_embeddings/` | Frozen vectors per encoder and dataset |
| `output_JEVPFN/<phase>/` | Validation evidence, logs, PCA parameters and resumable prediction records |

On VSC, DATA holds Git and compact records. GPFS PROJECT holds permanent data, models,
containers and results. Native scratch holds temporary working stores. Use the
[verified export/import workflow](docs/VSC.md#6-utilization-and-transfer) to move results.

### Notebooks

Notebooks run locally and write paper-width PDFs and printed reports. The runner supports
`--only NAME` and `--summaries-only`. Figures and captions are under
`output_JEVPFN/figures/`, and the aggregate report is `output_JEVPFN/All Results.md`.
Result figures appear as complete saved evidence becomes available.

The feature-comparison notebook examines row-level vote splits, matching-class
correlations, full probability differences, confidence, text grouping and exact reuse.
It includes hash-verified source examples and exports all joint rows to compressed
Parquet. Only fully downloaded extractors enter comparisons; agreement is not accuracy.

| Stage | Notebooks |
|---|---|
| Explore and prepare | [Data](<notebooks/0. exploration/01_data_exploration.ipynb>), [input design](<notebooks/0. exploration/02_jev_input_design.ipynb>), [questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) |
| Extract and validate | [Workload and feature widths](<notebooks/2. feature_creation/03_feature_creation.ipynb>), [decision features](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>), [0A](<notebooks/3. experiment_0/01_before_feature_creation.ipynb>), [0B](<notebooks/3. experiment_0/02_after_feature_creation.ipynb>) |
| Evaluate | [Decision features](<notebooks/4. experiment_1/01_paired_representation_effects.ipynb>), [frozen embeddings](<notebooks/5. experiment_2/01_frozen_embeddings.ipynb>), [TAR](<notebooks/6. experiment_3/01_target_aware_embeddings.ipynb>), [native text](<notebooks/7. experiment_4/01_native_text_systems.ipynb>), [synthesis](<notebooks/8. synthesis/01_representation_comparison.ipynb>) |

## Documentation

| Document | Purpose |
|---|---|
| [Experiments](docs/EXPERIMENTS.md) | Scientific design, grid, evaluation and qualification gates |
| [Data sources](docs/DATA_SOURCES.md) | Dataset selection, text rules and question semantics |
| [Literature review](docs/LITERATURE_REVIEW.md) | Evidence, model choices and interpretation |
| [TAR](docs/TAR.md) | Upstream provenance and explicit project differences |
| [VSC operations](docs/VSC.md) | Access, storage, runtimes, commands and recovery |
| [Operational evidence](docs/AGENTS_MEMORY.md) | Dated results, receipts and unresolved validation |
| [Changelog](docs/CHANGELOG.md) | Historical implementation changes |
| [Agent instructions](AGENTS.md) | Authorization boundaries and maintenance rules |

## Citation, licensing and template

Use [CITATION.cff](CITATION.cff) to cite this software. Project code is [MIT licensed](LICENSE);
datasets, weights and vendored code retain their own licences and attribution.

The project follows the [research template](docs/TEMPLATE.md) with numbered phase folders,
`output_JEVPFN`, project-specific aggregate report names, permanent paid-response stores,
and separate local/VSC execution gates.
The project also retains its staging-root/publication interface and its explicit destructive
output-reset helper. Unlike routine template cleanup, that helper can remove result files;
use it only for an authorized reset with required backups. Automatic desktop job monitoring is disabled;
current chat and platform instructions govern any requested monitoring.
