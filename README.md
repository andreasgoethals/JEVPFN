# JEVPFN

JEVPFN measures how much **label-free decision features from text** add to fixed tabular
predictors. It compares those features with frozen embeddings, supervised target-aware embeddings and native-text systems
on 18 of MulTaBench's 20 core text datasets.

## Project status

Experiment 0A has been accepted for the eleven open-weight extractors and three encoders.
The local Jev 1.14 pilot also passed on the first 100 rows of each dataset. Feature
creation uses a frozen VSC checkout. The 5 November audit found all frozen embeddings
and three open-weight providers complete; Perplexity and Intern need operational recovery.
Once extraction finishes, the next stage is to verify the
published features, prepare fold-local PCA and TAR, and qualify the benchmark in 0B.
TAR is implemented locally; its new runtime and training jobs still need real 0B validation.
The owner runs full hosted Jev extraction separately. [Run evidence](docs/AGENTS_MEMORY.md) ·
[VSC workflow](docs/VSC.md#46-experiment-0b)

## Local setup

From the repository root in Windows PowerShell, create a Python 3.12 environment once.

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

Check `$LASTEXITCODE` after each command. Preparation reuses verified downloads.
`src.jev.preparation --rebuild` regenerates the question stores transactionally while
preserving paid responses and features. For the full shared package lock and the
Windows/Linux interpreter difference, see [CPU environment parity](docs/VSC.md#cpu-environment-parity).

## Benchmark design

| Experiment | Comparison | Configurations |
|---|---|---:|
| 1 | A shared structured baseline versus decision features from twelve extractors, using text jointly, per column, or both | 3,996 |
| 2 | Three frozen encoders with joint PCA64 or separate PCA32 per text column, compared with the same structured baseline | 648 |
| 3 | E5-small-v2 TAR (LoRA), with the same two PCA recipes and six predictors | 216 |
| 4 | Three native-text systems on the original features | 54 |

One configuration is a dataset × predictor × representation. The grid contains **4,914
configurations**, with **24,570 outer fits** and **11,745 binary inner fits**. The structured
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

### Decision features

Each frozen decision model answers a fixed question using task and class definitions
plus one named text field or all available text fields. Requests exclude row targets,
class frequencies, non-text row values and model feedback. A binary response contributes
the positive-class probability, multiclass contributes the complete probability vector,
and regression contributes the probability-weighted mean of nine evidence levels from −4
to +4. These scores express the model's evidence. Regression scores use the rubric scale,
and calibration on benchmark targets requires separate evidence.

Empty text yields missing features. Identical requests reuse the stored response, including
joint/per-column matches. Context overflow stops decision extraction. Frozen encoders cover
long text in chunks and cache their vectors before PCA. Jev 1.14.0 runs locally through
TypeSafe. The eleven other extractors run on VSC B200. See the
[extractor roster and pins](<config/2. feature_creation/extractors.yaml>) and
[dataset and question definitions](docs/DATA_SOURCES.md).

### Evaluation and interpretation

All arms use the same five shuffled outer folds, stratified for classification. PCA,
imputation and other fitted preprocessing use training rows only. Binary F1 thresholds
use out-of-fold predictions from three inner splits of the outer training partition in
Experiments 1, 2 and 4. Experiment 3 preserves MulTaBench’s single encoder fit per outer
fold with internal early stopping; binary secondary metrics use a fixed 0.5 threshold.
TAR adds **90 shared encoder fits**, reused across predictors and PCA recipes. See
[TAR provenance, differences and preparation](docs/TAR.md).

Headline metrics are binary ROC-AUC, multiclass macro one-vs-rest ROC-AUC, and regression R².
The benchmark retains predictions, probabilities, secondary metrics, timings and resources.
Effects pair the same dataset, predictor and folds. Fold standard deviations describe
variation. Optional Wilcoxon/Holm tests use one effect per dataset and require an approved
[analysis plan](<config/8. synthesis/analysis.yaml>) before scores are inspected.

Shuffled CV measures performance on held-out rows from the released datasets. Exact repeats
can cross folds. The benchmark reports seen/unseen-text cohorts and 0B checks for exact
target copies and class recodings. Prospective feature availability and pretrained model
exposure require separate evidence. [Interpretation and limitations](docs/LITERATURE_REVIEW.md#6-leakage-protection-uncertainty-and-valid-interpretation)

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

| Stage | Notebooks |
|---|---|
| Explore and prepare | [Data](<notebooks/0. exploration/01_data_exploration.ipynb>), [input design](<notebooks/0. exploration/02_jev_input_design.ipynb>), [questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) |
| Extract and validate | [Workload and feature widths](<notebooks/2. feature_creation/03_feature_creation.ipynb>), [decision features](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>), [0A](<notebooks/3. experiment_0/01_before_feature_creation.ipynb>), [0B](<notebooks/3. experiment_0/02_after_feature_creation.ipynb>) |
| Evaluate | [Decision features](<notebooks/4. experiment_1/01_paired_representation_effects.ipynb>), [frozen embeddings](<notebooks/5. experiment_2/01_frozen_embeddings.ipynb>), [TAR](<notebooks/6. experiment_3/01_target_aware_embeddings.ipynb>), [native text](<notebooks/7. experiment_4/01_native_text_systems.ipynb>), [synthesis](<notebooks/8. synthesis/01_representation_comparison.ipynb>) |

[Changelog](docs/CHANGELOG.md) · [Citation](CITATION.cff) · [License](LICENSE)

The project follows the [research template](docs/TEMPLATE.md) with numbered phase folders,
`output_JEVPFN`, project-specific aggregate report names, permanent paid-response stores,
and separate local/VSC execution gates. Data and weights retain their source licences.
