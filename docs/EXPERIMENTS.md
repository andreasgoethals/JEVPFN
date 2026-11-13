# Experiments and evaluation

This is the scientific design for the current four-experiment study. Executable choices
live in `config/`; [VSC operations](VSC.md) gives commands and [Operational evidence](AGENTS_MEMORY.md)
records what has actually passed. A configured method is not a validated runtime or a result.

## Research question and scope

How useful are compact, task-conditioned decision features extracted from text columns
inside fixed tabular predictors, compared with frozen and supervised text embeddings?
Native-text systems provide a separate comparison of complete prediction pipelines.

The benchmark retains **18 datasets: three binary, six multiclass and nine regression**
tasks. All 20 core MulTaBench text datasets remain in exploration. Spotify and Zomato
were excluded before comparative outcomes because of class-count and context requirements.
The [data guide](DATA_SOURCES.md) documents the pinned inputs and selection.

## Experiment 1: decision features

For each dataset, six predictors receive a shared structured baseline or one of three
feature sets from each of thirteen decision models:

| Feature set | Added decision features |
|---|---|
| `non_text` | None; non-text columns and text-availability flags only |
| Per-column | A separate question/response for each available text field |
| Joint | One question/response for all available named text fields in the row |
| Both | The per-column and joint feature blocks together |

The baseline runs once per dataset/predictor, not once per decision model. The thirteen
extractors are Jev 1.14, Liquid d1-3B, Liquid d1-2B (LFM3), Perplexity Decider, Clef-Flash,
Intern-Decision, Strands, H2O Lightning, SII-Decision, Kev, Decision3, Bespoke-Nimble and Fastino GLiDE-27B.
Exact versions, licences and dependencies are in [extractors.yaml](<../config/2. feature_creation/extractors.yaml>).

GLiDE is the 13 November addition. Its planned grid entries do not constitute runtime
acceptance or completed features. Existing frozen campaign receipts retain their original
roster; GLiDE needs separate 0A evidence before extraction and complete features before 0B.

All models receive the same **semantic request**: fixed task/question definitions,
ordered class descriptions and available named row text. Native serialization and
tokenizers differ. Row targets, non-text row values, target statistics and prediction
feedback are excluded. These are task-conditioned features without dataset-label fitting;
pretrained models' exposure to public labels cannot be ruled out.

A binary response contributes its positive-class probability; multiclass contributes
the full class-probability vector; regression contributes the probability-weighted mean
of nine ordered evidence levels from −4 to +4. Full responses remain stored. Regression
features are evidence scores, not predictions in target units. Provider probabilities
and confidence fields are not automatically calibrated or interchangeable.

## Experiment 2: frozen embeddings

E5-small-v2, E5-large-v2 and Qwen3-Embedding-0.6B encode each text field independently.
Their raw widths are 384, 1,024 and 1,024 respectively. Frozen vectors are reusable across
folds. Long texts use the declared chunk/aggregation policy rather than silent truncation.

| Recipe | PCA input | Embedding columns after PCA |
|---|---|---:|
| Separate PCA32 | Each embedded text column separately | 32 × number of text columns |
| Joint PCA64 | All embedded text columns concatenated | 64 total |

Non-text features bypass PCA. Imputation and PCA fit only on the relevant training
partition; validation/test rows use that fitted transformation and receive the same
columns. Inner threshold folds have their own transformations. Recipes are fixed rather
than selected from test scores. They differ in both grouping and total width except on
two-text-column datasets. [Embedding configuration](<../config/5. experiment_2/embeddings.yaml>)

## Experiment 3: target-aware embeddings (TAR)

One E5-small-v2 LoRA adapter is fitted per dataset and outer fold: **18 × 5 = 90 adapters**.
Each uses outer-training labels and an internal holdout for early stopping. The resulting
vectors are shared across six predictors and the same separate PCA32/joint PCA64 recipes.
Scaling and PCA use outer-training rows only. Outer-test labels never enter adaptation,
checkpoint selection or compression.

The twelve vendored MulTaBench training/encoding files are preserved byte-for-byte at
their recorded commit. JEVPFN adds all-row five-fold CV, the two approved PCA recipes
and its own fixed predictors. Upstream column prefixes, missing placeholders and
512-token truncation are retained for TAR. These differ from Experiment 2, so the
comparison estimates a **complete pipeline effect**, not solely the effect of fine-tuning.

There are no inner TAR refits. Binary secondary decision metrics use a fixed 0.5 threshold;
headline ROC-AUC is threshold-free. [TAR provenance and validation](TAR.md)

## Experiment 4: native-text systems

TabSTAR v1.1, ConTextTab and AutoGluon-Multimodal receive original text and non-text
features. Required adaptation, internal validation or in-context training rows remain
within each training partition. Fixed settings replace an AutoML/HPO search. Publisher
length limits and effective context sizes are recorded.

These are descriptive **whole-system references**. They do not share a predictor with
Experiments 1–3 and cannot isolate representation replacement within that predictor.
[Native-text configuration](<../config/7. experiment_4/native_text.yaml>)

## Predictors and exact grid

The six structured predictors are CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, Kumo Tabular
and linear/logistic regression. [models.yaml](<../config/4. experiment_1/models.yaml>)
contains checkpoint and runtime pins. All augmented structured arms include the same
non-text features and availability flags. No rows or columns may be silently discarded
to fit a model.

One configuration is a dataset × predictor × representation. Each has five outer fits.

| Experiment | Configurations | Outer fits | Binary inner fits |
|---|---:|---:|---:|
| 1 | 18 × 6 × (1 + 13 × 3) = **4,320** | 21,600 | 10,800 |
| 2 | 18 × 6 × 3 × 2 = **648** | 3,240 | 1,620 |
| 3 | 18 × 6 × 1 × 2 = **216** | 1,080 | 0 |
| 4 | 18 × 3 = **54** | 270 | 135 |
| **Total** | **5,238** | **26,190** | **12,555** |

Experiments 2 and 3 reuse Experiment 1's structured baseline. Inner-fit counts are three
binary datasets × configurations per dataset × five outer folds × three inner folds.
TAR's 90 encoder fits are additional, shared work. Counts describe the planned grid,
not completed runs or independent statistical observations; retries and 0B profiles
can add operational work. `src.evaluation.campaign.benchmark_plan` generates the grid;
`src.evaluation.notebook_evidence.study_grid` checks these totals.

Earlier large PCA grids and winner-selected combinations are retired. There is no active
Experiment 5, grouped sensitivity experiment or lexical TF-IDF/SVD arm.

## Cross-validation and missing inputs

All representations use the same five shuffled outer folds, stratified for classification.
All outer-training rows are used. Fitted preprocessing, category handling and scaling use
training rows only; unsupported full-size fits are reported rather than silently subsampled.

In Experiments 1, 2 and 4, binary thresholds maximize F1 on three-fold out-of-fold
predictions from the outer training set. Preprocessing is fitted afresh within each inner
training split. The predictor is then refitted on the full outer training set and the
chosen threshold is held fixed on test rows. Default-0.5 results are also retained for
comparison with TAR. Multiclass uses aligned probability columns and argmax.

Empty decision inputs incur no request and retain missing features. A joint request
omits missing fields; a completely empty row remains missing. Frozen embeddings retain
missingness until fold-local preprocessing. TAR deliberately retains upstream missing
placeholders. Availability flags are shared across structured arms, so added value is
measured beyond simple text presence. Exact request reuse requires matching text,
question and model/settings identities; it never uses targets.

## Metrics, records and statistical interpretation

| Task | Headline metric | Important retained secondary measures |
|---|---|---|
| Binary | ROC-AUC | Average precision, log loss, Brier, F1, precision/recall, specificity, balanced accuracy, MCC, calibration bins and confusion matrix |
| Multiclass | Macro one-vs-rest ROC-AUC | Other AUC/AP aggregations, log loss, multiclass Brier, accuracy, macro/micro/weighted F1, per-class scores and confusion matrix |
| Regression | R² | MAE, RMSE/MSE, median/max absolute error, bias, explained variance and Pearson/Spearman correlation |

The [protocol](<../config/4. experiment_1/protocol.yaml>) defines the complete metric list,
null reasons, class orientation and scale conventions. MAPE/RMSLE require a meaningful
declared ratio scale; quantile metrics require actual quantile predictions. Undefined
metrics are not replaced with zero. Average precision and trapezoidal PR-AUC are distinct.

Saved records retain row IDs, targets, probabilities/predictions, split and feature
identities, checkpoint/settings hashes, loading/fit/predict time and available resource
measurements. Extraction cost is recorded separately once, not charged to every fold.
Checksummed per-dataset/phase SQLite stores support resume and later metric recomputation.

Effects pair the same dataset, predictor, folds and metric. Fold standard deviations
are descriptive because training sets overlap. Optional two-sided Wilcoxon tests use
one mean paired effect per dataset, with Holm correction across the declared contrasts.
The [analysis plan](<../config/8. synthesis/analysis.yaml>) is **not yet approved** and has
no selected contrasts. It must be fixed before comparative outcomes are inspected.
With only three binary datasets, the smallest ideal exact two-sided p-value is 0.25;
folds cannot be used to manufacture extra independent observations.

Shuffled CV permits exact repeats across folds and measures held-out-row performance,
not prospective or unseen-entity generalization. Shared seen/unseen/empty-text cohorts
help interpret results. They do not remove pretraining exposure or establish causality.
No winner-selected follow-up is part of the confirmatory grid.

## Qualification and execution order

| Stage | Required work and evidence |
|---|---|
| Preparation | Pinned raw data, deterministic versioned questions, exhaustive row/missingness and payload audit, matching local/VSC freeze |
| 0A | Local CPU/question and hosted-interface/pilot checks; VSC storage/locking, exact token coverage, real B200 decision and frozen-encoder contracts and dataset pilots; reviewed acceptance |
| Feature creation | Accepted open-weight and frozen-encoder extraction; budgeted local Jev; resumable response/vector stores, verified publication and independent copies |
| 0B preparation | Complete selected feature stores; fold-specific frozen PCA; 90 real TAR adapters and both TAR PCA recipes; sealed predictor runtimes |
| 0B qualification | Integrity/identity and leakage screens, nine predictor contracts, difficult input/class/missingness/resume cases, full-size profiles covering dataset shape and binary inner-OOF fitting, published prediction readback and reviewed acceptance |
| Experiments 1–4 | Enabled, frozen, acceptance-bound arrays; five-fold records; paired descriptive analysis and approved inferential contrasts |

TAR preparation is part of getting 0B ready, not an assumption that its training was
already tested in 0A. The generated qualification plan is authoritative for required
profiles. Local synthetic tests do not replace checkpoint, GPU or full-fold evidence.

## Reading feature diagnostics

The feature-creation notebooks compare only complete downloaded providers on shared rows.
They show vote agreement, matching-class correlations, full-vector distances, confidence,
grouping/reuse effects and hash-verified examples. Ties and missing inputs remain distinct
from consensus. Class imbalance can produce high agreement by chance; correlation and
probability distances answer different questions. Regression sign agreement concerns
evidence direction, not equality of predicted targets. These label-free analyses describe
representations; downstream held-out prediction determines their utility.
