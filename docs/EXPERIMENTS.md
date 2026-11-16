# Experiments and evaluation

This is the scientific design for the three-question study and its four core benchmark comparisons. Executable choices
live in `config/`; [VSC operations](VSC.md) gives commands and [Operational evidence](AGENTS_MEMORY.md)
records what has actually passed. A configured method is not a validated runtime or a result.

## Three research questions, one study

**RQ1 — How do decision models represent real-world text, and what systematic behaviours
and limitations do their outputs exhibit?** Describe probability distributions, confidence,
class preferences, agreement, disagreement and sensitivity to equivalent presentations.
A different output distribution is not by itself evidence of bias or error.

**RQ2 — When and how are decision-model representations useful for prediction?** Compare
representations within fixed predictors, then determine whether gains depend on text
availability, length, repeated text and baseline difficulty. This is the existing benchmark,
with explicit conditional analyses rather than only an average leaderboard.

**RQ3 — Can supervised adaptation improve Strands 2B decisions for tabular text?**
Compare calibration, head-only training and LoRA plus head against the unchanged model.
Use held-out rows and fixed training budgets; fine-tuning is not assumed to help.
The regression probability-versus-mean ablation belongs to RQ2.

The benchmark retains **18 datasets: three binary, six multiclass and nine regression**
tasks. All 20 core MulTaBench text datasets remain in exploration. Spotify and Zomato
were excluded before comparative outcomes because of class-count and context requirements.
The [data guide](DATA_SOURCES.md) documents selection. Multiple rows from one dataset
are not independent datasets for cross-task inference.

## RQ1: behaviour

| Study | What is measured | Execution |
|---|---|---|
| RQ1-A: distributions and agreement | Per-dataset class frequencies, probability concentration, ties, confidence, pairwise total variation and matched-class correlations; both row-weighted and unique-input views | Local analysis of complete saved stores; no new inference |
| RQ1-B: information and context | Per-column versus joint outputs; whether disagreement coincides with missing/short/repeated text; direct classification discrimination and calibration where labels apply | Local paired analysis; label-based findings kept separate from label-free descriptions |
| RQ1-C: controlled invariance | Same underlying input under option order, field order and a reviewed question paraphrase; class flips and aligned probability changes | New bounded VSC inference; Jev requires an explicit remaining budget |

RQ1-C selects at most 100 unique input identities per dataset, deterministically and
without targets or observed model outcomes. Each applicable intervention has one paired
variant; option order applies to unordered classification options, not reordered ordinal
rubrics. Field order requires multiple available fields. Paraphrases require a recorded
semantic-equivalence review. An upper bound is 18 × 15 × 100 × 3 = **81,000 variant requests**, before
removing inapplicable cases; baseline responses are reused. Hosted calls need budget approval. This is a
budget ceiling, not an already implemented/submitted inference grid. The existing 0A
robustness diagnostics are interface evidence, not this full research experiment.

Probability normalization is not calibration. A single text field may not contain enough
information to recover the row outcome. Regression evidence levels −4…+4 are not bins of
observed target values, so direct regression calibration against those labels would be
invalid. Analyse their distribution and downstream utility instead. Confidence, agreement
and correctness remain separate quantities. Aggregate within datasets before giving
datasets equal weight; report task types separately.

The sampling/alignment contracts are implemented in `src/evaluation/research_questions.py`.
The dedicated controlled-query runner and label-linked behavioural panels remain required;
`config/rq1/behavior.yaml` records this explicitly and keeps new queries disabled.

## RQ2: predictive usefulness

The four established experiments below are RQ2-A through RQ2-D. Their machine IDs
`experiment_1`…`experiment_4` stay unchanged in scientific identities. Config files live
under `config/rq2/`; one `notebooks/rq2_prediction.ipynb` presents all RQ2 experiments.
New result trees live under `output_JEVPFN/rq2/<phase>/results/`. Existing trees move only
at an idle, checksum-verified handover; the active extraction paths remain unchanged.

### RQ2-A / Experiment 1: decision features

For each dataset, six predictors receive a shared structured baseline or one of three
feature sets from each of fifteen decision models:

| Feature set | Added decision features |
|---|---|
| `non_text` | None; non-text columns and text-availability flags only |
| Per-column | A separate question/response for each available text field |
| Joint | One question/response for all available named text fields in the row |
| Both | The per-column and joint feature blocks together |

The baseline runs once per dataset/predictor, not once per decision model. The fifteen
extractors are Jev 1.14, Liquid d1-3B, Liquid d1-2B (LFM3), Perplexity Decider, Clef-Flash,
Intern-Decision, Strands, H2O Lightning, SII-Decision, Kev, Decision3, Bespoke-Nimble, Fastino GLiDE-27B, Baseten Mammoth-1 and Nace Drex Max.
Mammoth-1 and Drex Max have pinned native adapters and isolated runtime dependencies.
Their configuration entries remain disabled until separate real 0A evidence is accepted. Exact versions, licences and dependencies are in [extractors.yaml](<../config/general/2. feature_creation/extractors.yaml>).

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

### RQ2-B / Experiment 2: frozen embeddings

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
two-text-column datasets. [Embedding configuration](<../config/rq2/5. experiment_2/embeddings.yaml>)

### RQ2-C / Experiment 3: target-aware embeddings (TAR)

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

### RQ2-D / Experiment 4: native-text systems

TabSTAR v1.1, ConTextTab and AutoGluon-Multimodal receive original text and non-text
features. Required adaptation, internal validation or in-context training rows remain
within each training partition. Fixed settings replace an AutoML/HPO search. Publisher
length limits and effective context sizes are recorded.

These are descriptive **whole-system references**. They do not share a predictor with
Experiments 1–3 and cannot isolate representation replacement within that predictor.
[Native-text configuration](<../config/rq2/7. experiment_4/native_text.yaml>)

### RQ2-E: when does text help?

Use paired out-of-fold predictions from the same rows and folds. For classification,
row-level benefit is the structured baseline's log loss minus the augmented model's log
loss. For regression use squared error normalized by a scale estimated on training rows.
AUC is a cohort-level measure, not a per-row loss. Fixed cohorts describe availability,
training-defined text-length quantiles and seen/unseen text; classification baseline
uncertainty cutpoints must come from training out-of-fold predictions. The same rules
apply across datasets, rather than hand-picking favourable subgroups for each task.
Report within-dataset effects, coverage and equal-dataset summaries. These are observational
associations, not causal effects of text. The cohort boundary helper is tested locally;
full loss/cohort report integration still needs implementation before its analysis freeze.

### RQ2-F: mean versus nine probabilities

The main regression feature is E[S] for the fixed evidence levels S = −4,…,+4. A response
concentrated at zero and a response split equally between −4 and +4 both have mean zero,
but express different uncertainty. Retain all nine probabilities to test whether that
information helps prediction. They sum to one (eight degrees of freedom), but we retain
all nine named coordinates for consistent class semantics. This is **not eight extra
questions** and costs no additional decision-model inference.

Use the nine regression datasets, all fifteen extractors, the per-column mode, and two
prespecified predictors (linear and CatBoost). Existing RQ2 mean arms are the comparators;
**9 × 15 × 2 = 270 new configurations / 1,350 outer fits** are required. The new arms use
the same source features, split seed, rows, predictor settings and missingness flags.
They are planned by `regression_readout_plan` and executable through the existing
acceptance-gated campaign with `--phase regression_readout`; execution remains disabled.

The widest dataset is Rotten Tomatoes: 13 text and 2 non-text fields. Including 13 text
availability flags, separate nine-probability blocks give 2 + 13 + 13×9 = **132 columns**.
Joint-only gives 24; both gives 141. A conservative Kumo count-encoding allowance raises
the largest bound to **143**, below its 500-column limit. Actual model preprocessing
still checks width. This bound is for one extractor, not concatenating all extractors.
See `regression_width_table()` and the RQ2 notebook for all nine datasets.

## RQ3: Strands 2B adaptation

### RQ3-A: calibration; RQ3-B: head-only; RQ3-C: LoRA plus head

Use Strands 2B because its released pointer-head model and training code support explicit
head-only and backbone-LoRA adaptation. This is a controlled case study, not evidence
that all decision architectures respond alike. Compare the original model, temperature
scaling, head-only training, and LoRA plus head. Start from the exact released checkpoint
in each outer fold, using joint text requests and the original class definitions.

Start with Jigsaw and Product Sentiment, one fold each: two datasets × two trained variants
= **four pilot training runs**. If real B200, save/reload and leakage checks pass, expand
to the nine classification datasets × five folds × two trained variants = **90 training
runs**, plus 45 cheap temperature fits. Optimizer, internal validation/calibration and
outer-test rows must be disjoint; all selection happens inside outer training. Record
compute and data budgets, trainable tensors, initial checkpoint, adapters and predictions.
No silent input truncation or unapproved row cap is allowed. Regression adaptation is
excluded because the nine-level rubric is not a supervised target-bin definition.

Evaluate adapted decisions directly first. Feeding their in-sample training predictions
into a downstream predictor would give the learner overly optimistic features. A later
stacked-feature experiment would need inner cross-fitting (three inner fits plus an
outer refit: 360 training runs for the two trained variants), and is **not included**.
This differs from the existing TAR pipeline, which evaluates a supervised representation
trained within outer training; do not call the two adaptation mechanisms identical.

The training design and partition contracts are configured/tested; the dedicated Strands
training runner, sealed training environment and real pilot are **not yet implemented or
accepted**. B200 feasibility is plausible from the publisher's 2B training recipe, but
runtime and memory must be measured. No duration estimate is claimed from the unrelated
TAR pilot. Scientific execution is disabled in `config/rq3/adaptation.yaml`.

## Predictors and exact grid

The six structured predictors are CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, Kumo Tabular
and linear/logistic regression. [models.yaml](<../config/rq2/4. experiment_1/models.yaml>)
contains checkpoint and runtime pins. All augmented structured arms include the same
non-text features and availability flags. No rows or columns may be silently discarded
to fit a model.

One configuration is a dataset × predictor × representation. Each has five outer fits.

| Experiment | Configurations | Outer fits | Binary inner fits |
|---|---:|---:|---:|
| RQ2-A / 1 | 18 × 6 × (1 + 15 × 3) = **4,968** | 24,840 | 12,420 |
| 2 | 18 × 6 × 3 × 2 = **648** | 3,240 | 1,620 |
| 3 | 18 × 6 × 1 × 2 = **216** | 1,080 | 0 |
| 4 | 18 × 3 = **54** | 270 | 135 |
| RQ2-F / regression readout | 9 × 15 × 2 = **270** | 1,350 | 0 |
| **RQ2 total** | **6,156** | **30,780** | **14,175** |

Experiments 2 and 3 reuse Experiment 1's structured baseline. Inner-fit counts are three
binary datasets × configurations per dataset × five outer folds × three inner folds.
TAR's 90 encoder fits are additional, shared work. Counts describe the planned grid,
not completed runs or independent statistical observations; retries and 0B profiles
can add operational work. `src.evaluation.campaign.benchmark_plan` generates the grid;
`src.evaluation.notebook_evidence.study_grid` checks these totals.


Earlier large PCA grids, winner-selected combinations and the former Experiments 4/5 ablation plans are retired. The new RQ1/RQ3 studies are explicitly defined above; no grouped sensitivity or lexical TF-IDF/SVD arm is included.

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

The [protocol](<../config/rq2/4. experiment_1/protocol.yaml>) defines the complete metric list,
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
The [analysis plan](<../config/rq2/8. synthesis/analysis.yaml>) is **not yet approved** and has
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
| RQ2, including regression readout | Enabled, frozen, acceptance-bound arrays; five-fold records; paired descriptive analysis and approved inferential contrasts |
| RQ1-C / RQ3 | Additional controlled-query/training implementations, supplementary 0A and 0B gates, then bounded pilots before full scientific runs |

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

### Additional Experiment 0 coverage

The relocation preserves old output IDs but changes the source/configuration fingerprint;
review it at an idle handover, never underneath a running campaign. New real 0B profiles
include the largest/tallest regression probability-vector arms for linear and CatBoost.
Predictor contracts now include a nine-probability-per-field fixture with missing blocks.
Record identities distinguish the probability view from the mean view.

`config/general/3. experiment_0/research_questions.yaml` separates local contracts from
outstanding real gates. RQ1 needs inverse option alignment, stable label-free sampling,
paired missingness and reviewed semantic equivalence. RQ3 adaptation needs released-weight
initialization, frozen-backbone checks, LoRA gradient checks, disjoint partitions,
held-out-label invariance, no truncation, resume and adapter replay. Implementing a test
plan or passing synthetic tests does not establish these future GPU behaviours.

## Stage order and notebook ownership

0A qualifies extraction requests, exact counts, native runtimes, probability decoding,
batching/replay and dataset pilots. Feature creation publishes complete stores. 0B then
checks feature identities and missingness, fold-local fitting, PCA/TAR, all predictor
interfaces, metrics, resume/publication and each enabled RQ2 arm. It must also exercise
the RQ1 analysis and controlled-query contracts before those are accepted.

Run RQ1 first to characterize the representations, then RQ2 using the prespecified grid.
RQ1 findings may motivate future studies but must not silently select RQ2 winners or
change its questions. RQ3 follows as the prespecified Strands case study. Its separate
training gate tests gradients, frozen weights, split isolation, save/reload and resume
on two bounded pilots before the 90-run expansion. Later code changes require targeted
requalification, not a redundant complete output reset.

There is one report notebook per RQ: `rq1_behavior`, `rq2_prediction`, `rq3_adaptation`.
General preparation and validation notebooks remain in `notebooks/general/`. Figures,
figure registries, captions and report intermediates are grouped by RQ/general. Missing
results are shown as pending evidence, never simulated findings.
