# Literature Review

Evidence checked 30 September 2026. This is a durable reference for the research question,
published evidence, model interfaces and methodological limitations. It is not an experiment
schedule. Agreed project constraints live in [README](../README.md), data definitions in
[DATA_SOURCES](DATA_SOURCES.md), runtime/storage instructions in [VSC](VSC.md), and the candidate
roster and unresolved evaluation settings in `config/experiment_1/`.

The read-only TFM Library snapshot is `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`.
Its `papers/text/2026/` contains the TabPFN-3, TabPFN-3.5, TabICLv2, LimiX-2, Causilo,
Mitra-v2 and EXAONE reports; `repositories/TabPFN .txt` contains the source snapshot.
For code evidence, refer to symbols such as `TextTransformer` and `InferenceConfig`, not
snapshot line numbers. Live upstream references below supplement that pin; they must be
rechecked when freezing package and checkpoint revisions.

## 1. Why text-derived features are a meaningful comparison

MulTaBench selects tasks where non-text and text/image inputs contribute jointly, and where
target-aware encoder adaptation improves on frozen embeddings for several curation learners.
Its text pipeline uses E5-small-v2; target-aware representations fine-tune with labels and
reduce representations with PCA. This differs fundamentally from a Jev feature that uses
only text and allowed task metadata. The latter can be reused across folds; supervised
encoder adaptation and fitted dimension reduction belong inside each training partition.
The curated collection is deliberately enriched for multimodal benefit, so conclusions should
be scoped to these tasks rather than all tabular prediction. [MulTaBench paper, §§3 and A](https://arxiv.org/html/2605.10616v1)

The paper-era protocol uses classification AUC and regression R², limited training sizes and
five runs. Current `master` is moving toward a different living benchmark, including changed
metrics. Neither is automatically the JEVPFN protocol. A local five-fold macro-F1/RMSE study
must not be presented as reproducing a published leaderboard score.
[Paper release](https://github.com/alanarazi7/MulTaBench/tree/paper_version),
[current repository protocol](https://github.com/alanarazi7/MulTaBench#protocol-and-tabarena)

The pinned paper result tables distinguish frozen joint representations (`all`) from
fine-tuned representations (`ft`). A read-only inspection of the core TEXT tables and
`more_baselines/` identifies strong TabICLv2, TabM, AutoGluon-MM and TabSTAR results.
AutoGluon is excluded from this project's candidate roster by owner decision; published
AutoGluon results remain literature context only.
Older TabPFN-v2/v2.5 results cover only 18 of the 20 text datasets in the frozen comparison;
coverage must accompany any rank. Rank within each dataset before averaging; never average
raw AUC and R² together or pool text and image leaderboards. Published CSVs contain run
metadata; retain only relevant aggregate evidence in research notes.
[Pinned result tables](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/leaderboard/results)

The TabPFN-3.5 report separately evaluates the 20 text datasets. Its local 3.5 and Fast models
use frozen E5-small-v2 representations, as do the non-native baselines; hosted Plus/Thinking
use their own raw-text pipeline. That text subset is more relevant here than a general
TabArena ranking. Authors' results are evidence, not an independent JEVPFN replication.
[TabPFN-3.5 report, §2.4 and Appendix C.4](https://arxiv.org/html/2609.17895v1)

E5-small-v2's model card specifies English input and a 512-token maximum. Frozen embeddings
are a reproducible semantic baseline, but their prefix, pooling, truncation and checkpoint
revision must be declared. Comparing Jev's longer inputs with truncated embeddings also
changes text coverage; it is not solely a comparison of representation quality.
[E5-small-v2 model card](https://huggingface.co/intfloat/e5-small-v2)

## 2. What “handles text” actually means

Three capabilities should be distinguished: accepting strings as categorical identifiers,
converting text into numerical representations before tabular inference, and using a learned
language encoder as part of the predictor. Accepting a string is not evidence that a model
uses its semantic meaning. Every tabular engine can in principle consume appropriately shaped
numeric Jev outputs; raw-text support is a separate comparison.

### TabPFN-3 and TabPFN-3.5

In package 9.0.0, `TextTransformer` is shared by the classification/regression estimator
pipeline, rather than restricted to the V3.5 checkpoint. `TRANSFORM_TEXT` is **false by
default**. Enabling it applies `skrub.StringEncoder`: character n-gram TF-IDF followed by
truncated SVD, at most 30 components by default. Eligibility depends on pandas string dtype,
cardinality and categorical declarations; ordinary object columns are not expanded. Missing
text becomes an empty string and hence an all-zero encoded row. Inspect which official text
columns were actually expanded; otherwise the experiment can silently compare categories.
[Versioned text implementation](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/src/tabpfn/preprocessing/text.py),
[inference configuration](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/src/tabpfn/inference_config.py)

Select `ModelVersion.V3` versus `ModelVersion.V3_5` explicitly. Local checkpoints both need
text encoding; the statement “3 cannot use text, 3.5 natively understands it” conflates model
and software/service versions. Hosted **3.5-Plus** and Thinking are separate systems. Both
3 and 3.5 have hosted Thinking selectors, which spend additional inference compute and are
not interchangeable with local defaults. Their access, costs and encoder details would need
separate experimental accounting. [Official model source](https://github.com/PriorLabs/TabPFN),
[Thinking documentation](https://docs.priorlabs.ai/capabilities/thinking-mode)

### Other numeric and categorical predictors

| Engine | Tasks and text treatment | Relevant interface limitation |
|---|---|---|
| TabICL v2 | Classification and regression; no built-in semantic text encoder. Strings/categories need the wrapper's encoding or explicit text embeddings. | Separate classifier/regressor checkpoints; `support_many_classes=True` enables the many-class wrapper. |
| LimiX-2 | Classification, regression and missing-value imputation. Numeric/categorical array interface; no documented raw-language encoder. | 400M model, distinct from the older 2M-parameter LimiX-2M; `LimiXPredictor` and V2 configs. |
| Causilo | Classification and regression. Strings/booleans are categorical, not language embeddings; missing feature values are accepted. | Native 10-class head, automatic error-correcting output-code decomposition for larger vocabularies. |
| Mitra-v2 | Classification and regression with synthetic pretraining; no model-level semantic text encoder. | Explicit v2 task checkpoints; many-class hierarchical handling and the paper's adaptation recipe require the corresponding wrapper. |
| EXAONE Tabular 1.0 | Classification, point and distributional regression. Raw text unsupported; numeric NumPy inputs only. | Encode categories using training data; built-in mean imputation and automatic many-class ECOC. |
| TabDPT 1.3 | Classification and regression; numeric tabular representation, requiring external text encoding. | Real-table pretraining, retrieval/context and FAISS environment need explicit accounting. |
| TabFM v1.0.0 | Classification and regression; mixed columns are ordinal-encoded/scaled, not interpreted as language. | Released classifier has a 10-class head; high-class tasks need a declared extension or explicit unsupported status. |
| CatBoost | Classification and regression; built-in text tokenizers, dictionaries and feature calculators, separate from categorical features. | BoW, NaiveBayes and BM25 are different encodings; validate regression-compatible calculators and empty vocabularies. |
| Logistic / linear regression | Classification / regression respectively; no raw-language input. | Training-only imputation, category encoding, scaling and optional TF-IDF/embeddings. |
| TabM | Classification and regression neural baseline; external text encoding. | Dataset-specific training and its budget differ from frozen foundation-model inference. |

Primary implementation references: [TabICL](https://github.com/soda-inria/tabicl),
[LimiX-2](https://github.com/limix-ldm-ai/LimiX), [Causilo](https://github.com/nums-ai/causilo),
[Mitra-v2 classifier](https://huggingface.co/autogluon/mitra-classifier-2),
[Mitra-v2 regressor](https://huggingface.co/autogluon/mitra-regressor-2),
[EXAONE](https://github.com/LGAI-Research/EXAONE-Tabular),
[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference),
[TabFM](https://github.com/google-research/tabfm),
[CatBoost text calculators](https://catboost.ai/docs/en/references/text-processing__feature_calcers),
[scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html),
[TabM](https://github.com/yandex-research/tabm).

These sources establish task families, not successful coverage of our exact 20 datasets.
Spotify's 114 classes are an explicit compatibility test. A many-class decomposition can
increase inference cost considerably. Declare context subsampling and failure coverage,
and keep missing values distinct from failed responses.

Mitra's published v2 recipe uses 50-step adaptation and eight-fold bagging. Inspection of
`mitra_finetune.api.MitraFinetune` and its runner confirms runtime imports of AutoGluon's
MITRA implementation even though package metadata lists AutoGluon as an optional extra.
No supported independent v2 runtime was verified. Mitra-v2 is therefore deferred under the
owner's no-AutoGluon constraint, without substituting v1 checkpoints or copying internals.
[Official model card](https://huggingface.co/autogluon/mitra-classifier-2),
[runtime source](https://huggingface.co/autogluon/mitra-finetune/blob/main/src/mitra_finetune/api.py)

LimiX-2 requires its source environment with Torch 2.9.1; Causilo requires Torch ≥2.13.
They cannot both be installed against our existing Torch 2.12.1 pin unchanged. EXAONE needs
NumPy ≥2.3.5 and a custom estimator lifecycle: it exposes fit/predict but not all cloning/grid
search methods. Model weights for these new engines and TabFM have separate noncommercial
terms; code licenses alone do not describe weight rights. These are dependency and provenance
facts, not reasons to silently omit a model. Use isolated frozen environments.
[LimiX documentation](https://www.limix.ai/doc/), [Causilo requirements/license](https://github.com/nums-ai/causilo),
[EXAONE interface](https://github.com/LGAI-Research/EXAONE-Tabular#requirements),
[TabFM weight notice](https://github.com/google-research/tabfm#license-notice-for-pretrained-weights)

### Predictors with language encoders

**TabSTAR** jointly uses text and other table features with a pretrained text representation
and task adaptation, exposing classifier and regressor interfaces. Its real-data pretraining
and task-specific training budget need to be distinguished from a label-free cached feature.
[Official implementation](https://github.com/alanarazi7/TabSTAR),
[NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/faf6e23e198314c7728eaa6ac44ae079-Paper-Conference.pdf)

**ConTextTab / SAP RPT-1 OSS** embeds column names and values before semantic in-context
learning. Both task families and missing values are supported. Its reference setup uses
Python 3.11; the recommended 8,192-row context and bagging can require 80GB GPUs. Weights
derive from real-table data with research restrictions.
[ConTextTab implementation](https://github.com/SAP-samples/sap-rpt-1-oss)

## 3. Jev as a probabilistic feature extractor

Jev exposes typed decisions rather than generated explanations. Noul returns one yes/no
probability; Choice returns the full option distribution; Score returns probabilities on an
ordered rubric and their expectation. Choice/Score also return confidence. Current limits
allow 255 Choice options and ten Score levels, sufficient for 114 classes and our nine-level
rubric. Returned probability meaning and calibration still require empirical inspection;
typed output prevents a format error, not an incorrect decision.
[API contract](https://docs.typesafe.ai/api), [primitives](https://docs.typesafe.ai/introduction)

Score descriptions need semantic distinctions. Numeric markers and degree adjectives alone
do not explain what separates adjacent levels. The provider recommends independently
meaningful descriptions of observable situations. Distinguish explicit comparisons, direct
relevant information, qualified evidence and indirect hints, without counting clues or using
label-derived quantiles/target means. This defines evidence strength, not a calibrated target
value. Jev explicitly warns against using interpolated Score outputs as
exact numeric magnitudes. Retaining the full distribution also distinguishes neutral evidence
from opposing evidence that happens to average to zero.
[Score guidance](https://docs.typesafe.ai/primitives/score),
[numeric calibration limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Generic directional evidence remains an experimental representation: its reference is not a
dataset mean, quantile or target unit. Explicit wording is not automatically stronger evidence
than accurate indirect information. A nine-level rubric therefore needs label-blind semantic
checks, including negation, unsupported assertions, conflicting cues and irrelevant text.
Whether nine distinctions work reliably is an empirical question, not guaranteed by Score's
ability to return nine probabilities. The implementation retains its unapproved draft until
wording is agreed; a numeric template version alone is not scientific validation.

The documented weaknesses include numeric/date reasoning, multi-step indirection and long
states with irrelevant details. Officially designated text fields can contain identifiers,
dates and measurements, so the text-column audit is essential. Joint input is not guaranteed
to outperform separate columns; more context can introduce distraction. Do not combine
separate-column questions against an all-text state and call that the same ablation: each
question would then see other columns.
[Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Identical request construction and identical API execution are different claims. We guarantee
the former and reuse the first validated cached response for every matching input. Reproducible
features therefore do not depend on an unverified promise of bit-identical repeated inference.
Pin model IDs, not moving aliases; retain the actual response model and usage. This makes
the experiment reproduce the saved representation even if provider execution changes later.

### Transport, volume and cost

The current reference is `jev-1.13.0`, USD 0.042 per million input tokens, output free.
Published limits are 40 requests/s and 100K tokens/s, adjustable without notice; context is
64K overall and 32K for state plus the longest question. At 2,032,351 unique requests,
the request-rate ceiling alone implies 14.1 hours at perfect utilization. This is a lower
bound, not a completion estimate; the account's actual allowance must be verified.
[Current model reference](https://docs.typesafe.ai/models)

Use the official asynchronous Python client with a reused HTTP connection and bounded
concurrency. Its SDK supports HTTP/2, configurable retry policies, request IDs, and usage
accounting. Rate-limit and overload errors require backoff and the returned retry delay.
Retrying a lost response can incur duplicate billing unless the provider confirms idempotency;
a local completed-response cache alone cannot close that interruption window.
[SDK usage](https://docs.typesafe.ai/sdk/python/usage),
[retry policy](https://docs.typesafe.ai/sdk/python/api/retries)

A future extractor needs one owner per request key, a resumable queue, durable response
commits, a global rate budget including retries, bounded retry counts, timeouts, and a hard
spending stop driven by measured usage. Keep debug bodies out of shared logs. No provider
batch endpoint or billing-idempotency guarantee is assumed. The current code has no live
transport; this description specifies operational requirements, not tested API behaviour.

The offline audit estimates approximately 853M input tokens after exact reuse at four
characters/token, or USD 35.84 before unknown overhead/retries. Three-to-five-character
tokenization and extra per-request tokens can move the bill materially at two million
requests. Calibrate against returned usage in an approved pilot. Over-limit text needs an
explicit policy before this is an executable workload. Regenerate measured figures with
`python -m src.jev.review`; do not treat this dated estimate as a quote.

The 32K state-plus-longest-question limit binds a one-question request before the 64K
overall limit does. Character-based estimates are suitable for budgeting, not proof that a
request fits. Truncation/chunking changes the feature definition and must be deterministic,
versioned, included in request identity, and accompanied by retained-text coverage. Research
on other language models shows position-sensitive long-context retrieval; it motivates
checking beginning/middle/end evidence, but does not establish a best truncation policy for
Jev. [Lost in the Middle](https://arxiv.org/abs/2307.03172)

The verified public contacts relevant to access and research support are
`hello@typesafe.ai` (general), `sales@typesafe.ai` (volume/custom arrangements), and
`support@typesafe.ai` (technical support). No public research-credit entitlement was found.
[Company site](https://typesafe.ai/), [volume contact](https://docs.typesafe.ai/models),
[support contact](https://typesafe.ai/legal/mca)

## 4. Evaluation implications

Feature reuse across CV folds is valid only while construction remains independent of row
targets and fitted label statistics. Class names are part of the declared task; their
frequencies are not. Manual rubric changes selected from evaluation scores would violate this
separation even without gradient training. Freeze the feature definition before evaluation.

Threshold optimization changes binary decisions without changing probability estimates.
Choose thresholds from held-out or inner out-of-fold predictions within each outer training
set, never outer test predictions or the estimator's in-sample predictions. Refitting after
threshold selection can change calibration and must follow an explicit rule. Multiclass
argmax and macro-F1 are distinct from binary threshold optimization; regression requires a
continuous-target metric. [Threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html)

Nested evaluation separates selection from assessment: choices optimized on noisy validation
scores can overfit that validation procedure, so the final outer test fold must remain untouched.
Scikit-learn's threshold wrapper supports both internal CV and a single validation fraction;
the former permits a final refit, whereas `refit=False` requires a single split. Its default
objective is balanced accuracy, not F1. Positive-class semantics and the F1 scorer must be
supplied explicitly. The library interface does not establish which validation budget is
best for these datasets. [Cawley and Talbot (2010)](https://www.jmlr.org/beta/papers/v11/cawley10a.html),
[TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)

Exact text repetition lowers extraction cost but does not prove independent rows. Repeated
entities, target proxies and time dependence require dataset-specific split review. Frozen
language embeddings may be computed once if truly independent of these labels, but PCA,
TF-IDF vocabulary fitting, category encoding and imputation belong within training folds.
Real-table-pretrained models additionally require overlap review; synthetic-only pretraining
does not eliminate every risk of overlap in auxiliary language models.

Group splits estimate generalization to unseen entities; forward splits estimate prediction
of later observations. A date column alone does not make a task forecasting, and identical
short text does not establish entity identity. Choose the prediction setting from the task's
provenance, then apply the same restrictions to outer and inner splits. Grouping can prevent
perfect stratification, so feasibility and class coverage need explicit checks.
[Cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html)

Probability quality, ranking and hard decisions answer different questions. Log loss and
Brier evaluate probabilities; AUC evaluates rankings; F1 depends on a decision rule. Average
precision and trapezoidal PR-AUC use different interpolation conventions. Binary positive-class
Brier is in [0,1], whereas a sum over all class coordinates is in [0,2]. State the convention
rather than comparing incompatible values. ECE depends on binning and is not a proper scoring
rule; retain bin counts and use it diagnostically. Calibration fitted with labels belongs
inside training partitions, not on the test fold.
[Scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html),
[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)

For regression, percentage/log errors require meaningful target units; positive numbers alone
do not justify MAPE or RMSLE on an already transformed target such as `log_price`. Distributional
scores require actual quantiles or predictive distributions, not fabricated uncertainty from a
point estimate. Report unavailable metrics explicitly. Separate fit, prediction, encoding,
model loading and selection costs: some in-context learners do most computation at prediction
time. CUDA timings require synchronization; one-time Jev extraction is not a per-fold fitting cost.

Compare improvements within the same dataset, split, model and compute budget. Aggregate
across datasets with equal dataset weights and report coverage; large datasets should not
dominate solely by row count. Do not treat five correlated folds as five independent datasets
for significance testing. Distinguish default inference, tuned inference, supervised
fine-tuning and ensembles. These distinctions explain why a leaderboard's “best model” is
not automatically the best controlled baseline for this study.

Demšar's multi-dataset analysis recommends paired Wilcoxon comparisons for two methods and
Friedman/post-hoc comparisons for several methods, using datasets as comparison units. Apply
multiple-comparison corrections to a prespecified family and retain effect sizes/coverage.
Raw RMSE differences across unrelated target units should not be pooled; neither should raw
F1 and RMSE. Small task families (here only three binary datasets) have weak inferential power.
[Demšar (2006)](https://www.jmlr.org/papers/v7/demsar06a.html)

TabArena highlights standardized evaluation and explicit treatment of tuning/ensembling budgets.
Its methodology is relevant even though AutoGluon execution is excluded here. A checkpoint,
an estimator preset, a fine-tuned model and a large ensemble are different comparison units.
Training rows exposed to the wrapper and effective in-context rows must both be recorded;
silently lowering only a failing model's context changes the comparison.
[TabArena](https://arxiv.org/abs/2506.16791)
