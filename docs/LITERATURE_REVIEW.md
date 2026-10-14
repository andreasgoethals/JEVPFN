# Literature Review

Tabular model evidence checked 30 September 2026; split code checked 1 October.
Decision-model releases, probability papers and Jev limits refreshed 12 October 2026;
wording evidence, Microsoft context listing and VSC resources checked 13 October.
Open-weight availability and additional decision interfaces refreshed 14 October 2026.
This is a durable reference for the research question,
published evidence, model interfaces and methodological limitations. It is not an experiment
schedule. Agreed project constraints live in [README](../README.md), data definitions in
[DATA_SOURCES](DATA_SOURCES.md), runtime/storage instructions in [VSC](VSC.md), and the candidate
roster and unresolved evaluation settings in `config/4. experiment_1/`.

The current read-only TFM Library checkout is `de324efcca55be392feb0c01830b3b6ce364587f`
(existing index/checkout verified 13 October; no submodule change here). The 12 October
review used `5b6c24dd9337c6e7bef45733730bf52ee280e61c`. Earlier tabular
interface review used `1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`; the current checkout
also contains MulTaBench v2, dated 27 September, inspected for the representation/curation claims.
Its `papers/text/2026/` contains the TabPFN-3, TabPFN-3.5, TabICLv2, LimiX-2, Causilo,
Mitra-v2 and EXAONE reports; `repositories/TabPFN .txt` contains the source snapshot.
For code evidence, refer to symbols such as `TextTransformer` and `InferenceConfig`, not
snapshot line numbers. Live upstream references below supplement that pin; they must be
rechecked when freezing package and checkpoint revisions.

## 1. Why text-derived features are a meaningful comparison

### Closest existing work and the novelty boundary

The public **tab-jev** project already converts Jev option probabilities into numeric features
for tabular predictors, including a TabPFN-3.5 API backend. Its Kickstarter demonstration
compares table-only prediction, text-only Jev, lexical text encodings and several combined
pipelines at fixed label budgets; it also provides optional manually specified rubrics,
few-shot examples and supervised calibration. This is direct prior art, not merely a generic
decision-model release. Its demonstration is a binary task, not a verified three-task,
multi-dataset replication. JEVPFN must not claim to introduce the Jev-plus-TabPFN combination.
[Project and results](https://github.com/edamame-labs/tab-jev),
[evaluation procedure](https://github.com/edamame-labs/tab-jev/blob/main/demo/README.md)

TypeSafe's feature-discovery cookbook also converts Jev answers into features for a CatBoost
wine-score regressor. An LLM proposes questions; subsequent proposals use training labels
and model errors. That procedure differs from JEVPFN's fixed, mechanical, label-free
construction, but establishes that Jev-based tabular feature engineering is already public.
The cookbook's results are provider-produced evidence on its specified split, not an
independent reproduction. [Official cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)

A defensible research contribution therefore concerns independently measured utility and
limits of task-conditioned probabilistic text representations: coverage of three task types,
multiple text columns, deterministic templates, conventional text-representation controls,
and reuse/cost under a frozen protocol. No claim of firstness or publishability follows from
this literature search alone. Evidence about a decision model's isolated accuracy is distinct
from evidence that its features improve a tabular learner.

MulTaBench selects tasks where non-text and text/image inputs contribute jointly, and where
target-aware encoder adaptation improves on frozen embeddings for several curation learners.
Its text pipeline uses E5-small-v2; target-aware representations fine-tune with labels and
reduce representations with PCA. This differs fundamentally from a Jev feature that uses
only text and allowed task metadata. The latter can be reused across folds; supervised
encoder adaptation and fitted dimension reduction belong inside each training partition.
The curated collection is deliberately enriched for multimodal benefit, so conclusions should
be scoped to these tasks rather than all tabular prediction. [MulTaBench paper v2, §§3 and A](https://arxiv.org/html/2605.10616v2)

The paper-era protocol uses classification AUC and regression R², five random seeds and a
10,000-example training cap. The pinned `split_to_test` / `do_split` code uses repeated random
train/test holdouts (`SEED + fold`), not five disjoint test partitions. Its default test size
is min(10% of rows, 2,000); a supplied training cap overrides this by reserving the remaining
rows for testing. Classification uses stratification, with special binary/rare-class handling.
`split_to_val` requests min(10% of training rows, 1,000) for validation; the classification
helpers additionally enforce their class-coverage rules. These constants alone
must not be described as an unconditional 80/10/10 split or standard five-fold KFold.
[Pinned split implementation](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/preprocessing/splits.py),
[paper setup, Appendix A.2](https://arxiv.org/html/2605.10616v1#A2)

JEVPFN instead uses the owner's five-fold stratified classification / shuffled regression CV,
all outer-training rows, fixed model settings and no HPO. This retains a controlled comparison
between Jev feature combinations without reproducing the paper's limited-data scores. The
paper's CatBoost setup is not literally the library default: it specifies 2,000 iterations
and early-stopping parameters. JEVPFN uses pinned library defaults instead. Current `master`
also has a changing living-benchmark protocol; neither should silently replace this setup.
[Paper release](https://github.com/alanarazi7/MulTaBench/tree/paper_version),
[current repository protocol](https://github.com/alanarazi7/MulTaBench#protocol-and-tabarena)

The pinned paper result tables distinguish frozen joint representations (`all`) from
fine-tuned representations (`ft`). A read-only inspection of the core TEXT tables and
`more_baselines/` identifies strong TabICLv2, TabM, AutoGluon-MM and TabSTAR results.
AutoGluon and TabM are excluded from this project's roster by owner decision; these published
results remain literature context only. A high leaderboard rank is not a reason to expand
the focused Jev study.
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

An earlier multimodal-embedding study evaluates TabPFN as a head on frozen image, text and
audio encoders, with gains varying by label budget and representation dimension. Thus both
semantic embeddings plus TabPFN and downstream calibration assessment precede this project;
compactness is a hypothesis to measure, not a novel property of attaching features to TabPFN.
[TabPFN beyond Tabular Data](https://arxiv.org/abs/2607.11007)

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

The selected main predictors are CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular
and logistic/linear regression. Each compares four feature configurations on shared folds:
non-text only, plus joint, plus per-column, plus both decision representations. TabPFN-3 and
the additional interfaces below remain research references, outside this selected roster.
Google TabFM, TabM and AutoGluon are excluded.

| Engine | Tasks and text treatment | Relevant interface limitation |
|---|---|---|
| TabICL v2 | Classification and regression; no built-in semantic text encoder. Strings/categories need the wrapper's encoding or explicit text embeddings. | Separate classifier/regressor checkpoints; `support_many_classes=True` enables the many-class wrapper. |
| LimiX-2 | Classification, regression and missing-value imputation. Numeric/categorical array interface; no documented raw-language encoder. | 400M model, distinct from the older 2M-parameter LimiX-2M; `LimiXPredictor` and V2 configs. |
| NVIDIA Kumo Tabular | Classification/regression over numeric and categorical tables through SDM; no native language encoder. | Task-specific checkpoints, many-class decomposition; regression point prediction is the mean of 999 predicted quantiles. Default preprocessing selects at most 500 columns. |
| Causilo | Classification and regression. Strings/booleans are categorical, not language embeddings; missing feature values are accepted. | Native 10-class head, automatic error-correcting output-code decomposition for larger vocabularies. |
| Mitra-v2 | Classification and regression with synthetic pretraining; no model-level semantic text encoder. | Explicit v2 task checkpoints; many-class hierarchical handling and the paper's adaptation recipe require the corresponding wrapper. |
| EXAONE Tabular 1.0 | Classification, point and distributional regression. Raw text unsupported; numeric NumPy inputs only. | Encode categories using training data; built-in mean imputation and automatic many-class ECOC. |
| TabDPT 1.3 | Classification and regression; numeric tabular representation, requiring external text encoding. | Real-table pretraining, retrieval/context and FAISS environment need explicit accounting. |
| CatBoost | Classification and regression; built-in text tokenizers, dictionaries and feature calculators, separate from categorical features. | BoW, NaiveBayes and BM25 are different encodings; validate regression-compatible calculators and empty vocabularies. |
| Logistic / linear regression | Classification / regression respectively; no raw-language input. | Training-only imputation, category encoding, scaling and optional TF-IDF/embeddings. |

Primary implementation references: [TabICL](https://github.com/soda-inria/tabicl),
[LimiX-2](https://github.com/limix-ldm-ai/LimiX), [Causilo](https://github.com/nums-ai/causilo),
[Mitra-v2 classifier](https://huggingface.co/autogluon/mitra-classifier-2),
[Mitra-v2 regressor](https://huggingface.co/autogluon/mitra-regressor-2),
[EXAONE](https://github.com/LGAI-Research/EXAONE-Tabular),
[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference),
[CatBoost text calculators](https://catboost.ai/docs/en/references/text-processing__feature_calcers),
[scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html).

These sources establish task families, not successful coverage of our exact 20 datasets.
Spotify's 114 classes motivated its exclusion from extraction and all experiments on
12 October; raw data remains available in the 20-dataset exploration. A many-class decomposition can
increase inference cost considerably. Declare context subsampling and failure coverage,
and keep missing values distinct from failed responses.

Mitra's published v2 recipe uses 50-step adaptation and eight-fold bagging. Inspection of
`mitra_finetune.api.MitraFinetune` and its runner confirms runtime imports of AutoGluon's
MITRA implementation even though package metadata lists AutoGluon as an optional extra.
No supported independent v2 runtime was verified. Mitra-v2 is therefore deferred under the
owner's no-AutoGluon constraint, without substituting v1 checkpoints or copying internals.
[Official model card](https://huggingface.co/autogluon/mitra-classifier-2),
[runtime source](https://huggingface.co/autogluon/mitra-finetune/blob/main/src/mitra_finetune/api.py)

Current LimiX-2 source requires Python ≥3.12 and Torch ≥2.9.1; the older exact-2.9.1 statement
was stale. Causilo requires Torch ≥2.13 and cannot use our existing 2.12.1 pin unchanged. EXAONE needs
NumPy ≥2.3.5 and a custom estimator lifecycle: it exposes fit/predict but not all cloning/grid
search methods. Model weights for these deferred engines have separate noncommercial
terms; code licenses alone do not describe weight rights. These are dependency and provenance
facts, not reasons to silently omit a model. Use isolated frozen environments.
[LimiX documentation](https://www.limix.ai/doc/), [Causilo requirements/license](https://github.com/nums-ai/causilo),
[EXAONE interface](https://github.com/LGAI-Research/EXAONE-Tabular#requirements)

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

The selected original-text baselines are TabSTAR and ConTextTab only; they receive all original
features without decision-model outputs. This is not a claim that they are the three highest
scoring MulTaBench methods. AutoGluon-MM remains excluded by owner choice.

Source audit on 13 October: TabSTAR `BaseTabSTAR.fit` performs supervised adaptation and
creates internal validation from the supplied training fold when none is passed; its verbalizer
is fitted on that internal training subset. This required adaptation is part of TabSTAR, not
an added HPO campaign. ConTextTab `RPT` wrappers default to 8,192-row contexts and bagging;
our adapter explicitly uses one bag and the complete outer/inner training partition. That
removes silent row sampling but does not prove that large contexts fit memory. `SentenceEmbedder.embed`
truncates each embedded string to 512 tokens, which must be reported as baseline preprocessing.
Column names and values are embedded, then labeled training rows provide prediction context.
[Pinned TabSTAR source](https://github.com/alanarazi7/TabSTAR/tree/32b232243e06018f3d3f86f006d9337de68ac15b),
[pinned ConTextTab source](https://github.com/SAP-samples/sap-rpt-1-oss/tree/a0067b3c9db56bc35a6dcab09a9223a3ff0821e0)

NVIDIA Kumo Tabular is the single-table model in `structured-data-models`, not the relational
KumoRFM API. The adapter follows the publisher's SDM prediction path, keeps every training
row, batches queries, reorders class probabilities and averages predicted quantiles for point
regression. Its default 500-column selection is rejected on overflow. LimiX uses the publisher's
V2 no-retrieval recipes. Their `quantile_norm_all_data` name should not be treated as evidence
of test-set fitting: inspection of `RebalanceFeatureDistribution.fit_transform` shows the
train/test split and fitting on training rows. These interfaces are source-checked; real
checkpoint loading, full-fold resources and pretraining overlap remain pilot requirements.
[Pinned SDM implementation](https://github.com/NVIDIA/structured-data-models/tree/bd2d55db104679393fcd6afe13305f37e40b5467),
[Kumo weights](https://huggingface.co/nvidia/Kumo-Tabular),
[pinned LimiX implementation](https://github.com/limix-ldm-ai/LimiX/tree/516bf396333feb3198cf7aff8a6c10421f218e24)

A related 8 October preprint, [Estimating Uncoded Crash Factors with Tabular Foundation and
System One Models](https://arxiv.org/abs/2610.10321), uses Kumo on coded crash records and Jev
on narratives, with human recalibration and population-estimation methods. This was checked
at abstract level. It is relevant prior art for combining tabular and decision systems, not
evidence that this project's six-extractor/four-representation MulTaBench comparison has
already been evaluated. Novelty must rest on the controlled representation comparison,
coverage and reproducible extraction evidence rather than merely mentioning Jev plus a TFM.

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
ability to return nine probabilities. The owner accepted the evidence rubric on 1 October;
`directional-v2` now implements it centrally with symmetric lower/higher descriptions and
zero for no favored direction. Acceptance of wording is not validation of model behaviour.

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

The main retained view is already selected: a binary Noul probability, the complete multiclass
Choice probability vector, and the regression expectation over −4..+4. Raw responses/full
probabilities and available confidence are retained independently for inspection or later
ablations. A headline metric can be deferred while all configured metrics are recorded;
choose it before reviewing comparative outcomes, or explicitly label later selection exploratory.

### Transport, volume and cost

The current reference is `jev-1.13.0`, USD 0.042 per million input tokens, output free.
Published limits checked 12 October are 80 requests/s and 100K tokens/s, adjustable without
notice; context is 64K overall and 32K for state plus the longest question. Current extraction
counts and cost scenarios belong in generated reports, rather than a fixed historical total
here. Published ceilings are not completion estimates or verified account allowances.
[Current model reference](https://docs.typesafe.ai/models)

The official SDK calls `client.system_one(state, questions)`. Construct the client with
`model="jev-1.13.0"` and an environment-supplied API key; use `Noul(instructions=...)`,
`Choice(instructions=..., criteria={...})`, or `Score(instructions=..., criteria=[...])`.
The numeric pin is selected at the client/request layer, not by loading local model weights.
Version changes invalidate request keys and require a new feature artifact. Existing paid
features remain reusable from their original pinned cache; SDK transport remains disabled here.
[Python SDK usage](https://docs.typesafe.ai/sdk/python/usage)

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
batch endpoint or billing-idempotency guarantee is assumed. The guarded
`src/jev/extraction.py` wrapper disables hidden SDK retries and prepares persistent dataset
reservations, measured usage, bounded concurrency and spending allocations. Unknown
response/billing outcomes stop rather than automatically retry. This was exercised with fake
transports only; account throughput, tokenizer/context validation and actual billing behavior
remain live-pilot questions. Initial configurations enable no network extraction.

Offline budget totals depend on template wording as well as text. The accepted expanded
regression rubric supersedes the earlier draft's token estimate; current numbers belong in
generated reports. Three-to-five-character tokenization and extra per-request tokens can move
the bill materially at two million requests. Calibrate against returned usage in an approved
pilot. Over-limit text needs an explicit policy before this is an executable workload.
Regenerate figures with `python -m src.jev.review`; they remain scenarios, not price quotes.

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

Exact text repetition lowers extraction cost but does not prove independent rows. The owner
chose shuffled row CV, so the benchmark does not estimate unseen-entity or future-time
generalization. Target-proxy/input-availability review remains relevant. Frozen
language embeddings may be computed once if truly independent of these labels, but PCA,
TF-IDF vocabulary fitting, category encoding and imputation belong within training folds.
Real-table-pretrained models additionally require overlap review; synthetic-only pretraining
does not eliminate every risk of overlap in auxiliary language models.

Group splits estimate generalization to unseen entities; forward splits estimate prediction
of later observations. They address different questions and are outside the agreed initial
scope. Stratification applies to both binary and multiclass targets; check rare-class fold
feasibility without silently merging or dropping classes.
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

If formal multi-dataset testing is later justified, Demšar discusses paired Wilcoxon comparisons
for two methods and Friedman/post-hoc comparisons for several methods, using datasets as
comparison units. This is background, not a required model-ranking campaign for JEVPFN.
The initial comparison reports per-dataset Jev deltas and fold mean/SD with coverage.
Raw RMSE differences across unrelated target units should not be pooled; neither should raw
F1 and RMSE. Small task families (here only three binary datasets) have weak inferential power.
[Demšar (2006)](https://www.jmlr.org/papers/v7/demsar06a.html)

TabArena highlights standardized evaluation and explicit treatment of tuning/ensembling budgets.
Its methodology is relevant even though AutoGluon execution is excluded here. A checkpoint,
an estimator preset, a fine-tuned model and a large ensemble are different comparison units.
Training rows exposed to the wrapper and effective in-context rows must both be recorded;
silently lowering only a failing model's context changes the comparison.
[TabArena](https://arxiv.org/abs/2506.16791)


## 5. Decision-model releases and probability evidence — checked 14 October 2026

These are text-to-probability extractors, distinct from downstream tabular predictors. No new
extractor is enabled by this review. Comparable extraction must use the same label-free states,
class vocabulary, regression rubric and dataset selection. Vendor decision leaderboards do not
establish improvement on MulTaBench or equivalence of calibration across providers.

| Extractor | Verified primary evidence | Relevance and limitation |
|---|---|---|
| Jev 1.13 | [TypeSafe model reference](https://docs.typesafe.ai/models), checked 12 October: $0.042/M input tokens, output free; published 80 RPS/100k TPS, dynamic. | Main extractor stays pinned; confirm organization-specific limits with TypeSafe. The owner's direct correspondence confirms the price but offers no research exemption. |
| Microsoft-Decision-1 | [9 October release](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/): post-trained Qwen3.5-9B, single-pass option probabilities; $0.042/M input, output free. [Foundry API](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) documents Noul/Choice/Score, model version 1 and public-preview deployment. | Suitable sixth candidate for the same text-to-probability protocol. Reads text/JSON but does not generate prose. Deployment name differs from response model identity. Open Qwen weights do not establish availability of Microsoft's post-trained weights. Publisher quality/latency results do not establish MulTaBench gains; verify refusals, context behavior, account access and version immutability. Future MAI/OpenAI backbones are announced plans. |
| Liquid d1-3B | [7 October open-weight release](https://www.liquid.ai/blog/d1-open); [publisher model card](https://huggingface.co/LiquidAI/d1-3B). Approximately 3B parameters, not 13B. Noul/Choice/Score, full probabilities, 32,768 context. | Close schema match and small self-hosted comparator. The [LFM Open License](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE) explicitly exempts qualifying nonprofits' noncommercial/research use from its commercial revenue threshold; attribution/redistribution conditions remain. Use its custom decision interface, not generated chat examples. No model downloaded. |
| Cloudflare Clef-flash | [Hosted reference](https://developers.cloudflare.com/workers-ai/models/clef-flash/): $0.038/M input, 24,576 tokens. The [public Apache-2.0 checkpoint](https://huggingface.co/Cloudflare/clef-flash) is now available, including the Qwen3.5-9B backbone, joint schema head and complete encoding/inference source. Hub revision observed 14 October: `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` (public, ungated). | Self-hosted route is now feasible in principle, not implemented here. `joint_schema_model.encode_record` renders state/questions; the joint head supplies logits for every option, normalized per question. The official runtime was tested on H200; an A100 pilot is still required. Hosted service truncates long states. Audit the released renderer before claiming exact Clef counts; do not infer hosted parity from downloadable weights. |
| AWS Strands Decider | [1 October launch](https://strandsagents.com/blog/introducing-strands-decider/); [official repository](https://github.com/strands-labs/strands-decider). Apache-2.0, Noul/Choice/Score, full probability vectors; pointer readout over supplied options. | Small self-hosted comparator. The newer [Qwen3.5 2B checkpoint](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) was created 9 October and saves a 4,096-token window. Original launch and updated checkpoint are different revisions. Verify complete input and the remaining class vocabularies; published latency is hardware-specific. |
| Perplexity Decider v1.1-27B | Original [v1 model](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) repository created 1 October; [v1.1 announcement](https://community.perplexity.ai/t/pplx-decider-v1-1-27b-our-updated-open-weights-multimodal-decision-model-is-now-available/6286), 6 October. Apache-2.0, native decision head, noncausal full attention, saved calibration temperature; Noul/Choice/Score supported by supplied inference code. | [Current card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) reports stronger Decision Index results, not MulTaBench gains. Local weights require about 49 GiB plus working memory; preparation rejects over 8,192 tokens. Public Hub metadata is currently private=false/gated=false, despite stale private-access wording in the card; no weight download was attempted. Check tasksource training overlap. Publisher announced hosted pricing of $0.02/M for v1.1; hosted access and version pin still require verification. |
| Fastino GLiNER2.5-Decide | [Publisher release](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model): 340M encoder, Apache-2.0. However, current [deployment capabilities](https://docs.fastino.ai/concepts/gliner-2-5-decide-capabilities) describe unordered classification labels without label-description objects and direct ordered scoring to GLiDE. | Do not equate release claims with the current hosted schema. Complete probability mapping and long-input behavior of the selected local runtime need verification. `max_position_embeddings=512` in its DeBERTa config alone does not prove a 512-token hard cutoff for this relative-position architecture. Not admitted to the uniform extraction protocol yet. |
| Fastino GLiDE | [Publisher release](https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model), dated 30 September; adaptive reasoning and full probabilities. [Capabilities](https://docs.fastino.ai/concepts/glide-capabilities): 40,000 tokens per rendered state/question, 2–255 choice options, up to 255 score levels; overlong input is rejected. [Pricing](https://docs.fastino.ai/pricing): $0.15/M input, output free. | Suitable optional reasoning comparator, hosted only in the checked evidence. Reported input usage can sum multiple internal passes, so externally counted text tokens cannot establish the bill. Freeze the version and measure account throughput, latency and billed usage in a pilot. Publisher benchmark claims are not downstream tabular evidence. |
| OpenAI Decisions / GPT-6 Luna | [Official guide](https://developers.openai.com/api/docs/guides/decisions): public beta, `POST /v1/decisions`; predicate, choice and ordered score with full distributions. $0.10/M input, no output/cache fees on this endpoint; regional/long-context adjustments can apply. | Relevant hosted candidate, not yet in the configured six-provider roster. Map nine ordered indices 0–8 back to −4…+4 (expected score minus four); preserve all probabilities and treat refusals as failures requiring review, not empty text. The model's general context specification does not replace testing this endpoint's actual limits/usage. Snapshot pin and account throughput remain to verify. |
| Intern-Decision-4B | [Publisher card/source](https://huggingface.co/internlm/Intern-Decision-4B): Apache-2.0, Qwen3.5-4B, one forward pass over candidate-symbol logits followed by checkpoint calibration; Noul/Choice/Score. Default complete-input limit 8,192, 62 options, strict rejection. | Strong additional open-weight candidate. All retained tasks have at most 30 classes or nine ordered levels; option capacity fits. Its specific tokenizer/renderer and live GPU runtime still need an audit and adapter. Preserve published calibration; do not fit it with benchmark targets. Author-run accuracy/latency do not establish our task performance. |
| Torchcast Decision 27B | [Publisher card/source](https://huggingface.co/torchcast-ai/torchcast-decision-27b): all three typed outputs, full probabilities, published context up to 262,144; weights CC BY-NC 4.0 and code Apache-2.0. Choices above 26 options take additional rounds. | Technically plausible research-only reserve candidate; no adapter or exact token audit here. Adds another large GPU runtime; the 30-class Wine Review task must exercise its multi-round path. Published Decision Index results are author-run and guided model development; inspect training overlap and actual GPU throughput rather than using the headline to select it. |

Microsoft context evidence was refreshed 13 October: [OpenRouter's model listing](https://openrouter.ai/microsoft/microsoft-decision-1)
specifies 32,768 tokens. The earlier unknown ceiling meant unverified documentation, not
unlimited context. Deployed Foundry input accounting still needs verification. The same
listing says weights are continually updated, so an unchanged model alias or API version
does not establish immutable weights. Capture provider/version/date and complete responses;
seek an immutable deployment before a campaign and disclose unresolved model drift.

The six published interfaces remain compatible with one binary probability, a complete
multiclass vector and an expected nine-level direction. This is schema compatibility;
their accuracy, numerical behavior, account access and throughput on these datasets remain
unverified. Score is an ordinal readout, not a continuous-target regression model.

The pinned [Perplexity inference source](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py)
requires one singular `question` per row: `DecisionModel.predict([row], batch_size=1)[0]`,
then `answer(question, probabilities)`. Preserve its saved noncausal attention and calibrated
temperature; `prepare` rejects over 8,192 tokens. A map of named `questions` is the other
providers' interface, not this Python method's argument. The offline preview now shows
these exact arguments without importing the model or downloading weights.

[Strands serving documentation](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md)
specifies `--strict-window` to reject overlong inputs; default serving truncates the state.
It also reports shared mutable option offsets and unverified concurrent behavior, so future
calls must be serialized per engine until the pinned runtime passes concurrency checks.
Keep its unauthenticated server on loopback. The checkpoint contains an adapter/readout;
stage the separate Qwen base too. Its pinned `provenance.json` names base revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, with an explicit caveat that the publisher inferred
this training-time revision because the training hosts had not pinned it. Pin both artifacts
and the runtime, retaining that provenance limitation. [Pinned checkpoint provenance](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/provenance.json).

Cloudflare also [documents automatic state truncation](https://developers.cloudflare.com/workers-ai/models/clef-flash/);
local token validation must prevent silent differences in input coverage. Microsoft's returned
model name can differ from the requested deployment alias. Liquid returns an answers/usage
object without requiring Jev's model field. Provider-specific response normalization and
provenance validation are therefore required; the Jev response validator cannot simply be
reused unchanged for all extractors.

Token accounting checked 14 October against the same pinned sources: Liquid's single-question
route encodes the complete `prompt.render` result without adding special tokens again.
Strands `_fit(strict_window=True)` separately encodes the state with special tokens and the
question without them; tokenizing their concatenation can give a different count. Perplexity
`prepare` renders `decision_messages` with its saved chat template and disables thinking.
The project now executes these weight-free formats with the actual staged tokenizers.
The checked hosted API references expose response usage, but do not establish the exact
hosted bill from these open-model counts. Clef's newly verified released renderer enables a
future fourth offline counter; it has not yet been integrated or counted. Jev and Microsoft
hosted counts remain unverified. The saved 18-dataset audit contains 1,698,127 unique inputs
per provider: Liquid 821,257,572, Strands 1,141,530,844 and Perplexity 960,172,532 tokens,
all within their configured windows. These totals are measurements for those three formats,
not confidence bounds on other providers' token usage. Hosted cost estimates must state their
token assumption and then be replaced by pilot billing measurements.

### Research evidence and its limits

The Microsoft announcement's current text reports 2.5 times the speed of H2O-Lightning-4B
and 35 times GPT-6 Sol in its own experiment; it does not currently support the newsletter's
specific 2.8-times-Jev claim. Those are publisher measurements with workload/hardware caveats,
not an independently reproduced comparison. Using a pre-trained backbone is standard transfer
learning and does not establish the vendor's confidence, commitment or motives. Funding and
geopolitical commentary are not scientific evidence for selecting this project's extractors.

`src/jev/alternatives.py` provides offline previews of the shared question semantics and
documented provider transports. API endpoints/deployments/auth differ even when semantics
match. Score maps probabilities over indices 0..8 to expected direction by subtracting 4;
it is not native prediction in target units. Self-hosted weights avoid API token fees but
still incur compute/storage cost. A controlled comparison needs matched input text and
declared coverage; differing context limits must not silently change the inputs. Compare
within fixed downstream predictors, retaining cheap text statistics and conventional text
encodings as controls. A six-provider catalog is preparation, not proof of novelty or readiness.

The following are arXiv preprints as checked on 12 October, not verified peer-reviewed
acceptances. Main methods, results and limitations were inspected for the ten core papers
below; the specialized SecJev paper was checked at abstract/release level. Positive vendor
scores and independent research evidence must remain distinguishable. None of these decision
benchmarks by itself establishes downstream gains on the selected MulTaBench datasets.

| Paper and initial date | What was studied and what the evidence supports | Implication for interpreting JEVPFN |
|---|---|---|
| [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550), 22 September; v5 updated 9 October | Compares Jev with generative/reward judges and evaluates confidence-based escalation. Stronger on evidence-grounded judgments than on judgments requiring derivation; confidence routing has workload-specific failures. | Cheap readout can be useful, but confidence is not an automatic correctness guarantee. Judging and tabular augmentation are distinct tasks. |
| [JevOut](https://arxiv.org/html/2609.30243v2), 24 September; v2 updated 8 October | Tests natural-context redirection across four systems/seven datasets. Probability-guided additions can change initially correct choices without changing the appropriate answer. | Joint context can distract. These adaptive, answer-preserving attacks are not an estimate of error on ordinary untouched table cells. |
| [Early Evidence Audit](https://arxiv.org/html/2609.32160v1), 26 September | Reviews 28 early papers and derives an evaluation checklist. No independent accuracy advantage of typed readout over comparable label-probability readouts was established by that early corpus. | Distinguish representation utility from architectural superiority; include suitable low-cost text baselines and measured usage/timing. The survey covers an early snapshot. |
| [Beyond Calibration](https://arxiv.org/html/2609.33209v1), 27 September | Tests logically related Noul/Choice questions on 160 items. Jev is more coherent than the tested Qwen readouts but still violates negation/complement identities; separate single-label probabilities sum to 1.14 on average. | A normalized Choice vector is not interchangeable with separately queried and renormalized Noul answers. Fixed wording and complete probability retention matter. |
| [Evaluating and Benchmarking Jev](https://arxiv.org/html/2609.37647v1), 29 September | 37 datasets, 346,009 requests; same-prompt non-thinking open-model comparisons. Strong zero-shot results on several short text tasks; weaker low-resource, fine-label, legal and rubric tasks. One request per item; closed-model pretraining remains unknown. | Direct positive evidence, with prompt/serving/split limitations. Application-specific probability calibration and threshold handling need evaluation. This suite is distinct from Decision Index and JEVal. |
| [Beyond Answer Confidence](https://arxiv.org/html/2610.01006v1), 1 October | Familiar-task and missing-information contrasts. Confidence can remain high when the answer is undecidable; follow-up familiarity questions largely track surface cues under controls. | Probability features are predictive inputs, not certified measures of sufficient evidence or model knowledge. Neutral/irrelevant-input checks need not use row labels. |
| [General Decision Models / JEVal](https://arxiv.org/html/2610.03935v1), 2 October | 11,257 bilingual instances, 36 datasets, ten domains, 25 configurations; also agent/social simulations and InnerJev distillation. Explicit-evidence tasks fare better than exact computation, specialist knowledge and faithful uncertainty estimation. | Broad evidence for the model class, including limitations. Local decision quality does not automatically transfer to reliable system-level outcomes or downstream tabular benefit. |
| [SearchJev](https://arxiv.org/html/2610.05107v1), 4 October | Six search-decision types; soft-label learning combines cross-entropy/Brier losses and held-out temperature calibration. Faster decisions and better reported search-agent outcomes than its specified baselines. | An example of specialized probability-focused training, not validation of general-purpose text-to-tabular regression. |
| [Political Science Replications](https://arxiv.org/html/2610.06625v2), 5 October; v2 updated 6 October | Seven published annotation/scaling tasks; human labels and commercial/open LLM comparisons. Jev often approaches comparator capability, but has no cost advantage over the study's commercial batch-price baseline and does not consistently beat the open model's calibration. | Independent evidence is mixed. Report actual task-specific cost and calibration; do not extrapolate universal superiority from low listed input pricing. |
| [Readout Stability](https://arxiv.org/html/2610.07716v1), 6 October | Seven model families and ten datasets; studies candidate-menu interventions and inference-time compute. Ranking stability is stronger than probability stability; extra order-permuted passes mainly improve calibration in the tested setting. | Candidate vocabulary/order and repeated execution are different from cache reuse. Do not transfer the paper's menu-selection results to this project's fixed full-class vocabulary without testing. |

[SecJev, 2 October](https://arxiv.org/abs/2610.03073) is a security-specialized decision-model
family, with reported gains from domain-specific training. Its scope does not justify replacing
a general extractor on heterogeneous table datasets. Likewise, the [network-traffic study,
30 September](https://arxiv.org/abs/2610.00376), finds trained tree ensembles stronger than its
Jev configurations on numerical packet data; supervision budgets differ, so this does not
isolate architecture as the cause or test our text-augmentation method.

All four requested alternative families were released after Jev's 15 September launch;
first family releases and later checkpoint updates are different dates. Their shared typed
schemas establish interface feasibility, not successful coverage of our longest inputs,
remaining class vocabularies, or continuous targets. Spotify's 114-class task remains exploration-only. Score predicts an ordered rubric, not a target measured
in salary, price or other dataset units. Provider/checkpoint identity must enter request keys
before sharing a cache across extractors. Jev and the three selected self-hosted adapters
have offline contract tests; Clef/Microsoft response adapters are pending. No extractor has
been enabled or executed on these datasets.

The reviewed evidence supports evaluating a separately approved comparator, not automatically
replacing Jev or expanding into a broad model leaderboard. This is a dated primary-source
review, not an exhaustive claim that no other papers exist. The local TFM resource was read
without modification; the current pin is recorded at the start of this document.

## 6. Question wording, task definitions and context comparability

Linguistic clarity is a design criterion, not evidence of optimal model performance.
Grice's account of cooperative communication motivates relevant information, clear referents,
unambiguous expressions and avoiding unnecessary verbosity; it concerns human discourse, not
a validated recipe for decision models. [Logic and Conversation, original scan](https://danielwharris.com/teaching/spring16/readings/Grice_1975_LogicConv.pdf).
For task questions, explicitly identify the case, target, evidence and answer alternatives;
place definitions in a fixed context object rather than repeatedly embedding them in a long sentence.

Kennedy's analysis of gradable language shows why words such as higher/lower need a comparison
standard. Saying 'typical cases of the documented kind' identifies a comparison class but still
leaves the numerical reference to the model's background knowledge. Without target statistics,
directional regression features cannot claim a common dataset-relative reference or calibrated
target units. The fixed nine levels remain an ordinal evidence rubric, with distinct meanings;
pilot checks and distribution ablations should address this limitation. [Vagueness and Grammar,
author manuscript](https://semantics.uchicago.edu/kennedy/classes/s06/readings/kennedy05.pdf).

[Weller et al., Learning from Task Descriptions (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.105/)
formalizes task definitions supplied separately from example inputs. [Gen-Z (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/af7cc9e2366b8f8837a6ef339810277a-Abstract-Conference.html)
studies contextualized class descriptions for generative zero-shot classification. Both support
examining documentary definitions, not importing their training procedure or assuming gains for Jev.
Safe task context consists of sourced observation units, target definitions/units, text-column
meanings and official class-code meanings. It excludes row targets, label frequencies, target
averages/quantiles, feature-target relationships and model feedback. Free-form documentation still
requires human provenance/leakage review; field validation cannot certify its contents.

[Webson and Pavlick (NAACL 2022)](https://aclanthology.org/2022.naacl-main.167/)
find that instructive-looking prompts need not be the cause of good NLI performance.
[Sclar et al. (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html)
find substantial sensitivity to meaning-preserving prompt formatting and weak cross-model
correlation of formats. Those experiments study generative/prompt-based models, not all six
extractors here. Their relevance is methodological: freeze common semantics and inspect several
predeclared paraphrases/orderings on a small shared pilot, without selecting prompts from final
CV performance. Distinguish probability changes, class flips and extraction failures.

Context windows differ and tokenizers count differently. Compare complete identical inputs on
the intersection of inputs supported by every selected extractor, declaring exclusions and
coverage; a common deterministic text-shortening policy is a separate representation experiment.
Never silently accept provider-specific truncation. Chunking requires an aggregation rule and
can destroy cross-column interactions; it is not an interchangeable drop-in fix. [Lost in the
Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/) also shows that supported length does
not imply equally effective use of evidence at every position, for the models/tasks it studied.
