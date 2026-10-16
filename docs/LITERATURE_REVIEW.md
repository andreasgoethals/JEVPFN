# Literature review

Primary-source findings checked through 16 October 2026. This reference records prior art,
published evidence and limits of interpretation; it is not a run plan or acceptance report.
Current choices are in [README](../README.md), provenance in [DATA_SOURCES](DATA_SOURCES.md),
execution instructions in [VSC](VSC.md), and run evidence in [AGENTS_MEMORY](AGENTS_MEMORY.md).
The read-only library pin is `de324efcca55be392feb0c01830b3b6ce364587f`; cite code snapshots
by symbol rather than drifting line numbers. Linked live sources need rechecking when pins change.

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

The project's shared shuffled CV/all-row protocol differs from those published holdouts;
compare feature combinations on our fixed folds rather than treating paper scores as directly
replicated. Pinned library defaults also differ from the paper's 2,000-iteration/early-stopped
CatBoost. [Paper release](https://github.com/alanarazi7/MulTaBench/tree/paper_version)

The pinned paper result tables distinguish frozen joint representations (`all`) from
fine-tuned representations (`ft`). A read-only inspection of the core TEXT tables and
`more_baselines/` identifies strong TabICLv2, TabM, AutoGluon-MM and TabSTAR results.
AutoGluon and TabM are excluded from this project's roster by owner decision; these published
results remain literature context only. Published ranks do not isolate the benefit of decision-derived features.
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

JEVPFN's conventional control pins E5-small-v2 to
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, uses the card's `query: ` prefix for
classification/features, attention-mask mean pooling and L2 normalization. Its long-text
extension uses nonoverlapping token chunks, weighting chunk vectors by content-token count
before final normalization; this is our declared coverage policy, not a claim to reproduce
the MulTaBench encoder exactly. No source token is silently discarded. Full 384-dimensional
vectors are reusable; per-column PCA is fitted only on the current training partition,
including each inner threshold-selection fold. Model inference remains to be verified.

### Paired inference and pretraining-overlap review

The scientific comparison is within a fixed predictor, using identical splits. TabSTAR and
ConTextTab are descriptive reference systems; their different predictors cannot isolate the
effect of replacing a text representation. A frozen encoder in the same predictor supplies
that missing conventional comparison. Observed paired gains can be reported descriptively
without a significance test, but fold SD is not uncertainty for an independent-sample mean:
training sets overlap. [Bengio and Grandvalet (2004)](https://jmlr.org/papers/v5/grandvalet04a.html)
establish the general cross-validation variance difficulty. [Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html)
supports using datasets as comparison units. The implemented optional Wilcoxon/Holm procedure
requires a complete predeclared family, separate task metrics and one five-fold mean delta
per dataset. It cannot turn this curated set into a random sample of all tabular tasks;
few classification datasets and multiple comparisons limit power. Do not choose the metric,
model or best feature arm after seeing test results and call that choice confirmatory.

Pretraining review applies to every pretrained extractor, E5 and downstream/native-text
model. Before interpreting results: (1) inspect the pinned model card and released training
source lists, (2) cross-reference dataset aliases and original sources in DATA_SOURCES,
(3) record documented overlap, plausible overlap or unknown, with source/version/date,
and (4) report any documented-overlap subset separately without silently removing datasets.
For open models with available corpora, exact text-hash matching is useful; absence of an
exact match cannot exclude paraphrases or undisclosed base-model data. Closed training sets
must remain unknown. A publication-after-training-cutoff subset is defensible only when both
dates are verified. No cutoff or absence of overlap should be inferred from a release date.

Masking names/numbers or paraphrasing can test robustness but also removes legitimate signal
and cannot prove absence of contamination. Oren et al.'s
[black-box contamination test](https://arxiv.org/abs/2310.17623) relies on likelihood access
and exchangeability assumptions; a decision-probability API does not expose those likelihoods.
Here "label-free extraction" means no row targets/statistics enter the representation pipeline;
it does not establish that pretrained weights never saw public labels. Seen/unseen-text CV
cohorts address observed repetition, not all pretraining contamination or semantic confounds.

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

The selected numeric predictors require explicit handling of categorical inputs and missing
values. Their raw-text capabilities are separate from receiving numeric decision features.

| Engine | Tasks and text treatment | Relevant interface limitation |
|---|---|---|
| TabICL v2 | Classification and regression; no built-in semantic text encoder. Strings/categories need the wrapper's encoding or explicit text embeddings. | Separate classifier/regressor checkpoints; `support_many_classes=True` enables the many-class wrapper. |
| LimiX-2 | Classification, regression and missing-value imputation. Numeric/categorical array interface; no documented raw-language encoder. | 400M model, distinct from the older 2M-parameter LimiX-2M; `LimiXPredictor` and V2 configs. |
| NVIDIA Kumo Tabular | Classification/regression over numeric and categorical tables through SDM; no native language encoder. | Task-specific checkpoints, many-class decomposition; regression point prediction is the mean of 999 predicted quantiles. Default preprocessing selects at most 500 columns. |
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

Mitra-v2's reviewed runtime imports AutoGluon despite marking it optional; no supported
independent route was verified. Causilo, Mitra-v2, EXAONE and TabDPT remain literature
references, outside the executable catalog. Weight licences and runtime constraints need
independent review when any model is selected.

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

TabSTAR and ConTextTab provide original-text controls with language encoders; their supervised
adaptation and training-context policies must be distinguished from cached label-free features.

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
evidence that this project's six-extractor/four-configuration MulTaBench comparison has
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
ability to return nine probabilities. Acceptance of wording is not validation of model behavior; the concrete rubric is centralized
in the source rather than maintained in this literature review.

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

### Deployment evidence

The publisher's [model reference](https://docs.typesafe.ai/models) documents Jev's version,
price and dynamic limits. Those references do not establish our account's allowance or bill.
Calibration against returned usage, explicit version identity and ambiguous-response handling
are empirical requirements. Current configuration and connection steps live in VSC/config;
character-based estimates must never be treated as exact counts or safe context bounds.

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
| Microsoft-Decision-1 | [9 October release](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/): hosted Qwen3.5-9B decision head, Noul/Choice/Score. | Excluded by owner decision for practical access/throughput constraints before comparative outcomes; no inference adapter or evaluation arm. This is not evidence of inferior quality. |
| Liquid d1-3B | [7 October open-weight release](https://www.liquid.ai/blog/d1-open); [publisher model card](https://huggingface.co/LiquidAI/d1-3B). Approximately 3B parameters, not 13B. Noul/Choice/Score, full probabilities, 32,768 context. | Close schema match and small self-hosted comparator. The [LFM Open License](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE) explicitly exempts qualifying nonprofits' noncommercial/research use from its commercial revenue threshold; attribution/redistribution conditions remain. Use its custom decision interface, not generated chat examples. No model downloaded. |
| Cloudflare Clef-flash | [Hosted reference](https://developers.cloudflare.com/workers-ai/models/clef-flash/): $0.038/M input, 24,576 tokens. The [public Apache-2.0 checkpoint](https://huggingface.co/Cloudflare/clef-flash) is publicly available, including the Qwen3.5-9B backbone, joint schema head and complete encoding/inference source. Hub revision observed 14 October: `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` (public, ungated). | Selected for self-hosting only. `joint_schema_model.encode_record` renders state/questions; the joint head supplies logits for every option, normalized per question. The official runtime was tested on H200; an A100 pilot is still required. The pinned local encoder defaults to 16,384 tokens and silently slices the state. Our precheck rejects overflow and retains full-precision softmax instead of the convenience API's four-decimal rounding. Do not infer parity between the hosted and local context limits. |
| AWS Strands Decider | [1 October launch](https://strandsagents.com/blog/introducing-strands-decider/); [official repository](https://github.com/strands-labs/strands-decider). Apache-2.0, Noul/Choice/Score, full probability vectors; pointer readout over supplied options. | Small self-hosted comparator. The newer [Qwen3.5 2B checkpoint](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) was created 9 October and saves a 4,096-token window. Original launch and updated checkpoint are different revisions. Verify complete input and the remaining class vocabularies; published latency is hardware-specific. |
| Perplexity Decider v1.1-27B | Original [v1 model](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) repository created 1 October; [v1.1 announcement](https://community.perplexity.ai/t/pplx-decider-v1-1-27b-our-updated-open-weights-multimodal-decision-model-is-now-available/6286), 6 October. Apache-2.0, native decision head, noncausal full attention, saved calibration temperature; Noul/Choice/Score supported by supplied inference code. | [Publisher card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) reports stronger Decision Index results, not MulTaBench gains. Local weights require about 49 GiB plus working memory; preparation rejects over 8,192 tokens. Public Hub metadata is currently private=false/gated=false, despite stale private-access wording in the card; Check tasksource training overlap. Publisher announced hosted pricing of $0.02/M for v1.1; hosted access and version pin still require verification. |
| Fastino GLiNER2.5-Decide | [Publisher release](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model): 340M encoder, Apache-2.0. However, current [deployment capabilities](https://docs.fastino.ai/concepts/gliner-2-5-decide-capabilities) describe unordered classification labels without label-description objects and direct ordered scoring to GLiDE. | Do not equate release claims with the current hosted schema. Complete probability mapping and long-input behavior of the selected local runtime need verification. `max_position_embeddings=512` in its DeBERTa config alone does not prove a 512-token hard cutoff for this relative-position architecture. Not admitted to the uniform extraction protocol yet. |
| Fastino GLiDE | [Publisher release](https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model), dated 30 September; adaptive reasoning and full probabilities. [Capabilities](https://docs.fastino.ai/concepts/glide-capabilities): 40,000 tokens per rendered state/question, 2–255 choice options, up to 255 score levels; overlong input is rejected. [Pricing](https://docs.fastino.ai/pricing): $0.15/M input, output free. | Excluded by owner choice; retained here only as related work. Hosted only in the checked evidence. Reported input usage can sum multiple internal passes, so externally counted text tokens cannot establish the bill. Freeze the version and measure account throughput, latency and billed usage in a pilot. Publisher benchmark claims are not downstream tabular evidence. |
| OpenAI Decisions / GPT-6 Luna | [Official guide](https://developers.openai.com/api/docs/guides/decisions): public beta, `POST /v1/decisions`; predicate, choice and ordered score with full distributions. $0.10/M input, no output/cache fees on this endpoint; regional/long-context adjustments can apply. | Excluded by owner choice and cost; related work only, with no adapter or campaign entry. Map nine ordered indices 0–8 back to −4…+4 (expected score minus four); preserve all probabilities and treat refusals as failures requiring review, not empty text. The model's general context specification does not replace testing this endpoint's actual limits/usage. Snapshot pin and account throughput remain to verify. |
| Intern-Decision-4B | [Publisher card/source](https://huggingface.co/internlm/Intern-Decision-4B): Apache-2.0, Qwen3.5-4B, one forward pass over candidate-symbol logits followed by checkpoint calibration; Noul/Choice/Score. Default complete-input limit 8,192, 62 options, strict rejection. | Selected open-weight candidate, pinned at `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd`. All retained tasks have at most 30 classes or nine ordered levels; option capacity fits. Exact rendering and real hardware compatibility require independent verification. Preserve published calibration; do not fit it with benchmark targets. Author-run accuracy/latency do not establish our task performance. |
| Torchcast Decision 27B | [Publisher card/source](https://huggingface.co/torchcast-ai/torchcast-decision-27b): all three typed outputs, full probabilities, published context up to 262,144; weights CC BY-NC 4.0 and code Apache-2.0. Choices above 26 options take additional rounds. | Excluded by owner choice; no adapter or exact token audit here. Adds another large GPU runtime; the 30-class Wine Review task must exercise its multi-round path. Published Decision Index results are author-run and guided model development; inspect training overlap and actual GPU throughput rather than using the headline to select it. |

The six selected interfaces can represent binary probabilities, complete class vectors and
ordered regression probabilities. This does not establish coverage or calibrated target-unit
prediction. Runtime-specific rendering and calibration must be preserved for fair comparison.

Native rendering, attention, option encoding and calibration differ despite shared typed
semantics. Strands' reviewed engine has mutable option offsets, so concurrency is not assumed.
Clef's convenience response rounds probabilities and its local renderer truncates state;
retain precise probabilities and enforce complete-input counts before inference. Connection
recipes and acceptance evidence live in VSC/run reports rather than this literature record.

### Public decision benchmarks

Public comparisons already exist. [Decision Index](https://github.com/apolinario/decision-index)
has a reproducible 37-benchmark public component; its full score also uses private tests.
[JevBench by Benchmark Heaven](https://huggingface.co/datasets/benchmarkheaven/jevbench)
reports decision quality/calibration/latency/cost, but its released snapshot is aggregate-only
and cost fields may be estimates. The similarly named metamorphic coherence benchmark is a
separate project. JEVal and the 37-dataset Jev paper below provide academic evidence with
different coverage and sometimes different calibration conclusions. The rapid release wave
is observable; its size does not establish model quality, independent architectures or
useful tabular features. This project's contribution is a controlled, label-free comparison
of reusable text representations on 18 mixed-feature datasets, measuring incremental
prediction benefit, coverage, probability behavior and extraction cost. It must not be
framed as the first general decision-model benchmark.

### Research evidence and its limits

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
| [General Decision Models / JEVal](https://arxiv.org/html/2610.03935v2), 2 October; v2 12 October | 11,257 bilingual instances, 36 datasets, ten domains, 25 configurations; also agent/social simulations and SII-Decision distillation (renamed from InnerJev in v1). Explicit-evidence tasks fare better than exact computation, specialist knowledge and faithful uncertainty estimation. | Broad evidence for the model class, including limitations. Local decision quality does not automatically transfer to reliable system-level outcomes or downstream tabular benefit. |
| [SearchJev](https://arxiv.org/html/2610.05107v1), 4 October | Six search-decision types; soft-label learning combines cross-entropy/Brier losses and held-out temperature calibration. Faster decisions and better reported search-agent outcomes than its specified baselines. | An example of specialized probability-focused training, not validation of general-purpose text-to-tabular regression. |
| [Political Science Replications](https://arxiv.org/html/2610.06625v2), 5 October; v2 updated 6 October | Seven published annotation/scaling tasks; human labels and commercial/open LLM comparisons. Jev often approaches comparator capability, but has no cost advantage over the study's commercial batch-price baseline and does not consistently beat the open model's calibration. | Independent evidence is mixed. Report actual task-specific cost and calibration; do not extrapolate universal superiority from low listed input pricing. |
| [Readout Stability](https://arxiv.org/html/2610.07716v1), 6 October | Seven model families and ten datasets; studies candidate-menu interventions and inference-time compute. Ranking stability is stronger than probability stability; extra order-permuted passes mainly improve calibration in the tested setting. | Candidate vocabulary/order and repeated execution are different from cache reuse. Do not transfer the paper's menu-selection results to this project's fixed full-class vocabulary without testing. |

[SecJev, 2 October](https://arxiv.org/abs/2610.03073) is a security-specialized decision-model
family, with reported gains from domain-specific training. Its scope does not justify replacing
a general extractor on heterogeneous table datasets. Likewise, the [network-traffic study,
30 September](https://arxiv.org/abs/2610.00376), finds trained tree ensembles stronger than its
Jev configurations on numerical packet data; supervision budgets differ, so this does not
isolate architecture as the cause or test our text-augmentation method.

Family release dates and checkpoint update dates differ; these are vendor releases, not
independent demonstrations of tabular benefit. Uniform typed schemas establish representation
feasibility, while actual GPU coverage/quality must be tested. The reviewed papers support
measuring incremental predictive utility, probability behavior and cost under a fixed protocol;
they do not guarantee publishability or justify claiming the first decision-model benchmark.

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
