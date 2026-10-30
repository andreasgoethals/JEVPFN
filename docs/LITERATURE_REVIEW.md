# Literature review

JEVPFN-specific research evidence and limits of interpretation, reviewed on
**24 October 2026** (SII interface and statistical resolution clarified 26 October;
native-text versions and controls 27 October). It is neither an execution report nor a
list of future experiments. Methods: [README](../README.md); dataset meanings:
[DATA_SOURCES](DATA_SOURCES.md); operations: [VSC](VSC.md).

Since 30 October new Jev calls pin **1.14.0**. The 1.13 evidence below is historical and
does not claim that every documented behavior persists in 1.14; the 1.14 interface and
sensitivity results were reviewed separately from saved responses
([AGENTS_MEMORY](AGENTS_MEMORY.md)). Protocol and library pins are unchanged. A source
review or passing interface test does not establish real GPU compatibility.

## 1. Evidence, scope and provenance

The independent libraries remain read-only at these pins:

| Library | SHA used for this review | Relevant coverage |
|---|---|---|
| [DM Library](../dm-library/README.md) | `5f1b3f8f1611681579aba052a86c2626a6a17b96` | Sixteen selected arXiv papers, fifteen web-source records and official implementation snapshots; refreshed 23 October. |
| [TFM Library](../tfm-library/README.md) | `702d9ab0b66f6639ba8d443ff9c4a91231964465` | Tabular predictors, text representations, benchmarks and source snapshots. |
| [VSC Library](../vsc-library/README.md) | `3bcdf55e576d10161d6189b40ac39380bbba9eca` | Infrastructure documentation and execution examples; operational evidence is separate from scientific evidence. |

On this review date the canonical DM Library and mounted submodule had the same clean
HEAD. There were no newer canonical papers missing from this pin. The DM overview's
introductory “seven web-only” count is stale: its actual web-source table contains fifteen.
This upstream documentation discrepancy does not change the project roster.

All sixteen filed DM research papers are **preprints**; no peer-reviewed venue is confirmed
in the curated records. A publisher model card, benchmark claim and independent evaluation
are different evidence types. The libraries contain a curated, dated collection, not an
exhaustive global search. Filed paper versions and released checkpoints can differ; neither
a release date nor a repository snapshot establishes a model's training cutoff.

Detailed summaries and source maps belong in [DM summaries](../dm-library/SUMMARIES.md),
[DM synthesis](../dm-library/SYNTHESIS.md) and [DM implementations](../dm-library/REPOSITORIES.md).
The sections below retain the findings that affect this project's claims and implementation.

## 2. Closest prior art and the contribution boundary

**Jev-derived features plus TabPFN already exist.** The public
[tab-jev project](https://github.com/edamame-labs/tab-jev) converts option probabilities into
numeric features and includes a TabPFN-3.5 API backend. Its Kickstarter demonstration
compares table-only prediction, text-only Jev, lexical encodings and combined pipelines at
fixed label budgets; it also offers manual rubrics, few-shot examples, supervised
calibration and a local Kev-to-TabICL backend. This is direct feature-engineering prior art,
although not a verified
three-task, multi-dataset replication of JEVPFN.
[Evaluation procedure](https://github.com/edamame-labs/tab-jev/blob/main/demo/README.md)

Other particularly close sources are:

| Source | Evidence and distinction |
|---|---|
| [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536), Zhang, Ye and Zha, v1 | Four decision models and three generative LLMs make direct decisions on row classification, fact verification and anomaly detection, with 21 applicable request rewrites. This precedes our comparison of decision models on tabular tasks. It does not evaluate cached text-only representations inside fixed trained predictors across binary, multiclass and regression tasks. |
| [Estimating Uncoded Crash Factors with Tabular Foundation and System One Models: Kumo Tabular and Jev](https://arxiv.org/abs/2610.10321), Rafe and Das | Combines a coded-record Kumo reader, a narrative Jev reader and human recalibration for population estimation. It is prior complementary use of a TFM and Jev; supervised recalibration differs from label-free extraction. |
| [TypeSafe feature-discovery cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery) | Generates Jev-derived features for CatBoost wine-score regression. An LLM proposes questions using training labels and errors. It establishes public feature engineering, while differing from the fixed mechanical questions here. Its results are provider evidence. |
| [TILE: Bridging LLMs and Table Foundation Models for Tabular Learning with Strings](https://arxiv.org/abs/2610.17065), v1 | Studies contextualized representations of serialized strings with intermediate-layer pooling, compression and tabular predictors. A representation interface between language and table models is already an active research direction. |
| [TabPFN beyond Tabular Data](https://arxiv.org/abs/2607.11007) | Uses TabPFN on frozen image, text and audio representations. Frozen semantic embeddings plus a TFM are established prior art; effectiveness depends on representation dimension and label budget. |

The potential contribution is **measuring the utility, compression, robustness and cost of
task-conditioned probabilistic text representations within fixed predictors**, against
frozen embeddings and native-text reference systems. Separate-column and joint-column
representations, three task types and cache reuse make this a specific controlled study.
It is not the invention of Jev-plus-TabPFN, the first decision-model benchmark, proof of a
new decision architecture, or a demonstrated performance advance before results exist.

The current eighteen tasks are a deliberate subset of MulTaBench's twenty core text tasks.
Spotify and Zomato remain visible in exploration with documented exclusions. Results should
be attributed to this selected collection, not all text-containing tables. A literature
review cannot prove that no related work exists or guarantee publishability.

## 3. What recent decision-model research establishes

The following studies motivate checks already represented in the code or interpretation.
Reported results are the authors' measurements, not results reproduced by JEVPFN.

| Research evidence | Finding relevant to this project | Consequence |
|---|---|---|
| [System1-Arena: Which Decisions Can Be Left to System 1?](https://arxiv.org/abs/2610.21244), v1 | Common probability-scoring requests across 30 systems, 105 tasks and 92,125 questions; static accuracy/calibration and closed-loop success give different rankings. Its headline control table uses 127 of the 138 supplied environments. | The broader decision-model benchmark category already exists. Our downstream representation question is different. One request per example and possible pretraining exposure limit conclusions about stability and independence. |
| [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536), v1 | Aggregate accuracy can hide large changes in individual predictions under table rewrites. Confidence tracks stability more reliably than correctness. | Preserve per-row predictions and probabilities; assess option and context interventions separately. Re-querying changes compute and is a separate policy, not free robustness. |
| [Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling](https://arxiv.org/abs/2610.22131), v1 | Predicts answer placeholders in one causal pass without feeding one answer into other fields. A known-distribution pilot improves after temperature scaling; its 96 cases are paired variants of 48 settings. | Preserve the checkpoint's temperature. Better author-reported accuracy does not imply better probabilities on our tasks; a fixed short-request GPU latency does not predict full extraction throughput. |
| [Releasing the Inner Jev: A Simple Recipe for Open Decision Models](https://arxiv.org/abs/2610.17205), v1 | Frozen answer-letter readout is competitive; backbone LoRA provides more improvement than head-only training. Validation resembles public decision benchmarks and selects calibration/checkpoints. | “Decision model” is not one architecture. Do not attribute downstream gains solely to a specialized head. This paper's **Jeff** release is distinct from TypeSafe Jev and SII-Decision. |
| [LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to Fine-Tune Them](https://arxiv.org/abs/2610.02076), v3 | Joint, prefix-free candidate-suffix scoring can work without adaptation; adaptation gains concentrate in represented domains. Full suffix scoring and optional thinking require additional computation. | An ordinary LLM probability readout is a relevant architecture control if making architecture claims, but is not automatically equivalent to a one-pass decision head or an approved roster addition. |
| [AnyJev Technical Report](../dm-library/SUMMARIES.md#anyjev-technical-report) | Separates frozen-backbone capability, label-prior correction, option rotations and adaptive inference cost. | Option-order averaging can change both quality and latency. Fixed one-pass extraction and averaged extraction must not be pooled as the same method. |
| [TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models](https://arxiv.org/abs/2610.11392), v2 | Rule-generated tasks separate calibration, framing and asymmetric decision costs; finite-sample calibration errors have a nonzero noise floor. | Top-class agreement alone misses probability shifts that change our numeric features. Calibration metrics require appropriate ground truth and sample-size interpretation. |
| [Are AI Decision Models Biased? A Case Study of Jev](https://arxiv.org/abs/2610.19619), v1 | Framing and irrelevant context can alter probabilities without changing the selected class. Its 41,500-response main study and separate 15,000-response conjunction probe are different samples. | Report probability distance, class flips and expected-score change separately. Many repeated calls to few scenarios do not supply independent scenarios. |
| [General Decision Models: Benchmarking and Insights Beyond Jev](https://arxiv.org/abs/2610.03935), v2 | JEVal covers 11,257 bilingual instances and 36 datasets, with analytic probability and interactive tests. Correct argmax decisions can coexist with poor uncertainty or system outcomes. | Decision Index, JEVal and downstream tabular prediction answer different questions. SII-Decision author results do not establish MulTaBench gains. |
| [JevOut: Natural Context Can Flip Decision Models](https://arxiv.org/abs/2609.30243), v2 | Answer-preserving additions can redirect predictions under an explicit attack-search budget. | Treat supplied text as untrusted data. Adaptive adversarial success is a robustness stress test, not the ordinary-cell error rate. |
| [Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?](https://arxiv.org/abs/2609.33209), v1 | Logical negation/complement and categorical formulations can disagree even when each response is normalized. | Separate binary queries renormalized afterward do not reproduce a native multiclass Choice distribution. Freeze the requested answer space. |
| [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647), v1 | Broad short-text evidence is positive in some domains and weaker for specialist, low-resource and fine-label tasks. It uses one request per item and cannot resolve closed pretraining exposure. | Measure heterogeneous effects and failure coverage rather than assuming Jev works uniformly across tables. |
| [JEV-as-a-Judge: Accept When Confident, Escalate When Unsure](https://arxiv.org/abs/2609.26550) and [SearchJev](https://arxiv.org/abs/2610.05107) | Study confidence routing and specialized search decisions, including task-specific calibration/training. | Their quality/efficiency claims concern specific workloads. They do not certify general-purpose regression features or universal confidence thresholds. |
| [Benchmarking System One decision models against trained classifiers and language models for automated decision gates](https://arxiv.org/abs/2610.00376) | Matched gates include trained classifiers, calibrated decisions and cascade costs. | Ordinary trained predictors remain necessary comparators. A fast decision API need not outperform a learner with labels. |
| [StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions](https://arxiv.org/abs/2610.17309), filed v1 | Combines instant typed readout with selective reasoning and learned probability objectives. | Reasoning and option-grouping introduce extra passes; public benchmarks guiding development are not independent validation. |

The papers generally support the usefulness of compact typed interfaces, while giving
mixed evidence for calibration, framing invariance, causal understanding and robustness.
They do not establish that the twelve selected extractors produce interchangeable beliefs.
The comparison must preserve their differences rather than normalize them away.

Public tools already apply these interfaces to individual text columns. The
[vd-jev VisiData plugin](https://github.com/ondata/vd-jev) returns typed columns and its
procurement demonstration shows that documentary class descriptions change mappings of
organization names. This is a developer demonstration, not an independent benchmark;
it also illustrates that a short name may lack the information needed for the true label.

## 4. Why these twelve extractors and how their interfaces differ

The selected roster combines one funded hosted reference and eleven open-weight releases
that can express the complete answer spaces in the retained tasks. The largest target
vocabulary has thirty classes; regression uses nine options. Revisions, licences, configured
context bounds and base-model dependencies belong in
[extractor configuration](../config/2.%20feature_creation/extractors.yaml), not duplicated
mutable tables. Open-weight availability does not mean zero compute cost or unrestricted licensing.

| Selected extractor | Source behavior that must survive the adapter |
|---|---|
| [Jev](https://docs.typesafe.ai/api) (1.14.0; reviewed as 1.13) | Hosted Noul, Choice and Score. Pin the model; preserve usage/version and raw response. Organization throughput and context/billing bounds are separate concepts. |
| [Liquid d1-3B](https://huggingface.co/LiquidAI/d1-3B) | Custom decision runtime, not ordinary generated chat. LFM Open License is a conditional open-weight licence, not a blanket permissive open-source claim; qualifying research permissions and redistribution conditions must be respected. |
| [Liquid d1-2B](https://huggingface.co/LiquidAI/d1-2B) | Separate LFM3 successor, text-only in this study; same questions and explicit mean readout. Its new runtime/window need independent acceptance; vendor benchmark improvements are not downstream evidence. |
| [Perplexity Decider v1.1-27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | Native decision head, noncausal full attention and saved calibration temperature. Generic causal-LM next-token inference is not equivalent. |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) | `joint_schema_model.encode_record` and its joint head render/score the schema. The convenience encoder slices overlong state; JEVPFN rejects overflow and retains full-precision softmax. Hosted and open-weight context limits are not interchangeable. |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | Shifted answer-slot logits over legal candidates, then checkpoint-bound temperature. Keep the published calibration; do not refit it on benchmark labels. |
| [Strands Decider 2B](https://github.com/strands-labs/strands-decider) | Pointer readout over supplied options and native ordered-scale handling. Mutable option offsets require serialized access to a shared engine. The original release and selected Qwen3.5 checkpoint are different versions. |
| [H2O Lightning 4B](https://huggingface.co/h2oai/h2o-lightning-4b) | `pooled_readout.label_logits` and `h2o_lightning_shim.format_answer` use mixed attention/readout and type-specific processing. Its binary formatter imposes a 0.801 winning-probability floor. Preserve and disclose it; require publisher-anchor parity rather than claiming calibrated binary probabilities. |
| [SII-Decision-27B-Full](https://huggingface.co/jylin001206/SII-Decision-27B-Full) | `sii_decision.prompts.prompt` and `Readout.forward` score ordered single-token labels at temperature one. Use the full checkpoint and complete descriptions. |
| [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) | `kev.model.encode`, `DecisionModel.forward` and `PointerHead.forward` implement branch-aware readout with checkpoint temperature. Native state/branch bounds differ from the project-wide complete-request check. |
| [Decision3 d3-turbo](https://huggingface.co/vllm-sr/d3-turbo) | `d3_format.render`/`encode_split` and native probability readout use prefix-bidirectional attention. An ordinary vLLM chat server is not a substitute. |
| [Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | `serving_schema.prepare_prompts`, candidate logits and extended candidate-code schema. Pin the adapter and Qwen base separately; binary enum fields retain both class descriptions and the extended schema must retain all thirty Wine classes together. |

Implementation comparison points are `src/jev/registry.py:ExtendedCounter`,
`src/jev/open_runtimes.py`, `src/jev/extended_runtimes.py:ExtendedRuntime` and
`src/jev/provider.py:response_features`; source maps are in the DM Library.
Matching a schema is not sufficient: attention, calibration, token rendering, option order,
numeric mapping, rounding and complete-input handling also need real runtime contracts.

The pinned SII publisher snapshot (`dm-library/repositories/SIIDecision.txt`,
`sii_decision.prompts.option_labels` and `prompt`) filters A–Z and two-letter candidates
to 255 single-token labels. Its training and inference use standalone label IDs after
`The correct answer is (`. Labels beyond 26 are therefore not inherently multi-token.
JEVPFN additionally validates distinct token IDs and the reviewed answer cue before
loading weights. This preserves the publisher interface; it does not prove real GPU
accuracy or calibrate the 30-class Wine task.

The common **semantic** inputs are the same task definition, question, named text and answer
descriptions/order. Native requests are not byte-identical: each checkpoint requires its
own tokenizer, serialization and answer representation. The common feature schema is one
positive-class probability, one probability per multiclass label, or a probability-weighted
direction on −4…+4; complete response distributions remain stored.

Jev's observed two-decimal and Strands' four-decimal outputs can require bounded simplex
reconstruction. The decoder permits only provider-specific rounding envelopes, retains raw
values and rejects larger inconsistencies. Provider “confidence” fields are not a shared
uncertainty scale. Derived entropy, maximum probability and directional variance have a
common definition but still inherit different calibration and temperature policies.

Other public releases are not automatically suitable replacements. Microsoft, OpenAI
Decisions and hosted GLiDE are excluded by the owner's access/cost/scope choices, not
evidence that they are worse. Jeff's checked serving release supports only sixteen options;
StartLux/Torchcast grouping beyond twenty-six would change the thirty-class task's
computation and probability meaning. Fastino's checked schema and full ordered-distribution
mapping were not established for this common protocol. Reasoning, multi-round grouping or
domain-specific extractors belong to separate methodological choices. Adding every release
would not by itself improve an interpretable, reproducible comparison.

**28 October addition: Liquid d1-2B.** The [official model card](https://huggingface.co/LiquidAI/d1-2B)
and publisher code at `cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41` identify a distinct LFM3
checkpoint with a 128,000-token configured text window and an LFM Open License v1.0.
Vendor DecisionBench/ImajevBench improvements do not establish downstream tabular gains.
We retain d1-3B and add d1-2B as `liquid_lfm3`, using identical canonical text/questions,
complete class distributions and the same nine directional levels. `api.py:answer` defaults
to a median level, so `SelfHostedClient.evaluate` explicitly passes `score_readout="mean"`.
`lfm3.py` registers the non-thinking answer lead; the CPU counter reads that exact registration
without importing Torch. Images, multi-select and native numerical-estimation outputs are
outside the approved comparison. Its pinned Transformers 5.19 runtime is isolated from the
older counters. CPU rendering is verified locally; real B200 acceptance remains pending.
This is a targeted official-source check beyond the dated library corpus, not a global refresh.


## 5. MulTaBench, embeddings and native-text predictors

### Dataset scope and target-aware adaptation

[MulTaBench v2](https://arxiv.org/html/2605.10616v2) has forty core datasets:
twenty text and twenty image tasks. The extended collection adds twenty of each; it is not
forty core text tasks. Its curation requires joint multimodal signal and evidence that
target-aware encoder adaptation helps several curation learners. This selects tasks
enriched for multimodal/adaptation benefit, which limits population-wide generalization.

The paper's five random-seed holdouts use a training cap and bounded test subsets, rather
than five disjoint outer CV folds over all rows. JEVPFN's approved all-row shuffled five-fold
protocol is intentionally different; published scores are context, not directly paired
baselines. [Pinned split source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/preprocessing/splits.py)

The 27 October recheck of Figure 4 found TabICL v2 narrowly highest for text-tabular TAR,
close to TabPFN-2.5; the largest frozen text bar belongs to TabPFN-2.5. AG-MM is highest
among its original-text E2E references, ahead of TabSTAR and ConTextTab. These are aggregate
normalized means with overlapping uncertainty, not proof of a universal winner. Table 10
instead measures TAR-over-frozen win frequency and is not a ranking of absolute accuracy.
The paper uses AUC for classification (macro one-vs-rest for multiclass in its implementation)
and R² for regression. The owner selected these headline metrics; all other scores remain
recorded. We keep six fixed predictors and three native references, not every historical
learner, image experiments, text-only arm or supervised TAR condition from the paper.

Appendix A.1's target-aware representation (TAR) shares E5-small across columns and adapts its
last three layers with LoRA (rank 16, alpha 32, dropout 0.1). Column-name/value examples use
training-row targets; an internal 90/10 split selects the checkpoint. Regression adaptation
uses twenty equal-frequency target bins. This is valid supervised representation learning
if bins, adaptation, selection and compression use only the relevant training partition.
The paper describes outer-test isolation; this review does not certify every upstream line.

JEVPFN's decision features use documentary target semantics without dataset-label fitting.
Call this **task-conditioned feature extraction**, not supervised representation learning.
Adding TAR would require training inside every applicable inner/outer partition, substantial
additional compute and a separate experiment. Frozen controls therefore answer a narrower
question than “better than every possible fine-tuned language pipeline.”

[STRABLE: Benchmarking Tabular Machine Learning with Strings](https://arxiv.org/abs/2605.12292)
emphasizes that strings include names, codes, numbers, categories and prose. A compact
decision feature can encode a category or familiar entity rather than natural-language
reasoning. Text burden, missingness, exact reuse, named columns and seen/unseen-text cohorts
should accompany performance, without claiming that these diagnostics identify causal mechanisms.

### Frozen encoder controls and PCA

[E5](https://arxiv.org/abs/2212.03533) learns text similarity through contrastive pretraining;
it returns a vector, not task-class probabilities. E5-small-v2 is an inexpensive reproducible
MulTaBench control, not a claim about the best current encoder. The selected E5-large-v2 and
[Qwen3 Embedding](https://arxiv.org/abs/2506.05176) add scale/family comparisons. Strong MTEB
results do not establish which encoder will best support these tabular predictions.
[E5-small card](https://huggingface.co/intfloat/e5-small-v2),
[E5-large card](https://huggingface.co/intfloat/e5-large-v2),
[Qwen3-Embedding-0.6B card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

The approved dimensions are 384 for E5-small and 1,024 for E5-large/Qwen. E5 uses its
`query: ` feature prefix, masked mean pooling and L2 normalization; Qwen uses documented
last-token pooling without a query instruction for these unconditional vectors.
Nonoverlapping chunks, content-token-weighted aggregation and normalization cover long
text. This is a declared project policy, not equivalence to full-window contextualization
or MulTaBench's original preprocessing. Qwen's chunk length is below its native window.

Per-column PCA32 preserves separate column blocks; joint PCA256/128/64/32/16/8 compresses
the concatenated text vectors. Non-text features bypass PCA. Joint PCA can capture
cross-column covariance but high-variance columns can dominate. Per-column versus joint
at the same component count also changes total width; it is a grouping/compression
comparison, not an isolated grouping effect. Explained variance does not establish
predictive relevance, and none of these dimensions is assumed optimal.

Imputation and PCA fit on training rows only, including each inner threshold fold.
Validation/test rows are transformed by that fitted map and receive exactly the same
output columns. Frozen embeddings can be cached across folds; fitted PCA cannot.
See `src/data/text_embeddings.py` and
`src/evaluation/embedding_control.py`, `prepare_pca.py`, `pca_cache.py`.

A useful fourth representation family is word/character **TF-IDF**, optionally followed by
TruncatedSVD. Length/count controls do not replace lexical representations. This remains a
scope limitation: the owner declined these extra baselines on 27 October. TF-IDF is a
weighted vocabulary-count representation; SVD compresses it, not a second pretrained encoder.
Vocabulary, IDF and SVD would need fold-local
fitting. [Official text-feature guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

### What the prediction systems mean by text support

The six fixed predictors compare representations while retaining original non-text columns.
Raw categorical strings are not automatically semantic language encoding.

| System | Interpretation and important limitation |
|---|---|
| TabPFN-3.5 local | Its selected interface uses our numeric/categorical features. TabPFN v9's optional `TextTransformer` uses skrub character TF-IDF/SVD; this is not the hosted Plus/Thinking semantic workflow, and is not exclusive to 3.5. |
| TabICL v2 | Classification/regression checkpoints with numeric/categorical inputs; no native general-language encoder in the selected recipe. Many-class handling must retain every class. |
| LimiX-2 | V2 numeric/categorical predictor, distinct from the older LimiX-2M. `RebalanceFeatureDistribution.fit_transform` already fits on training rows despite the confusing `quantile_norm_all_data` name. Other upstream filtering/category/polynomial steps inspect query rows; our declared training-only adaptation is documented in [README](../README.md#models-and-comparisons) and still requires real 0B validation. |
| Kumo Tabular | NVIDIA SDM single-table model, not relational KumoRFM. The pinned recipe count-encodes some categories and selects at most 500 columns. JEVPFN rejects feature loss; 500 is a recipe bound, not a proven architectural maximum. |
| CatBoost | Has separate native text tokenizers/dictionaries/calculators, but the selected main comparison uses numeric/categorical input plus the explicit representations. |
| Linear/logistic regression | Requires fold-local imputation, categorical encoding and scaling; has no implicit language encoder. |

Sources: [TabPFN text source](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/src/tabpfn/preprocessing/text.py),
[TabICL](https://github.com/soda-inria/tabicl),
[LimiX pin](https://github.com/limix-ldm-ai/LimiX/tree/516bf396333feb3198cf7aff8a6c10421f218e24),
[Kumo recipe](https://github.com/NVIDIA/structured-data-models/blob/bd2d55db104679393fcd6afe13305f37e40b5467/sdm/models/kumo/tabular/recipe.py),
[CatBoost calculators](https://catboost.ai/docs/en/references/text-processing__feature_calcers).
Kumo point regression averages its 999 predicted quantiles; probability columns are reordered
to the dataset vocabulary. Source compatibility does not prove full-fold GPU capacity.

The three native-text references have different learning procedures:

- **[TabSTAR](https://github.com/alanarazi7/TabSTAR)** uses text representations and supervised
  task adaptation. Its internal validation and verbalizer fit use the supplied training fold.
  Required adaptation is part of this model, not a new HPO campaign.
  The adapter now explicitly selects **TabSTAR v1.1**, `tabstar.tabstarv1p1_model`, through
  package 1.2.2 with separate classification/regression checkpoint revisions. The original
  interface remains in the package but is not selected. Offline assets include the E5
  dependency, and the published three-epoch prediction averaging is retained. The current
  model cards describe expanded pretraining, overlap
  filtering and fixed downstream LoRA adaptation; these are publisher claims, not an
  independent reproduction or proof that all eighteen tasks are uncontaminated.
  [Classification card](https://github.com/alanarazi7/TabSTAR/blob/master/docs/model_cards/tabstar-v1.1-cls.md),
  [regression card](https://github.com/alanarazi7/TabSTAR/blob/master/docs/model_cards/tabstar-v1.1-reg.md).
  The pinned `repositories/TabSTAR.txt`, sections `tabstarv1p1_pretrain/README.md` and
  `finetunability/datasets.py:select_finetunability_datasets`, explicitly identify **TabArena**
  as the epoch-selection suite (fold 0, capped subsamples, Elo against baselines). The same
  README describes exclusion fingerprints from every MulTaBench `data.csv` before pretraining.
  Thus the review's claim that the epoch-selection suite is unspecified is not supported by
  these sources. This is publisher-documented procedure, not an independently reproduced
  guarantee of no overlap through source aliases or the pretrained text encoder. We retain
  v1.1 and do not treat this as evidence of leakage from JEVPFN's test folds.
  The source/version check used TFM Library `e604ade99f7cf135f41c17a49e1f91da528e835d`;
  the earlier pins in section 1 identify the broader literature review.
- **[ConTextTab / SAP RPT-1 OSS](https://github.com/SAP-samples/sap-rpt-1-oss)** embeds column
  names/values and uses labeled training rows as semantic in-context examples. The publisher
  encoder truncates individual strings to 512 tokens. Our all-training-row, one-bag recipe
  must pass resource tests; the default 8,192-row context is not evidence that arbitrary
  full folds fit. Code and research-restricted checkpoint licences differ.
- **[AG-MM](https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.fit.html)**
  is AutoGluon MultiModalPredictor with a fixed `medium_quality`/ELECTRA-base recipe,
  numerical/categorical encoders and fusion. It is not a TabularPredictor search. HPO and
  model ensembling are disabled; internal validation remains training-only.

“E2E” in MulTaBench denotes systems accepting original text through their prediction
pipeline; it does not mean all of them fine-tune in the same way. MulTaBench does not
identify TabSTAR and ConTextTab as its two strongest methods: AG-MM and other methods are
competitive, while ConTextTab is weaker in its text analysis. All three are now descriptive
reference systems, not exhaustive SOTA coverage. Their different predictors cannot isolate
the benefit of replacing a text representation within a fixed predictor.
[MulTaBench results and ablations](https://arxiv.org/html/2605.10616v2)

## 6. Leakage protection, uncertainty and valid interpretation

### What label-free extraction does and does not guarantee

Allowed context consists of sourced observation units, target definitions/units, original
class vocabulary/meanings, text-column names and row text. Row targets, frequencies,
target summaries, correlations and model feedback are excluded. Documentary descriptions
still require human provenance review: schema validation cannot certify that prose contains
no answer leakage. The exhaustive question audit checks file hashes, schema, row mappings,
fixed questions and empty-input behavior; it cannot prove perfect linguistic interpretation.
See `src/jev/question_audit.py:audit_dataset`.

The main comparison uses the same outer splits across representations, with stratification
for classification and shuffled KFold for regression. Binary thresholds maximize F1 using
three-fold out-of-fold predictions from the outer training partition, followed by a refit
on all outer training rows. PCA, preprocessing and any internal model validation remain
inside their training partitions. Multiclass uses the full probability vector/argmax;
regression retains point predictions and applicable continuous-target metrics.
[Threshold selection guidance](https://scikit-learn.org/stable/modules/classification_threshold.html),
[Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)

Shuffled row CV estimates performance for held-out rows under this distribution. It does
not estimate generalization to new times, entities or disjoint text identities. Repeated
text is not automatically leakage under this approved estimand, but can permit memorization
and make apparent semantic benefit less general. Text-only seen/unseen/empty cohorts are
defined from shared folds, so within-cohort paired arm comparisons remain valid.
They do not prove a causal mechanism. See `src/evaluation/text_overlap.py`.

Pretraining overlap remains a separate limitation for every pretrained extractor, encoder
and predictor. Review model cards/training lists against dataset aliases and original
sources; record documented, plausible or unknown overlap with evidence and dates.
Exact matches help when corpora are available, but absence of a match does not exclude
paraphrases or undisclosed base-model data. Release dates are not training cutoffs.
Paraphrases or entity masking also remove useful signal and cannot prove clean pretraining.
[Oren et al.'s contamination test](https://arxiv.org/abs/2310.17623) requires likelihood
access/assumptions not supplied by a typed-probability API. “Label-free” therefore describes
our pipeline, not a guarantee that pretrained weights never saw public labels.

### Comparisons and probability meaning

Decision features receive target semantics; frozen E5/Qwen vectors do not. Comparing them
estimates practical representation utility, not solely architecture superiority.
The active benchmark does not include separate presence-only, text-length or shuffled-feature
controls. Missingness is handled consistently, but the main comparisons do not isolate semantic
content from length/availability cues. Claims concern predictive utility, not a causal mechanism.
The owner also declined a lexical TF-IDF/SVD family.

A binary value is evidence for a supplied class, not an empirically calibrated
`P(dataset target | all original features)`. Multiclass probabilities concern the whole
provided option set, not independent binary answers. A regression expectation on −4…+4
is an analyst-defined summary of ordered evidence levels, **not target units**. Equal
numeric spacing is a rubric assumption, and opposite strong evidence can average to zero.
Retaining nine probabilities makes entropy/variance or full-distribution sensitivity possible.
Calibration claims require suitable labels, definitions and task-specific evidence;
[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html) does not imply that a
provider's temperature automatically calibrates these new tasks.

The primary representation families in Experiments 1–3 are prespecified. No selected-winner combinations are part of this paper. A future combination selected
using these outer-test results would be descriptive/exploratory.
Reusing the same test rows after selecting a winner cannot establish an independent
confirmatory gain. Recording many metrics is useful, but a headline claim chosen afterward
must be identified as exploratory. Full out-of-fold predictions, probabilities, row/split
identities and fit/predict timings support later transparent reanalysis.

Fold standard deviation is descriptive, not an independent-sample confidence interval:
outer training sets overlap. The optional paired Wilcoxon/Holm analysis uses per-dataset
mean deltas, a complete predeclared comparison family and separate task metrics.
With only three binary, six multiclass and nine regression datasets, ideal minimum
two-sided exact p-values are 0.25, 0.03125 and 0.00390625 before correction. Binary
significance at .05 is impossible with three independent dataset units under that test.
Report paired effects, coverage and limitations rather than manufacturing power from folds.
[Bengio and Grandvalet, 2004](https://jmlr.org/papers/v5/grandvalet04a.html),
[Demšar, 2006](https://jmlr.org/papers/v7/demsar06a.html)

The implementation explicitly uses SciPy's `method="auto"`, dropping zero differences;
the ideal exact bounds above are resolution diagnostics, not a promise of an exact
null distribution or a power calculation. Reports retain the nonzero dataset count,
method and correction-family size. Holm uses ordered thresholds .05/(m−rank+1), so a
six-dataset contrast can pass later in a mixed family even though it could not pass
the first threshold alone. Do not select a smaller family after viewing results.
[SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html),
[Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

## 7. Question clarity, context and useful diagnostics

Clarity means explicit referents, relevant definitions and distinct alternatives; it does
not establish an optimal prompt. Grice's [Logic and Conversation](https://danielwharris.com/teaching/spring16/readings/Grice_1975_LogicConv.pdf)
concerns human discourse, not a validated decision-model recipe.
[Learning from Task Descriptions](https://aclanthology.org/2020.emnlp-main.105/) and
[Gen-Z](https://proceedings.iclr.cc/paper_files/paper/2024/hash/af7cc9e2366b8f8837a6ef339810277a-Abstract-Conference.html)
motivate explicit task/class definitions, but do not establish gains for all selected models.

“Higher/lower” requires a comparison class. Documentary descriptions specify the kind of
observation, while the model supplies background expectations without dataset target
statistics. The weak/moderate/clear/strong nine-level rubric makes levels distinguishable;
it does not establish a common numerical reference across models or datasets.
[Kennedy, Vagueness and Grammar](https://semantics.uchicago.edu/kennedy/classes/s06/readings/kennedy05.pdf)

Prompt-based studies find that apparently instructive wording can perform for unexpected
reasons and that meaning-preserving formatting changes can alter performance.
Freeze common semantics and predeclare small label-blind sensitivity probes instead of
choosing wording from final CV scores. Distinguish probability shifts, class flips and
request failures. These probes diagnose sensitivity, not prove semantic equivalence or
absence of all prompt injection.
[Webson and Pavlick, 2022](https://aclanthology.org/2022.naacl-main.167/),
[Sclar et al., 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html),
[Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Count each fully rendered request with the actual provider tokenizer. Input strings,
token counts and context windows are different quantities. Current policy rejects overflow;
it does not silently truncate, drop rows or adopt an intersection subset. Any later common
shortening, chunking or coverage restriction changes the estimand and needs an explicit
decision before outcomes. Decision chunking also needs a defined aggregation rule and may
lose cross-column interactions. Supported length is not proof of equally effective use of
every position. [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

The most informative figures consequently show text-column counts, nonempty cells, lengths
and tails, exact per-column/joint reuse, request and affected-row coverage, feature width,
probability/expected-score disagreement and paired predictive gains. Accuracy alone hides
calibration and individual disagreement; provider-average distances need shared row coverage
and aligned option identities. Missing inputs, failures and excluded datasets should remain
distinct. Plot declared no-data states until real features/results exist.

For efficiency, distinguish initialization, tokenization, GPU inference, API round trip,
cache hits and downstream fit/predict time. Count unique paid requests separately from row
mappings; reuse across folds is an amortization advantage, not repeated free inference.
A GPU or provider latency from a paper is not a throughput guarantee for this workload.
Cached artifacts provide reproducible reuse even when repeated uncached inference is not
deterministic. Operational timing and hardware evidence belong with the actual run.


The owner narrowed the executable scope to Experiments 1–3 on 29 October. Separate
question-ablation and representation-control experiments are removed; bounded 0A
robustness diagnostics remain operational checks. No lexical or presence-only control
is included. Improvements over the non-text arm can therefore reflect both text
availability and content; this design cannot attribute gains solely to semantics.
