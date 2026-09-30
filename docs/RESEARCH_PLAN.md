# Research plan and feature design

Updated 30-09-2026. **Exploration now.** No live Jev client, paid responses, generated features,
model training or VSC experiment has run. Experiment configurations are reviewable proposals,
not runnable benchmark implementations.

## Agreed scope and sequence

1. **Exploration:** inspect all 20 core text datasets, actual request examples, missing inputs,
   exact per-column/joint duplicates, length extremes, target meanings and extraction cost.
2. **Feature creation, locally:** agree design and budget; implement transport; run a small
   pilot across binary/multiclass/regression and edge cases; test restart/reuse; create, validate
   and back up the complete features. This phase has its own config and notebook.
3. **Experiment 0, on VSC:** verify environments, data/feature joins, model compatibility,
   five-fold splits, preprocessing, threshold selection, resource needs and restart behaviour.
4. **Predictive experiments:** compare combinations of the same cached features. No new Jev
   calls are needed when changing the downstream model or selecting already-retained outputs.

All Jev questions are deterministic templates. The main method needs no generative LLM to
invent questions. Jev receives task metadata and text, never non-text features. Empty inputs
are skipped and stored as missing numeric features. Joint inputs omit missing fields.
The extended MulTaBench collection and generated-question side experiment are outside scope.

## Proposed experiments

| Phase | Question | Comparisons |
|---|---|---|
| Experiment 0 | Does the complete pipeline work on VSC? | Tiny approved CPU/GPU checks across task types, models and feature combinations; no claims about accuracy. |
| Experiment 1 | Does Jev improve prediction across models? | Original non-text features alone, + per-column Jev, + joint Jev, + both. Same folds for every model/combination. |
| Experiment 2 | Does Jev add information beyond the original text? | TabPFN-3.5 with non-text + original text, then + per-column, + joint, + both Jev. Reuse relevant experiment-1 results. TabSTAR is an optional text-aware comparator. |
| Experiment 3 | Which numeric representation of Jev works best? | Expected regression direction versus nine probabilities; class probabilities with/without available confidence. Hold input mode and folds fixed; initially TabPFN-3.5 and CatBoost. |

With eight candidate engines, four feature combinations, 20 datasets and five folds,
experiment 1 has at most 3,200 outer comparisons before equivalent-view reuse. Inner
validation/tuning adds computation; unsupported cases and actual resources must be recorded.

Possible later experiment 4: training-size sensitivity using nested training subsets and the
same untouched test folds. Sample sizes are not chosen yet. Avoid selecting experiments or
model settings from test-fold performance. Shared fits should be referenced, not rerun under
a new experiment number. For one-text-column datasets, per-column, joint and combined Jev
views coincide: fit once and mark the equivalent comparisons explicitly.

## Prediction engines

Candidates and package references live in `config/experiment_1/models.yaml`.
Nothing imports these engines during exploration. The first five are owner-requested;
additional models are proposals subject to experiment-0 checks.

| Engine | Proposed role | Preparation needed |
|---|---|---|
| TabPFN-3.5 | Main local model | Explicit V3_5 selection, checkpoint hash, categorical/text handling, inference budget. |
| TabPFN-3 | Earlier foundation-model baseline | Explicit V3 selection, comparable inference budget. |
| CatBoost | Boosted-tree baseline | Training-only categorical preprocessing and fixed tuning budget. |
| Logistic / linear regression | Simple classification / regression baseline | Training-only imputation, one-hot encoding, scaling; agree regularization. |
| TabICL v2 | Additional foundation model | Pin classifier/regressor checkpoints and ensemble/context settings. |
| TabDPT 1.3 | Additional foundation model | FAISS/runtime checks, context budget and real-data pretraining overlap. |
| Mitra | Additional foundation model | Freeze exact release/checkpoints, class limits and isolated single-model AutoGluon settings. |
| TabFM | Additional foundation model | Pin source/backend, review weight terms and context sampling; not yet a pyproject dependency. |
| TabSTAR | Optional experiment-2 text comparator | Text encoder, task-specific fine-tuning budget and pretraining overlap. |

Primary references checked 30-09-2026: [TabPFN](https://github.com/PriorLabs/TabPFN),
[TabICL v2](https://github.com/soda-inria/tabicl),
[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference),
[Mitra integration](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html),
[TabFM](https://github.com/google-research/tabfm), [TabSTAR](https://github.com/alanarazi7/TabSTAR),
[CatBoost](https://catboost.ai/docs/) and [linear models](https://scikit-learn.org/stable/modules/linear_model.html).

TabICL v2 provides classifier and regressor checkpoints. TabDPT's current model family is 1.3;
its package reference is 1.3.1. The AutoGluon MITRA adapter must not be assumed to select a
Mitra-v2 checkpoint: that choice remains unresolved. TabFM defaults to sampled contexts of
100 rows and 500 features; a large-table claim requires reporting its actual context budget.
Its code and weights have different licences. No candidate is assumed to handle every one of
the 20 tasks, especially Spotify's 114 classes, without checking. Record failures/unsupported
cases and pretraining overlap explicitly. Do not silently shrink datasets or drop difficult tasks.

The local TabPFN-3.5 text path uses character n-gram TF-IDF and SVD in the
[versioned text implementation](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/src/tabpfn/preprocessing/text.py).
Explicit text dtype/configuration and fold-local fitting must be verified in experiment 0.
Hosted 3.5-Plus/Thinking is a separate service with a different compute budget; it is not part
of this current proposal. Both 3 and 3.5 have hosted Thinking selectors in the
[official documentation](https://docs.priorlabs.ai/capabilities/thinking-mode).

## Evaluation: agreed elements and remaining details

The owner requested **five-fold cross-validation** and **validation-only F1 threshold
optimization**. The configuration interprets this as five outer folds, one repeat; repeated
five-fold CV has not been requested. Persist original row IDs and a shared split manifest.

For each binary outer fold:

1. Hold out the outer test fold completely.
2. Split only the remaining training rows into fitting and validation data, or obtain inner
   out-of-fold predictions. The choice and fraction remain to be agreed.
3. Fit preprocessing and the model on fitting data. Select the probability threshold with the
   highest F1 on validation predictions, using an agreed positive class and tie rule.
4. Freeze that threshold and score the untouched outer test fold. Save probabilities,
   predictions, threshold, row IDs, timings and split/config/artifact hashes.

Whether to refit on all outer-training rows after threshold selection is still open; refitting
can change calibration. Do not tune on the test fold or on predictions from rows used to fit
the same estimator. See the [scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

For **multiclass**, propose argmax decisions and macro-F1; weighted-F1/per-class threshold
optimization are separate choices. For **regression**, F1 and binary thresholds do not apply;
agree RMSE/MAE/R² and aggregation before running. Never average classification and regression
scores directly. Statistical comparisons across datasets remain open.

Propose stratified classification folds and shuffled regression folds only where the dataset's
prediction task permits random row splitting. Inspect repeated entities, time and near/exact
duplicates first; group/time splits may be necessary. Exact text duplicates are an extraction
reuse opportunity, not by themselves evidence that a row split is valid. Keep model tuning,
calibration, category encoding, imputation, scaling and native text preprocessing within the
training portion. Jev features may be cached before CV because they do not use row labels.

## Feature inputs and numeric outputs

For a row with t available text columns, there are t per-column slots and one joint slot before
reuse. The joint slot uses all named available fields. The combined representation joins those
two saved tables; it is not a third extraction. One multiclass request returns the whole vector.

| Task | Documented Jev mechanism | Preserve per input | Proposed compact prediction input |
|---|---|---|---|
| Binary | Noul | Probability favouring a fixed class over the other | One probability |
| Multiclass | Choice | K class probabilities and returned confidence | K probabilities |
| Regression | Score | Nine probabilities, confidence and expected direction | One expected direction |

These are documentation-based mappings in `src/jev/provider.py`, not live-validated responses:
[Noul](https://docs.typesafe.ai/primitives/noul), [Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score). Noul has no separate confidence value.
The compact view remains a proposal; raw responses and full vectors are retained regardless.
With t text columns and d retained inputs per request, per-column adds t*d inputs, joint adds d,
and both add (t+1)*d before equivalent-view removal. For binary data with two text columns,
that means two separate probabilities and one joint probability: **three new features**.

Regression levels remain -4 to +4. They describe evidence for a lower/higher target, not target
units: +2 does not mean two euros. The scalar is `sum(level * probability)` over all nine levels.
Zero may represent neutrality or balanced opposing evidence, so the full vector matters.
The draft intermediate descriptions still need distinct meanings. Proposed wording is weak,
moderate, clear and strong evidence on each side of zero; this has not been silently substituted
into the frozen template. No target quantiles/means define the levels. The reference meaning of
"relatively lower/higher" and ambiguous target units need review before the pilot.

## Label-free reuse and storage

Allowed metadata: dataset/target names, task type, fixed class vocabulary and column names.
Forbidden inputs: row targets, label frequencies, target summaries, correlations, performance
or model errors. The builder rejects target/extra columns and includes only selected text values.
Binary orientation and class order currently use canonical JSON order, not label frequencies.

An identical **complete request** (task, named text, question, model/version and settings) reuses
the first cached response. The API need not be deterministic. Equal text in different columns
or tasks is not automatically an identical request. A joint input with just one available field
reuses that field's per-column request. Changing question/model/text-processing policy changes
identity and may incur new cost. No fuzzy deduplication is used.

```text
data/jev_cache/
  responses.sqlite3          future validated responses and row origins
  features/<artifact_id>/
    features.parquet         stable csv:N row IDs + numeric features, no target
    requests.parquet         row/group → request key, or explicit missing_text status
    manifest.json            source/schema/config/model/code hashes and completion marker
```

Two tables per dataset give **40 planned table views**, not 40 numeric features. Names identify
the original text column or the joint input; manifests preserve all mappings. Prediction
matrices join original non-text columns and selected features by row ID; targets are separate.
No persistent demo database exists. Skipped empty inputs become NaN; missing responses for
nonempty inputs are errors. Linear models will need training-only imputation later, while
models supporting missing numeric values can use them directly.

Completed-response caching and immutable, hash-checked Parquet assembly are implemented.
Live transport still needs spending controls, concurrency ownership, retry limits and recovery
when a server accepts a request but the client loses its response. Exactly-once billing across
that interruption window is not guaranteed. Test restart/reuse and back up paid outputs.

## Extraction budget

The dated [Jev model reference](https://docs.typesafe.ai/models), checked 30-09-2026, is
`jev-1.13.0`, USD **0.042 per million input tokens**, free output. Configuration and
`JEV_INPUT_PRICE_PER_MILLION` control the assumption; verify it before paid work.
At four characters/token, with actual request bodies and exact reuse:

| Scope | Nonempty slots | Distinct requests | Approx. input tokens | USD |
|---|---:|---:|---:|---:|
| Per-column alone | 3,164,764 | 1,401,423 | 500,299,639 | 21.01 |
| Joint alone | 820,403 | 738,163 | 371,359,956 | 15.60 |
| Both with shared cache | 3,985,167 | 2,032,351 | 853,391,404 | 35.84 |

The final row includes both modes. Sensitivity to 3–5 characters/token, 0–100 extra tokens per
unique request and 0–10% billed retries yields USD 28.68–61.94. This is a scenario range, not
a spending cap or quote. Over-limit inputs remain in the estimate until a length policy is
agreed. A first/middle/last-row preview has 303 distinct requests; pilot sampling is not fixed.
Machine-readable counts and sensitivity tables are generated by `python -m src.jev.review`.

## Decisions before enabling work

Before the local pilot: settle target meanings/class orientation; regression wording; long-text
policy; data-use scope; Jev model pin/live response contract; pilot selection; spending cap;
retry/recovery/backup behaviour. Missing-text policy, text-only inputs and local execution are settled.

Before full extraction: inspect pilot outputs and usage without optimizing against labels;
validate probabilities, row mappings, deduplication, restart behaviour and the final cost envelope.

Before prediction: approve experiment arms/output views, model/checkpoint roster, data splits and
entity/time rules, inner validation, positive class/ties/refitting, multiclass/regression metrics,
model compute/tuning budgets, small-N settings, failure handling and statistical tests.
Final venue/paper framing and optional future side experiments remain open. The existing
[Prior Labs Jev comparison](https://docs.priorlabs.ai/cookbook/tabpfn-vs-jev) is related work;
its labelled standalone prediction setup differs from this label-free feature construction.
