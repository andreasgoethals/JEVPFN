# Research and design notes

Initial agreement: 29-09-2026. This document records scope, not a final experimental protocol.

## Currently agreed

- Start with the 20 core MulTaBench TEXT datasets, excluding the 20 extended text datasets.
- Later use TabPFN-3.5 as the main downstream model.
- Jev semantic augmentation is the central method. Mechanically extracted metadata fills fixed
  questions; the main pipeline requires no generative LLM question authoring.
- Plan per-text-column, joint-text and concatenated per-column-plus-joint representations.
- Jev receives only text and task metadata. Non-text features go to TabPFN; no context ablation is planned.
- Skip empty inputs and store missing numeric features. Joint requests use available text fields.
- Binary output is intended as a probability for one mechanically selected class versus the other.
- Multiclass choices are the task class vocabulary; preserve the possibility of retaining its
  complete probability vector. Storage now retains it; the downstream input view remains open.
- Regression uses the nine ordered directional levels -4 through +4. Zero means no meaningful
  directional evidence. No observed target range, quantile or average defines these levels.
- Extract once where possible; reuse exact responses across rows/folds through content keys.
- Remain in the initial, dry-run-only stage. No Jev calls, modelling or benchmark execution.
- The owner's latest sequence is: design choices and local request previews -> small local
  Jev test -> full local Jev feature creation and validation -> experiment 0 on VSC ->
  predictive experiments using the saved features. This supersedes the earlier VSC-first plan.

## Label-free boundary

`TaskMetadata` contains dataset/target names, task type, distinct class labels and feature names.
Classification vocabulary is mechanically extracted from the full released task and fixed before
future folding. This uses the permitted class vocabulary only, never its frequencies. Regression
metadata carries no observed target values or summary statistics.

The request builder requires exactly the feature columns and refuses the target or extra columns.
It reads one feature row. Audit distributions, target quantiles, correlations, model performance
and errors have no input slot. Tests verify invariance to changed row labels and class frequencies.
This protects our construction; it does not establish that every upstream feature is appropriate
at a future prediction time. Existing task proxies require methodological review.

Binary orientation is the final class in canonical JSON order. Multiclass order is the same stable
canonical ordering, not frequency order or an inferred ordinal order. The frozen schema and data
hash identify the vocabulary. Review this convention before extracting features.

## Intended requests and cache

`src/jev/templates.py` centralises wording. `src/jev/requests.py` emits a provider-neutral JSON
schema explicitly marked `api_mapping_verified: false`. It preserves nonempty text and Unicode exactly. Null or whitespace-only text groups are skipped,
with missing numeric features and an explicit trace status. Joint inputs omit missing fields.
No truncation or imputation is applied. The three representation modes are construction options.
`src/jev/provider.py` provides a documentation-verified API-body preview and response decoder;
the request schema's false flag means it remains an intended request without live validation.
Pure cached-feature assembly is available; no real features have been generated.

Content keys hash the exact task/state/question plus model, version, template version and request
configuration. They exclude row IDs, timestamps and incidental execution provenance so identical
content reuses a response. Changing any semantic input/version changes the key. Separate origins
retain dataset, source hashes, positional CSV row ID, text columns, mode/slot, code/config hashes,
and timestamps. Positional IDs refer to the full pinned CSV, never a reindexed fold.

SQLite transactions prevent half-written responses. Existing different responses cannot overwrite
one key. Mock records are isolated from future verified-response records; reopening the cache
resumes completed work. `data/jev_cache/` survives the template cleanup utilities. Full raw
responses can be stored. The decoder follows the documented Noul/Choice/Score schema, preserves
complete probability vectors and requires a matching pinned model. Parquet feature artifacts
carry source/schema hashes and row/request mappings; mocks cannot become experiment features.

No network implementation exists. Provider-specific authentication, retry semantics, concurrent
in-flight reservation, recovery after a server accepted a call but before local persistence, and
any provider idempotency mechanism must be designed after the actual API is verified. The present
completed-response cache does not claim exactly-once billing across that interruption window.

## Still open — do not resolve automatically

- Exact live Jev implementation, authentication, retry behaviour and validation of the published
  API contract. `jev-1.13.0` is a dated documentation reference; approve its use and check actual
  response version behaviour during the pilot.
- Exact feature view supplied to TabPFN: scalar summaries, uncertainty and/or full probability
  vectors; regression reduction and interpretation of neutral answers. Lossless
  cache retention does not select the main model's input view.
- Exact TabPFN-3.5 interface/configuration and representation of categorical inputs.
- Final evaluation metrics, protocol, cross-validation/splits and leakage controls.
- Small-N settings and additional baseline models.
- Generated-semantic-question side experiment, outside the main method.
- Whether to add the 20 extended MulTaBench text datasets later.
- Statistical tests and final venue/paper framing.
- Intermediate regression descriptions currently differ mainly by numeric markers. Review distinct
  evidence-strength wording; see FEATURE_CREATION.md.
- Review of deterministic wording, binary orientation and class order; ambiguous target units,
  ordinal class labels, transformed regression targets and what “relatively” means.
- Treatment of upstream numeric-looking/date-like strings identified as text, identity fields,
  related measurements, and task/target proxies already present in the published features.
- Handling of the documented context limits, text truncation/chunking policy, actual tokenizer,
  data-use terms for sending the selected fields, and revalidation of price/rate limits before
  paid extraction. The current documented input price is recorded as a configurable assumption.
- Paid cache operations: atomic in-flight ownership, retries, crash reconciliation and backups.

These remain questions for the owner. The initial implementation does not select answers.

## Phase organisation

Exploration and semantic feature preparation have separate configuration folders/notebooks.
Experiment 0 is debugging on VSC; experiment 1 is the main comparison; experiments 2 and 3 are
proposed comparisons of inputs and outputs. Design checks, a small Jev test and the full feature
build all happen locally before experiment 0. See [the detailed plan](RESEARCH_PLAN.md),
[feature tables/budget](FEATURE_CREATION.md) and [VSC setup](VSC.md).

The [current review](REVIEW_2026_09_30.md) recommends a native-text TabPFN baseline and considers
Thinking as an optional hosted comparison. These are proposals for the final protocol.
