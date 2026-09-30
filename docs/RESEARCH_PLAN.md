# Research sequence and experiment numbering

Updated 30-09-2026 to the owner's latest decision: design choices and local request previews,
small local Jev tests, full local Jev feature creation, then experiment 0 on VSC and the predictive
experiments. Full feature creation no longer waits for experiment 0. API calls remain disabled
while the design is reviewed; choosing the local computer does not start a paid run.
All experiment configs are disabled. No final metrics, split protocol or sample sizes are selected.

In plain language: a **feature** is an input column; **non-text columns** are the numbers,
categories and other inputs that are not designated text. The target is separate. **Per-column**
means reading each text column separately; **joint** means reading all text columns together.
The next step is to inspect a few actual rows, the exact questions and proposed new numeric
columns before sending anything to Jev.

## Exploration: before experiment 0

Configuration: `config/exploration/`. Notebooks 01 and 02.

1. Reproduce the official 20-dataset collection and freeze hashes, targets and text-column lists.
2. Inspect dataset/column audits, task semantics, missingness and request lengths.
3. Review fixed question wording, class ordering and the nine regression descriptions.
4. Verify the current Jev model/API documentation, limits and price. Calibrate token estimates
   against actual usage only after a paid pilot is separately approved.
5. Agree how to handle long-text overflow. Missing text is settled: skip empty input and store NaN. No implicit truncation, summarisation,
   generated questions, target-based feature selection or label-informed thresholds.
6. Check the local environment and how completed Jev outputs will be saved, backed up and reused.

The local code checks are development verification. They do not count as experiment 0.

## Semantic feature preparation: an independent reusable-data phase

Configuration: `config/feature_creation/`. Notebook 03. Durable output: `data/jev_cache/`.

Call this **semantic feature preparation**, without giving it an experiment number. It produces
inputs consumed by every experiment, so its artifacts must not live under `experiment_1/`.

All feature creation happens locally, before experiment 0:

1. Complete this review. A VSC allocation is not required before a local pilot.
2. After wording, context policy, data-use scope and spending are approved, implement the live
   transport with a single writer, reservations, bounded retries, usage accounting and restart
   reconciliation. The current repository contains no transport.
3. Generate a small local pilot across all three task types, single/multiple text fields, missing
   values and long inputs. The illustrative first/middle/last sample gives 315 nonempty request slots and 303 distinct
   requests across both text-only modes; final pilot row selection remains open.
4. Test the small sample locally: inspect the numeric outputs, check the saved results match
   the right rows, stop/restart the process, and confirm completed requests are reused.
5. Once the small local test passes and the design/budget are agreed, create all selected Jev
   features locally. Record spending and progress and back up the saved responses.
6. Check the complete feature tables locally, then transfer them to VSC for experiment 0.

The local tests catch errors before the full extraction. Experiment 0 subsequently checks that
the completed features and predictive pipeline work on VSC. No bulk extraction code is enabled.

Each dataset has two text-only tables: per-column and joint. Using both means joining those
existing tables; non-text features never enter Jev. Lossless
raw responses and probability vectors are retained; the main model's selected output view is
still a research choice. Changing output views needs no new calls. Changing state, question,
model or text-processing policy produces different requests and may cost money.

## Experiment 0: debugging on VSC

Configuration: `config/experiment_0/debug.yaml`. No allocation has been submitted.

**Infrastructure check, after full local feature creation:** confirm the credit account,
Python/packages, storage writeability/quotas and CUDA availability. VSC does not need Jev API
access for this workflow. The prepared CPU/GPU Slurm scripts make no inference POST and load no model weights. A passing
preflight is deliberately not reported as full experiment-0 acceptance.

**Using the complete locally generated features:** validate all task-specific response types, complete vectors, finite values,
class/score mapping, label-free inputs, source hashes, row IDs, cache deduplication and restart
behaviour. Inspect missing and neutral outputs and check that each feature cell traces to a
request/response. The current code/tests provide these deterministic building blocks; a full
pilot runner and acceptance thresholds remain to be agreed.

**Later, with explicit model-run approval:** prepare the pinned TabPFN-3.5 weights and perform
a tiny end-to-end smoke run on VSC. Select hardware from measured memory/time. This checks
categorical handling, feature assembly and runtime, not benchmark conclusions.

**Before experiment 1:** validate all expected rows/feature tables and checksums again, then
freeze artifact IDs for experiment 1. No missing block is silently replaced with zeros or mocks.

## Experiment 1: the main experiment

Configuration: `config/experiment_1/main.yaml`; candidates only, disabled.

- Fix the evaluation protocol, sample settings, seeds and metrics before comparisons.
- Review four kinds of comparison: non-text only; non-text plus TabPFN's native text
  processing; non-text plus Jev; and native text plus Jev. The last comparison tests whether
  Jev adds information beyond the model's existing text handling. Per-column, joint and combined
  views reuse the same tables and paired splits. These arms remain proposals, not a fixed protocol.
- Choose the downstream output view before reading benchmark results.
- Pin TabPFN-3.5, preprocessing and inference configuration. Fit any downstream transformations
  within the appropriate training fold; the label-free semantic feature cache is fold-independent.
- Use local TabPFN-3.5 as the proposed reproducible core; consider hosted 3.5-Plus/Thinking only
  as separately budgeted comparisons. Current free API allowances include Thinking, but the
  complete benchmark's fit/prediction compute budget has not been estimated or approved.
- Save predictions, resource usage, seeds, split identities and exact source/feature/model versions.
- Review complete results before drawing conclusions or moving to ablations.

The final evaluation protocol is not implemented. Splits, metrics, small-N values, repetitions,
statistical tests and treatment of dataset-specific prediction-time proxies remain open.

## Experiments 2 and 3: proposed ablations

| Experiment | Question | Reuse |
|---|---|---|
| 2: representation | Does reading each column, all text together, or both work best? | Compare the same text-only cached tables with identical non-text features and splits. |
| 3: output view | Does a full probability vector or uncertainty help compared with a scalar summary? | Derive views from the same raw responses/probabilities; no new Jev calls. |

These are proposals, not fixed final experiments. Further possibilities include per-field
contributions and sample-size sensitivity, if the protocol warrants them. Additional baselines,
generated-semantic-question experiments, extended MulTaBench datasets, significance tests and
paper framing remain unapproved and outside the current implementation.

## Decisions before the first paid request

1. Final wording, binary class orientation, class semantics and what relative regression direction
   means without a numeric label-derived reference.
2. Text fields allowed to be transmitted and the context overflow/chunking/truncation policy.
   Non-text columns are excluded; empty inputs are skipped with missing features. Preserve official text definitions unless a change is explicit.
3. Pinned Jev model, actual live API validation, probability/confidence interpretation and any
   account-specific behaviour. Current documentation is a reference, not a tested service contract.
4. Pilot scope, budget cap, token calibration, concurrency, current account rate limits and retry
   policy. Determine how to reconcile an accepted request whose response was lost before saving.
5. Local credential placement and backup/restore of saved responses. The extraction host is
   decided: the local computer. VSC Jev connectivity is not a prerequisite.
   No key belongs in YAML, Git, logs, command-line arguments or Slurm scripts.
6. Which cached views the main comparison uses and experiment-0 acceptance criteria. Final
   TabPFN/evaluation choices must be settled before modelling, but need not change lossless storage.

The [30 September review](REVIEW_2026_09_30.md) records the model interfaces, source references,
dataset feature counts and a refined offline budget. Nothing in that review enables live execution.
