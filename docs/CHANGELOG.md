# Changelog

## 22-10-2026

- Excluded A100 by owner decision: H100 defaults for every GPU job and generated plan,
  removal of the A100 profile, rejection of legacy overrides before model loading, and
  updated operational guidance. CPU jobs and protected inputs/checkpoints are unchanged.
- Hardware-policy validation: 771 tests passed/two environment skips; lint, dependencies,
  23 shell syntax/LF checks and frozen-tokenizer preservation passed. No VSC execution.
- Timing validation: 755 tests passed/two environment-specific skips; final targeted checks
  73 passed/one skip after exception-flush refinements. Lint and dependencies passed.
- Added compact extraction timing journals for Jev calls/rate waiting and open-weight/
  embedding batches, with cache reuse separated, CUDA synchronization and safe transfer.
  Kept paid response persistence ahead of timing writes and buffered filesystem commits.
- Verified broad binary/multiclass/regression metric retention and prediction round trips;
  aligned the configured predictor timing stages with the implemented measurements.
- Local validation: 747 tests passed, two platform/runtime skips; lint, dependency and
  whitespace checks passed. GPU execution remains subject to actual VSC acceptance.
- Verified the latest VSC archive: three encoder failures reproduce the tokenizer issue
  already fixed locally; five decision contracts are pending for scheduler priority.
  Clarified current setup status and safe queue diagnosis without rebuilding weights.
- Recorded direct tabular decision-model prior art and distinguished reusable feature
  augmentation from direct decision benchmarks; clarified MulTaBench's core/extended scope.

## 21-10-2026

- Validated the latest review fixes: 747 tests passed/two environment-specific skips;
  all five notebooks/25 figures, dependency/lint/shell checks, and the three actual pinned
  encoder tokenizer preparation/Unicode-coverage checks passed without loading weights.

- Fixed TabSTAR cross-experiment pairing by excluding output routing from predictor identity;
  retained exact output phases and all model/checkpoint settings. Added campaign regression tests.
- Added presence-matched embedding and exploratory-combination comparisons, retaining the
  plain non-text comparison. Added regression coverage across two encoder recipes.
- Replaced removed Transformers 5 `prepare_for_model` with backend special-token processing;
  verified pinned E5/Qwen tokenizers locally without weights, including full chunk coverage.
  Every actual encoder now checks publisher token-ID parity before loading its model.
- Matched the full embedding job's 56 GB allocation with eight requested CPU cores.
  Recorded three successful container builds and the still-pending GPU setup evidence.

- Verified the current grid with 743 passing tests (two environment-specific skips), all
  five executed notebooks/25 figures, lint, dependencies and 23 shell syntax checks.

- Reduced primary Experiments 2/3 to embeddings-only/native-text-only; moved winner-selected
  combinations into explicitly exploratory Experiment 5 with checked selection provenance.
- Set per-column PCA32 and joint PCA256/128/64/32/16/8; prepare exact fold-local PCA on CPU
  once and reuse across predictors. Added dataset width table/figure and cached-PCA tests.
- Added concurrent prerequisite-aware 0A submission, regular A100/H100/B200 routing guards,
  single-GPU benchmark arrays, CPU-only predictor tests and compressed GPU telemetry.
  Fixed bare container aliases and parallel diagnostic cache contention; kept native-storage guards.
- Distinguished measured account limits, estimated tokens and conservative dispatcher timing
  in the throughput report; no automatic API limit/budget increase or additional paid calls.

- Restructured Experiments 1/2/3 into decision features, frozen encoders and native text,
  including all six extractors and three modes without test-score-based selection.
  Added pinned E5-large/Qwen encoders, isolated transferable caches, PCA64/32/16/8 and
  train-only joint PCA. Added explicit multi-extractor concatenation and width checks,
  including Kumo's category-count expansion before its 500-column selection.
- Expanded 0A to all three real encoders and 0B to the new PCA/native-text combinations;
  added a bounded-report Experiment 4 analysis and moved remaining controls to Experiment 5.
  Updated matched result comparisons, report identities, notebook figures and VSC commands.
  TabSTAR diagnostic training output now follows Experiment 0 instead of Experiment 3.

- Added the approved PCA8/PCA16 E5-only ablations across all six predictors and expanded
  0B fixtures. Added AG-MM as a third standard native-text reference with a fixed recipe,
  offline hashed encoder assets, no HPO/ensemble and a separate dependency runtime.
- Fixed Jev's observed 0.99 probability-vector sum with bounded, Jev-only two-decimal
  normalization. Preserve raw responses and spending; previously rejected saved responses
  recover without redispatch. Added malformed-response and paid-cache recovery tests.
- Liquid retry 62378847 still escaped the image despite the pre-install override. Creation
  and activation now use absolute image prefixes in a checked private wrapper template;
  the site installation is untouched. Added a host-env-name collision regression test.
- Refreshed the feature-creation notebook and model/method/storage documentation. Added
  guarded instructions for removing the accidental DATA env1, preserving the CPU environment.
- Final local regression suite: 724 passed, two skipped (local PyTorch and POSIX links).
  All five notebooks executed with zero error cells and 24 figures; lint, pip check,
  18 shell syntax checks and the three pinned-library checks passed. Paid dataset pilot
  completed all 18 datasets / 288 requests / 238,111 reported input tokens, costing
  USD 0.010000662 in total. Full extraction and model training stay disabled.

- Fixed pilot numeric input (`$0.042`, USD notation, grouped thousands), recorded owner-confirmed
  80 RPS / 100k TPS, and replaced the need for an unpublished byte/token bound with documented
  server context rejection plus conservative billing reservations. Pilot pacing remains slow;
  full extraction stays disabled.
- Added optional PCA16 E5-only arms for all six predictors, paired with main PCA32 results.
  Added complete E5-cache validation to 0B and actual predictor fixture paths for PCA16/32
  and combined embeddings/decision features. No new extraction or model training was run.
- Diagnosed Liquid build 62378491: Conda installed into host DATA `conda_envs/env1`, leaving
  no image Python. Added an image-local environment override, a pre-pip interpreter check,
  and guarded cleanup of one unsealed image. No environment or output was deleted locally.
- Rechecked all 18 saved question stores and recorded current token/Clef/Intern evidence.
  Input questions and both fingerprinted tokenizer modules are unchanged. No paid request,
  remote job, submodule update, commit or push was performed.
- Local validation: 698 tests passed, two skipped (PyTorch unavailable locally and a
  POSIX-only link test). Lint, dependency consistency, 18 shell syntax checks and all
  three pinned-library checks passed. The Liquid repair still requires a VSC build/test.

## 20-10-2026

- Reproduced the Jev robustness decoder failure from its saved response (score 7.99
  versus probability expectation 8.00). Added a bounded two-decimal compatibility check,
  discrepancy records and credential-free/network-free revalidation. Revalidated all
  15 actual saved responses without another API call.
- Fixed aliased-path fold completion and isolated failed open-weight batches in diagnostic
  continue mode. Added all-predictor presence-only controls and matched analysis, real-text
  sensitivity diagnostics with option alignment, and per-fit PCA explained-variance metadata.
- Added a read-only committed-submodule/pin check. Corrected the DM rename evidence:
  owner commit 6a12e8f omitted the new gitlink; publication remains the owner's task.
  Token job 62372655 is PENDING in the supplied 16:03 accounting. Local tests: 667 passed,
  two platform/runtime skips; all five notebooks completed with 24 figures, and lint passed. No remote
  submission, environment deletion, new paid request, commit or push occurred in this repair.

- Implemented the review follow-up: per-dataset/phase SQLite evaluation stores containing
  checksummed Parquet predictions and immutable fold records, bounded lock-file growth,
  transactional portable merging and legacy reading. Added corruption, concurrency,
  resume and import-conflict tests.
- Bound validation evidence to normalized source/configuration, prepared inputs and runtime
  pins; exact 0A campaign selection resolves multiple successful job reports without choosing
  the newest implicitly. Added prompt sensitivity/repetition diagnostics, reviewed acceptance
  and independent durable Jev diagnostic spending records.
- Unified text-missing indicators across representation arms, derived provider-neutral
  uncertainty summaries, batched E5 across texts, and added bounded Clef/Perplexity batches
  with serial-probability parity checks. Main questions and protected tokenizer files remain unchanged.
- Added runtime inode inventory, guarded retirement of replaced partial environments,
  predictor-container wrapping support and build-node index preflight. Updated research
  interpretation, storage and VSC instructions. Local suite: 660 passed, two platform/runtime
  skips; five notebooks passed with 24 figures. Actual GPU/batch/throughput validation and
  remote retirement remain pending. No paid inference, remote submission, cleanup or Git publication.

- Updated DM references/tooling to `dm-library`; the owner's 6a12e8f commit omitted the
  replacement gitlink, so fresh-clone installation remains incomplete until owner publication.

- Added DM Library as an independent read-only submodule for this project only; canonical maintenance and pin propagation run from `0. Literature`. No model settings changed.

- Added the independent read-only VSC Library submodule and daily pin-update proposals; shared HPC guidance moved out of TFM Library, with project instructions retained in docs/VSC.md. No cluster workload was submitted.

## 19-10-2026

- Diagnosed overlapping token jobs 62370453/62370457: duplicate writers collided on SQLite primary keys, although their combined reports cover all five tokenizers with zero overflows. Added coordinator/database writer locks and serialized report merging; matching saved counts remain reusable. Regression tests reproduce duplicate-run and lost-report-update gaps before the fix. No inputs, tokenizer fingerprints, containers or model settings changed. Local full suite: 648 passed, one POSIX-only skip and one Windows shell timeout; the timed-out test and all 36 shell tests passed on unchanged reruns. Token/GPU regression checks, lint and dependency checks passed; VSC retry is pending.

- Inspected the newer VSC ZIP: Clef container build 62369713 sealed successfully; GPU job 62370298 passed CUDA but stopped before model loading because the token-audit summary was absent. Added an actionable recovery error and read-only `gpu_contracts --check-inputs` CLI, shared with the execution gate. Added tests for missing-evidence refusal before model loading, dataset coverage, and rebuilding a lost summary from surviving exact counts without retokenization. Validation: 646 tests passed, one POSIX-only skip; lint, dependencies and 17 shell syntax/LF checks passed. Updated setup order and observed status; no model/container rebuild, cleanup, paid call, remote submission or Git publication performed.

- Diagnosed build 62369139's missing `mksquashfs`. Added a read-only prerequisite check, sourced before expensive build work, that resolves Apptainer's documented bundled packer and refuses missing/broken tools. The owner confirmed bundled version 4.6.1 runs on the login node. Added success/failure coverage and updated runtime status/instructions. Validation: 642 local tests passed, one POSIX-only skip; lint, dependencies and all 17 shell syntax/LF checks passed. Original environments remain intact; real VSC container build/GPU validation remains pending. No paid call, remote submission, cleanup or Git publication performed.

- Inspected downloaded container log 62368266 and fixed its absent private wrapper-alias failure; real VSC verification remains pending. Aligned notebook output with the supplied template: PDF-only figure folders, separate logs/manifests/results, one captions document and A4 panels. Added selective DATA/PROJECT/local cleanup, permanent smoke-test spending/response preservation, transactional active-input regeneration and a resumable one-job-at-a-time 0A local/VSC coordinator. Tests cover deletion scope, caption invalidation, rebuild rollback, ledger preservation and disconnected/failed job sequencing. All five notebooks executed successfully; all 24 final PDFs were visually inspected. Validation: 634 local tests passed, one POSIX-only test remains for Linux; the exhaustive audit passed all 1,698,127 unique inputs and 3,294,340 row/slot links. Dataset preflight, dependency/lint checks and 16 shell syntax/LF checks passed. No new paid inference, remote job, full output wipe or Git publication occurred.

- Added automatic local credentials from the owner's Git-ignored secrets/typesafe.local.json, with environment override, fixed non-disclosing diagnostics and isolated synthetic tests. An explicitly recorded HTTP 401 can be retried once after correction, retaining its diagnostic and conservative reservation; unknown outcomes remain blocked. Authenticated model listing and three new live Jev 1.13 probes passed (1,051 input tokens; computed USD 0.000044142); saved responses were then reused without another client. No dataset extraction ran.
- Directly inspected failed VSC build 62366728: the site wrapper's archival copy could not preserve native permissions inside its image. Prepared a checked copy-template replacement in a private scratch copy of wrapper 0.4.2; shared software and native environments remain untouched. Real VSC build/GPU verification remains pending. Updated setup/readiness instructions. Validation: 614 local tests passed, one Linux link test deferred to VSC; lint, dependency checks and all 16 shell syntax/LF checks passed. Protected tokenizers/submodule are unchanged. No output cleanup, VSC submission or Git publication performed.

## 16-10-2026

- Diagnosed the fresh-run freeze mismatch as 52 CRLF/LF-only differences despite identical Git HEADs; source hashing now normalizes text line endings and still rejects CRLF shell scripts. Added credential-format diagnostics without exposing keys, and retry coverage for pre-request initialization failures. Container retries preserve nonempty/linked destinations, reuse only empty directories and keep build work on shared Lustre scratch. Recorded the failed 62346143 build without inventing a cause after its log was deleted. Prepared inputs/uploads and protected tokenizers are unchanged. Validation: 599 tests, lint, dependency checks and all 15 shell syntax/LF checks passed; no paid request, remote job or publication was performed.

- Added scratch-based hpc-container-wrapper builds, image/wrapper fingerprints, automatic verified runtime selection and guarded retirement after matching GPU evidence. Added frozen source/config/runtime receipts for the owner's final clean validation. Three actual local Jev 1.13 binary/multiclass/score probes passed: 1,051 input tokens, computed USD 0.000044142; no dataset extraction. Recorded successful v3 CPU/token jobs and shared inode usage. Validation: 585 tests, lint, dependency checks and 15 shell syntax/LF checks passed. Container execution on VSC still needs verification; no remote submission, cleanup or publication was performed.

- Expanded all six predictors to eight configurations: non-text, E5/PCA-32 per text column, three decision views and their E5 combinations. Shared baselines avoid duplicate fits; combined joins retain both artifact identities and analysis includes matched incremental comparisons. Added the notebook design matrix, a bounded offline E5 GPU contract job, and hidden interactive Jev authentication. Owner transcripts establish fresh CPU/storage success and PROJECT file-count exhaustion during Liquid setup; version-3 token counting remains active. Read-only Jev authentication succeeded without inference or key persistence. Validation: 567 tests, five executed notebooks/18 figures, lint, dependency consistency and 12 shell syntax/LF checks passed; the new design figure was visually checked. Tokenizer source files were preserved.

- Critically assessed the fifth review: added a pinned, resumable frozen E5 control for CatBoost/TabPFN with training-only PCA and validated transfers; added dataset-level paired summaries, guarded predeclared Wilcoxon/Holm analysis and separate native-text references. Clarified pretraining-overlap limits without treating perturbations as proof. Added guarded EUR 5 Jev pilot configuration/authentication checks, all-five-provider runtime setup and explicit matching-evidence acceptance. Fixed a fresh Perplexity setup check that depended on not-yet-installed packages. All 539 local tests, five notebooks/17 figures, the exhaustive v3 question audit, lint, dependencies and eleven shell syntax checks passed. The simulated CPU dependency profile passed 533 tests with three SDK skips/three notebook deselections. No real inference, paid requests, VSC jobs, deletion or publication performed.
- Froze owner-approved question version 3: grammatical regression target wording and quoted binary descriptions, with unchanged meanings/options/rubric. Regenerated all 18 local stores while preserving historical sets; the read-only exhaustive audit verified 1,698,127 unique inputs and 3,294,340 original row/text-slot links. Added that audit to the VSC CPU job and prepared versioned upload support. Previous token/GPU evidence is historical until repeated with these inputs.
- Verified the independent review's Jev recovery defect. Added reviewed, state-bound reconciliation for uncertain billing and exhausted unbilled retries; saved invalid responses can be decoded again without another paid call. Cumulative charges and mergeable recovery history are preserved. Added conditional seen/unseen-text metrics on unchanged outer-test predictions; no new fits, split changes or target information in extraction. Several review statistics did not reproduce, including an empty-text counting issue.
- All 519 local tests, five notebooks/17 figures, lint, dependency and modified-shell checks passed. Recorded the owner's successful staging of the remaining three open-weight checkpoints. Their environments/GPU tests and live Jev tests remain pending. No paid calls, model inference, VSC submissions, output deletion or Git publication performed.
- Recorded passing VSC storage job 62338536, including the measured 29.509-second receipt visibility delay and successful SQLite snapshot/restore. The owner's PROJECT inventory confirms Clef/Intern checkpoints and environments only; the other three runtime setups and live Jev verification remain pending. All 486 local tests, lint and dependency checks passed. Updated current readiness without enabling execution or deleting output.

## 15-10-2026

- Diagnosed storage job 62333801 from both worker logs and receipts: locking succeeded, but an immediate cross-node file lookup falsely reported a missing result and prevented snapshot validation. Added a bounded receipt-visibility wait without repeating database operations, with delay/timeout/error regression coverage. All 486 local tests, lint and dependency checks passed. Clarified the final clean-validation sequence; protected inputs/caches and existing diagnostic output remain untouched.
- Verified CPU job 62333525 passed and the combined PROJECT/DATA transfer imported 72 sampled responses into 18 dataset stores; the owner completed all five local notebooks. Storage job 62333541 failed while parsing worker stdout. Replaced that fragile result channel with atomic, job/probe-bound receipts and retained launcher diagnostics. All 481 local tests and lint passed; the storage check still requires a VSC rerun. Existing inputs, tokenizers and GPU runtimes are unchanged.
- Reviewed downloaded current-input VSC evidence: all five token audits and Clef/Intern's 74-request GPU contracts passed; the CPU suite needs a fixture-fix rerun and no two-node storage job is recorded. Added reviewed Jev spending-policy revisions with cumulative charges, unresolved-reservation protection, per-request policy identity and mergeable history. Fixed unrealistic OS exceptions in the two Linux cleanup tests; clarified stage-to-stage configuration changes. All 470 local tests, lint and dependency checks passed, including installed-SDK mocked success/429 responses. No paid calls, remote jobs or Git publication performed.
- Fixed storage setup's unnecessary write probe in existing raw datasets. Raw-directory reads and mutable-destination writes are now reported separately; CPU preflight still verifies actual files/hashes. All 456 local tests and lint passed, including new immutable-input, unreadable-input, blocked-output and missing-allocation cases. The owner's VSC cleanup and environment check had succeeded; resume setup after updating the code.
- Reproduced and fixed real Windows WinError 5 when deleting read-only output directories. Recovery stays inside validated output paths, retries transient Windows locks and preserves other permission errors. Completed local output deletion and verified all 80 protected data/input/cache files byte-for-byte. Isolated a retry test that had written synthetic logs into the real output folder. All 448 tests passed; the real output folder remained empty after the suite.
- Added explicit permanent output deletion, optionally including earlier output archives, for local/VSC clean runs. All tiers are checked for links/mounts before deletion; permanent inputs/features and model environments remain protected. The follow-up full suite passed 440 tests; actual local cleanup was previewed only.
- Verified all five concrete findings in the second independent review. Slurm jobs now receive provider/environment paths as arguments and propagate their prepared environment to workers; memory requests respect A100 limits and token-job core accounting.
- Added settings hashes to GPU-contract and extraction summaries; reordered reset, input installation and fresh checks in VSC instructions. GPU speed/batching suggestions remain unmeasured.
- Protected cleanup against linked roots/figure folders; remove rebuildable comparison CSVs/transcripts while retaining personal notes and permanent artifacts. Added a local option to the reversible full-output reset.
- Inspected all notebook figures; removed duplicate/empty panels, corrected long-text ordering/pagination, compacted diagrams and made 30-class/six-provider comparisons readable. Executed outputs remain in the notebooks.
- Local verification: 429 tests, all five notebooks/17 figures, 97 synthetic comparison renders, dataset preflight, wheel build, dependency/lint checks and eleven shell syntax/LF checks passed. No VSC jobs, inference or Git publication performed.

## 14-10-2026

- Added a reversible, idle-job-checked VSC output reset and a checksum-validated 18-dataset question upload bundle with backups and byte-exact installation. Added isolated Clef/Intern environment setup commands.
- Fixed publisher imports creating bytecode inside fingerprinted checkpoint trees; bounded archive retries tolerate temporary Windows sharing locks without replacing archives.
- Critically verified the independent review: fixed durable extraction failures, concurrent-provider scheduling, pure bulk cache reads, runtime-bound feature identities and protected-output cleanup.
- Added guarded local Jev execution/backoff, known-answer GPU probes, a two-node storage test, resumable CV folds, resource/warning records, executable control arms, predictor/full-fold checks and complete-arm aggregation.
- Versioned question inputs to exclude provenance URLs from model state; regenerated all 18 files while retaining historical sets. Fixed unseen-date encoding and marked stale token screens as historical; current inputs require a new VSC audit.
- Removed Microsoft from the active pipeline; aligned six-extractor previews/configs, frozen-question budgets, metric version and the eight-predictor catalog; consolidated docs, ignore rules and citation metadata.
- Final local verification: 410 tests, all five notebooks/21 figures, lint, dependencies and eleven shell files passed. Dataset preflight passed; credential-free HTTPS reachability checked. No live inference, Slurm submission or Git publication.
- Fixed VSC notebook-test scope/Jinja2; added bounded real GPU contracts, resumable tokenizer reports, safe dataset-cache merging and two-tier export/import.
- Verified the owner's successful CPU rerun; exact token and real GPU results remain separately tracked in AGENTS_MEMORY.

## 13-10-2026

- Added saved-feature comparison, source-checked predictor adapters, empty-input rejection and one-file-per-dataset frozen questions with exact reuse/row links.
- Numbered phases consistently and separated local notebook analysis from VSC extraction/evaluation.

## 12-10-2026

- Retained all 20 raw datasets; excluded Spotify and Zomato from extraction/evaluation; tightened feature-group/row integrity and refreshed decision-model literature.

## 01-10-2026

- Persisted notebook outputs/progress; expanded visual audits, long-text analysis, metrics/timings and fold-local evaluation.
- Standardized uppercase durable docs, All Results.md and figures/Captions.md; made Git publication owner-only.

## 30-09-2026

- Restored and verified raw data after cleanup loss; retired the unsafe cleanup entry point.
- Consolidated docs, restricted inputs to text-only per-column/joint requests and prepared cache/feature schemas, CV configuration and storage routing.

## 29-09-2026

- Initialized the inner JEVPFN repository, Python 3.12, pinned data/source loading, deterministic templates, thin notebooks and read-only TFM Library integration.
- Preserved the template's source/style/Git conventions and fixed parallel audit races.

Detailed historical edits and validation snapshots remain in Git history. Current run evidence
belongs in AGENTS_MEMORY; code/configuration and README define the present implementation.
