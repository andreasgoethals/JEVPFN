# Changelog

## 20-10-2026

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

- Renamed the DM Library submodule to `dm-library`, preserving its exact library revision and updating references and tooling exclusions.

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
