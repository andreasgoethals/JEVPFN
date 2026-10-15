# Changelog

## 15-10-2026

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
