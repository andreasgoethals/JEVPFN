# Changelog

## 30-09-2026

- Verified 136 tests, dependency consistency and all four notebooks through the no-argument parallel runner (16 figures, about 65 seconds).
- Standardized uppercase documentation filenames and made notebook concurrency automatic, respecting CPU affinity and Slurm allocations.
- Excluded AutoGluon and deferred Mitra-v2 pending a verified independent runtime.
- Added offline binary/multiclass/regression metrics, raw-prediction records and separate timing stages; expanded evaluation references.
- Made Git publication owner-only; removed the temporary research plan and consolidated durable model/API findings in LITERATURE_REVIEW.md.
- Grouped notebooks by phase; added the 20-dataset visual overview, count/percentage reuse plots and feature-design charts; standardized All Results.md.
- Added Windows sharing-error retries and recovery copies for reports/audits, plus a single report-lock file; verified 124 tests.
- Expanded the disabled model catalog with LimiX-2, Causilo, Mitra-v2, EXAONE Tabular and text comparators; documented text support and incompatible runtimes.

- Verified 124 tests, 20 offline datasets, eight source hashes, four parallel notebooks with 16 figures, and all three research notebooks in the actual JEVPFN kernel.

- Centralized figures under output_JEVPFN/figures/<phase>/<notebook>; capitalized All Results.md and Captions.md.
- Added exact per-column/joint repetition, empty-input and overlap statistics to the exploration audit and notebook.
- Prepared disabled five-fold CV and validation-only binary F1 threshold configuration, a model catalog and three distinct experiment proposals.
- Kept TabPFN, Hugging Face and Torch weight caches on project storage; runtime/compiler caches stay disposable.
- Declared optional baseline/TFM dependencies; no packages, weights or models installed or run for this refactor.
- Consolidated eleven documents into five current guides/records; removed duplicate status reports, template manual and standalone GitHub guide.
- Created the public GitHub repository; preserved template Git history and remote.
- Restricted Jev to text-only per-column/joint inputs; skip empty inputs and reuse identical requests across rows/modes.
- Added full printed notebook reports, captions, saved plot values and parallel/partial-run aggregation.
- Restored all 20 pinned datasets after reported cleanup loss; disabled that cleanup command.
- Confirmed the sequence: exploration, local pilot/full features, experiment 0 on VSC, predictive experiments.

## 29-09-2026

- Initialized JEVPFN inside the inner repository folder, retaining the template's Python/YAML/style/test conventions.
- Prepared Python 3.12 and the JEVPFN notebook kernel; no large models downloaded.
- Added pinned loading/audits, deterministic label-free requests, mock caching and immutable feature-table preparation.
- Flattened raw data into 20 numbered folders; preserved upstream source snapshots under src/data/upstream.
- Synced the read-only TFM Library at 81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba.
- Added three research notebooks, phase configs and CreditPFN-based VSC preflight scripts.
- Fixed parallel Windows audit-cache races with OS-backed locking.

Repository origin: [andreasgoethals/0.-Template](https://github.com/andreasgoethals/0.-Template).
Earlier detailed edit records remain in Git history.
