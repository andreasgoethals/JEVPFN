# Changelog

## 30-09-2026

- Verified 118 tests, all 20 offline datasets, all four parallel notebooks, and the changed exploration/feature notebooks in the actual JEVPFN kernel; reuse counts match the independent API-body budget.

- Centralized figures under output_JEVPFN/figures/<phase>/<notebook>; capitalized Allresults.md and Captions.md.
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
