# Changelog

What changed in this repository. **One chapter per date, `DD-MM-YYYY`, newest at the
top.** Terse: what changed, and why if it is not obvious. All rules and rule changes are
recorded here.

Edits to the repository. The runs and the dead ends go in
[`AGENTS_MEMORY.md`](AGENTS_MEMORY.md) — keep them separate, mixing them makes both unreadable.

Conventions:

- **As short as possible.** What changed, plus the *why* only when it is not obvious from the
  *what* — after an em dash. Most changes fit on one line; a genuinely large one can take more, but
  it should still be the shortest version that is complete. The detail belongs in the commit, and a
  changelog nobody skims is a changelog nobody reads.
- **Newest date first.** A reader wants the current state, not the archaeology.
- Dates are `DD-MM-YYYY`. Not ISO, not `Jan 5`. One format, sortable by eye.
- Group a date's bullets under `### Added` / `### Changed` / `### Removed` / `### Fixed` only once
  there are enough to need it.

---

## 30-09-2026

- Created the public JEVPFN repository and connected origin; preserved the template remote and history.
- Restricted Jev to text-only per-column/joint requests; skip empty inputs, preserve row/group identity, and store missing numeric features.
- Recomputed the actual-data budget: 2,032,351 distinct requests, about USD 35.84 before pilot calibration and long-text policy changes.
- Moved generated output into phase-specific output_JEVPFN folders; retained large results on project storage and small reports/logs on personal DATA, with no silent storage fallback.
- Added complete printed/saved notebook reports, plotted values, per-figure captions and persistent summaries across parallel and partial reruns; generated reports remain gitignored.
- Updated configurations and design documentation to match the owner's decisions; renamed the proposed context ablation to representation comparison.
- Verified 113 tests, all 20 offline datasets, four parallel notebook runs and all four notebooks in the actual JEVPFN kernel; raw hashes unchanged.

- Recorded the owner's corrected order: design/local previews, small local Jev tests, full local features, then experiment 0 on VSC. Removed the VSC prerequisite and mandatory Jev network probes.
- Retired the cleanup command after reported dataset loss, removed its launch instructions and restored all 20 pinned releases.
- Added reproducible offline duplicate-aware API-body budgets, seven verification tests and the dated feature/TabPFN review; recorded the agreed pilot-before-VSC sequence.
- Verified 102 tests, 20 offline datasets, four runner notebooks and all three research notebooks in the actual JEVPFN kernel.
- Made notebook dependencies self-contained and pinned future Torch to 2.12.1; refreshed the local editable install.
- Added immediate JEVPFN launch instructions; the empty-folder cleanup supplied at that time is now retired after reported data loss.
- Removed 60 hash-identical legacy raw files; retained numbered originals and reproducibility source snapshots.
- Documented proposed Jev feature dimensions and refreshed the official pricing/rate-limit references.
- Kept future TabPFN weights on durable VSC project storage via the Slurm activator.

## 29-09-2026

- Moved all dataset files out of `v1`; retained version/hash checks and source scripts under `src/data`.
- Removed duplicate raw sources, `.env.example` and the mock SQLite file; notebook cache demonstrations now use temporary storage.
- Documented the 23 empty ReadOnly directories whose removal automatic approval blocked.
- Organised exploration, feature preparation and experiments 0–3 into separate config folders; added the feature-preparation notebook.
- Flattened all 20 raw datasets into numbered folders and moved data Python/reference snapshots into `src/data` without changing hashes.
- Synced the read-only TFM Library to `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba` and repaired its submodule URL.
- Added documented Jev body/response mapping and reusable Parquet feature artifacts; kept live calls and models disabled.
- Recorded current Jev pricing, full-audit cost/context screens and the pilot/debug/full-build plan.
- Prepared CreditPFN-based VSC preflights, optional Jev/TabPFN dependencies and GitHub/environment instructions.
- Installed approved notebook/Parquet dependencies locally and named the environment prompt JEVPFN; no model packages or weights downloaded.
- Removed the inherited initializer files at the owner's request; automatic approval blocked deletion of their empty directory.
- Verified 102 tests, four notebooks, a real JEVPFN kernel and all 20 offline datasets after reorganisation.
- Moved the repository into the inner `JEVPFN/` folder; preserved Git state and local files, and refreshed environment paths.
- Added pinned loading and full audits of the 20 core MulTaBench text datasets.
- Added deterministic label-free Jev requests, six workload scenarios and a resumable mock cache.
- Added two exploratory notebooks, research notes and tests; retained the original template files.
- Initialised project packaging and local Python 3.12 environment with owner approval; hardened secret ignores.
- Verified all 20 downloads, both new notebooks and the preserved example; no API calls or models ran.
- Serialised shared audit-cache access to fix Windows file replacement during parallel notebook runs.

## Template history (original date not filled in)

- Repository created from
  [andreasgoethals/0.-Template](https://github.com/andreasgoethals/0.-Template).
- `tfm-library/` added as a read-only submodule. Record the pin here whenever it moves,
  because a result depends on the literature it was checked against.
