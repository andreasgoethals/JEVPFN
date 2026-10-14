# AGENTS.md — JEVPFN

Instructions for AI agents working in this repository.

## 0. Before you start

1. [`README.md`](README.md): current layout, setup and outputs.
2. [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md): verified research and model interfaces.
3. [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md): protected data and known operational failures.

The owner requested consolidating documentation; generic template instructions and historical
status reports were merged into these files. Preserve useful conventions without duplicating docs.

This project is initialised; the owner requested removal of the inherited `_template` files.
Current phase layout and setup instructions live in `README.md`, not in that directory.
The owner selected text-only per-column and joint Jev inputs, skipping empty inputs and storing
missing features. Hosted feature creation is local; self-hosted extraction needs a VSC GPU pilot before full runs. Generated full
reports contain raw examples and are gitignored under `output_JEVPFN/`. All notebooks run
locally and save executed outputs into their .ipynb files. Figure/report/config/run artifacts
are under local figures/<phase>/<notebook>, with global output_JEVPFN/All Results.md
and output_JEVPFN/figures/Captions.md.
Config, notebook and output phases use numbered directories: 0. exploration,
1. question_preparation, 2. feature_creation, then 3. experiment_0, 4. experiment_1, etc.
The general figures folder is unnumbered. Exploration outputs only figures/reports.
Audit/budget intermediates use run-scoped temporary storage. Prepared inputs are exactly one
`data/decision_inputs/<dataset_id>.sqlite3` file for each selected dataset. Each file retains
question versions, unique exact text groups and every original row/representation link.
Do not export question copies under output. `python -m src.jev.preparation` creates/reuses all
18 files locally; logs/manifests use output_JEVPFN/1. question_preparation/. Request equality
includes task, question and named verbatim text, but excludes row IDs and slot names.
Permanent responses/features use separate data/jev_cache/<dataset_id>.sqlite3 files.
Source-backed meanings and explicit unknowns live in the question-preparation config.
Jev and five self-hosted response adapters/materializers are implemented; actual GPU
contracts remain unverified. Microsoft's adapter and hosted provider/campaign spending
accounting remain pending. Experiment 0A precedes extraction; Experiment 0B follows it.
Selected extractors: Jev/Microsoft hosted; Liquid/Strands/Perplexity/Clef/Intern-Decision-4B
open weights on VSC. Do not add further models or substitute paid APIs for open weights.
Use src.cluster.transfer export/import for VSC-to-local results: validated dataset cache
unions, backups, collision preservation and idempotent resume. Never overwrite a populated
cache with SCP. Download bundles go to PROJECT on VSC; compact receipts go to DATA.
`src.cluster.token_audit` counts complete pinned open-model inputs without weights. Its large
SQLite counts belong on PROJECT and compact summaries/logs on DATA, under experiment 0.
Explore all 20 datasets; exclude Zomato and Spotify from feature creation and predictive experiments based
on text burden and 114-class representation size, not predictive results. Preserve both raw datasets
and mark them REMOVED on selected exploration overviews; do not keep a Spotify stress-test arm. VSC experiment predictions and
large outputs use PROJECT; compact manifests and operational logs use DATA. The comparison is
decision-feature benefit within each fixed predictor, not a model leaderboard: no HPO,
arbitrary training-row cap, optional fine-tuning or time/group splitting. Selected main
predictors: CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular and linear/logistic
regression, each with four non-text/joint/per-column/both arms. Native-text baselines are
TabSTAR and ConTextTab only, with original features. TabSTAR's required default adaptation
is part of its method. All original non-text columns, including categorical columns, remain.
Lazy adapters are source-checked and synthetically tested; live runtimes are unvalidated.
Five shuffled stratified folds for classification, shuffled KFold for regression; binary
thresholds use three inner OOF folds, then refit on the full outer training fold.
Do not stage, commit, or push changes. The owner handles Git publication; this supersedes earlier
publication authorization. Read-only Git inspection is allowed.
Give proposed plans and answers in chat, not in temporary documents. Keep `docs/` limited to
durable references, without duplicating decisions or creating a research-plan file.

## 1. `tfm-library/` IS READ-ONLY. NO EXCEPTIONS BUT ONE.

A **pinned git submodule** holding the shared TFM literature. This repository does not track its
contents.

It is here so you can answer *"what does the literature say?"* and *"how does the official
implementation do this?"* **by reading and grepping files in this repository** — offline, no web
search, nothing from memory. Use it. A claim you can point at a path for beats a confident
sentence.

**Never create, edit, move, or delete anything inside it** — not a typo fix, not a note, not a
reformat. Anything you write there is lost when the pin moves, or corrupts a resource every other
project shares. **The one exception** is `tfm-library/PROJECT_SPECIFIC.md`, gitignored by the
library for exactly this purpose and created from `PROJECT_SPECIFIC.template.md`. If a library
document is wrong, report it to Andreas Goethals rather than patching it — the fix belongs in the
library's own checkout, where it flows down to every consumer. Never lint, format or test it.

Cite papers by path (`tfm-library/papers/<year>/...`, full text under `papers/text/`), and **code
dumps by symbol name, never by line number** — the dumps are re-snapshotted and line numbers drift
by thousands. Record the pin (`git submodule status`) next to any result that depends on it. Bump it with
`python -m src.utils.update_tfm_library`.

## 2. Never commit data or checkpoints

`data/` holds datasets under varying licences; `checkpoints/` holds multi-hundred-MB weights. Both
are gitignored. Do not `git add -f` them, and do not paste raw rows into commits, issues or docs.

## 3. Verify before you assert

This project's value is careful measurement. If a claim cannot be confirmed from the library, the
upstream source, or a primary reference, **say so** rather than filling the gap plausibly.
Distinguish what a paper *evaluated* from what its code merely *supports*; what a mechanism *can*
represent from how *often* it occurs; a library annotation from the primary source it summarises.

## 4. Honor the authorized stage

Main Jev features are settled: binary probability, full multiclass probability vector, expected regression direction. Jev version is pinned to jev-1.13.0. Headline metric selection does not block recording all metrics.

No paid extraction, model runs or VSC allocations are authorized in exploration. Do not enable
them as part of a refactor. Honor existing owner authorization for environment setup;
do not ask repeatedly for an already authorized action. Check before changing a shared environment.

## 5. Windows PowerShell 5.1 — no `&&`

No `&&`, no ternary, no `??`. One command per line, or `;` with `if ($?) { ... }`:

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — bash on Linux, normal POSIX syntax. Keep the two straight.

## 6. Write both logs

Newest first, dates `DD-MM-YYYY`.

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — every substantive change, one bullet, **as short as
  possible**: what, and why only if it is not obvious. Usually one line; longer only when the change
  genuinely is. The detail belongs in the commit.
- [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — two things. **Every cluster run**, one row in
  the table: config, outcome, headline number. And **every failure** that cost more than a couple of
  minutes and has reusable lessons. Keep the notes concise; Git history retains superseded details.

## 7. Notebooks and figures

- A notebook contains **no `def` and no `class`** — logic goes in `src/` — and its **last code
  cell prints a text summary**, section by section, in the notebook's own section order.
- **Never pick a colour or a size.** `src/visualize/style.py` owns both, so every notebook here
  looks the same. Add a new one there, once, not in the notebook.
- Save through `src/visualize/figures.FigureSaver`: **PDF only**, into `output_JEVPFN/figures/<phase>/<notebook>/`, which it clears before drawing. The notebook displays each figure inline.
- Use `style.figsize(style.WIDTH_FULL)` or `WIDTH_HALF`: every figure is drawn at the width it
  will occupy on an **A4** page, and never rescaled afterwards — rescaling carries the text with it.
- Detailed inspection dashboards may use the larger named canvas sizes in `style.py`; their
  captions identify them as inspection overviews, not figures to shrink onto an A4 page.
- Captions are the **paper's** captions: pure description, ready to paste under the figure.

## 8. Say you are done only when it runs

```powershell
python -m pytest -q
```

And `python -m src.utils.run_notebooks` if you touched a notebook or anything under
`src/visualize/`. Nothing runs these for you — no CI, no hook — so run them before you claim
anything is finished.
