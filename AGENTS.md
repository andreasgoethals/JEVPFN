# AGENTS.md — JEVPFN

Instructions for AI agents working in this repository.

## 0. Before you start

1. [`docs/TEMPLATE.md`](docs/TEMPLATE.md) — the layout and rules this project started from.
   **Follow it**, and when you deviate — which is allowed, it is a starting point rather than a
   contract — **say so in your reply**. Never silently.
2. [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — the cluster runs already done and the dead
   ends already hit. Reading it is not optional: it is how you avoid resubmitting a configuration
   that failed last month, or spending an hour on a known dead end.
3. [`README.md`](README.md) — what this project actually is.

This project is initialised; the owner requested removal of the inherited `_template` files.
Current phase layout and setup instructions live in `README.md`, not in that directory.
The owner selected text-only per-column and joint Jev inputs, skipping empty inputs and storing
missing features. All Jev feature creation is local before VSC experiment 0. Generated full
reports contain raw examples and are gitignored under `output_JEVPFN/<phase>/`.
Publication to the public `andreasgoethals/JEVPFN` repository is explicitly authorized in this session.

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

## 4. Do not train, install, or push without asking

Cluster runs cost real VSC credits. Installs change a shared environment. Pushes are
Andreas Goethals' action. Ask first.

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
  minutes, four lines: **Tried**, **Result**, **Why**, **Instead** — even when the eventual fix
  worked, because the dead end is the expensive part.

## 7. Notebooks and figures

- A notebook contains **no `def` and no `class`** — logic goes in `src/` — and its **last code
  cell prints a text summary**, section by section, in the notebook's own section order.
- **Never pick a colour or a size.** `src/visualize/style.py` owns both, so every notebook here
  looks the same. Add a new one there, once, not in the notebook.
- Save through `src/visualize/figures.FigureSaver`: **PDF only**, into that notebook's own
  folder, which it clears before drawing. The notebook displays each figure inline.
- Use `style.figsize(style.WIDTH_FULL)` or `WIDTH_HALF`: every figure is drawn at the width it
  will occupy on an **A4** page, and never rescaled afterwards — rescaling carries the text with it.
- Captions are the **paper's** captions: pure description, ready to paste under the figure.

## 8. Say you are done only when it runs

```powershell
python -m pytest -q
```

And `python -m src.utils.run_notebooks` if you touched a notebook or anything under
`src/visualize/`. Nothing runs these for you — no CI, no hook — so run them before you claim
anything is finished.
