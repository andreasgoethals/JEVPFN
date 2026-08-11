# AGENTS.md — {{PROJECT_NAME}}

Instructions for AI agents working in this repository.

## 0. Before you start

1. [`docs/TEMPLATE.md`](docs/TEMPLATE.md) — the governing document for the layout and every
   rule. **Adhere to it.** Deviate only when the user says so, and when you do, **say so in
   your reply** — never silently.
2. [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — what has already been tried here and
   **failed**. Reading it is not optional; it exists so you do not spend an hour on a known
   dead end.
3. [`README.md`](README.md) — what this project actually is.

**If a `_template/` folder still exists**, read [`_template/INITIALISE.md`](_template/INITIALISE.md)
first: this is either the un-initialised template or a project where nobody deleted the folder,
and that file tells the two apart in one command. Say which you found.

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
document is wrong, report it to {{AUTHOR}} rather than patching it — the fix belongs in the
library's own checkout, where it flows down to every consumer. Never lint, format or test it.

Cite papers by path (`tfm-library/papers/<year>/...`, full text under `papers/text/`), and **code
dumps by symbol name, never by line number** — the dumps are re-snapshotted and line numbers drift
by thousands. Record the pin (`git submodule status`) next to any result that depends on it.

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
{{AUTHOR}}'s action. Ask first.

## 5. Windows PowerShell 5.1 — no `&&`

No `&&`, no ternary, no `??`. One command per line, or `;` with `if ($?) { ... }`:

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — bash on Linux, normal POSIX syntax. Keep the two straight.

## 6. Write both logs

Newest first, dates `DD-MM-YYYY`.

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — every substantive change: what, and why if it is not
  obvious.
- [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — every **failure** that cost more than a couple
  of minutes: **Tried**, **Result**, **Why**, **Instead**. Write it even when the eventual fix
  worked — especially then, because the dead end is the expensive part.

## 7. Notebooks and figures

- A notebook contains **no `def` and no `class`** — logic goes in `src/` — and its **last code
  cell prints a text summary**. Both are enforced by the compliance test.
- **Never pick a colour.** `src/visualize/style.py` owns every colour, and a name means the same
  colour in every figure. Register a new series name there, once, by appending.
- Save through `src/visualize/figures.FigureSaver`: PDF plus PNG, into that notebook's own folder,
  which it clears before drawing.

## 8. Say you are done only when it passes

```powershell
python scripts/check.py
```

Ruff, an import of every `src` module, and pytest. That is the one command that answers "is this
repository healthy?" — run it before you claim anything is finished.
