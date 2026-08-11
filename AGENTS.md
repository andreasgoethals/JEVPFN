# AGENTS.md — {{PROJECT_NAME}}

Instructions for AI agents working in this repository. Read this, then
[`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md), before touching anything.

## 0. Before you start

1. Read [`docs/TEMPLATE.md`](docs/TEMPLATE.md) — the governing document for the
   layout and every rule.
2. Read [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — what has already been
   tried here and **failed**. It exists so you do not spend an hour rediscovering
   a dead end. Reading it is not optional.
3. Read [`README.md`](README.md) for what this project actually is.

**If a `_template/` folder still exists, stop and read
[`_template/INITIALISE.md`](_template/INITIALISE.md) first.** This repository is either the
un-initialised template or a project where nobody deleted the folder, and that file tells the
two apart in one command. Until a project name has been written in, the compliance test's
template-hash check stays dormant — so the layout is not yet being enforced, and turning it on
is the first job. Say in your reply which of the two cases you found.

## 1. `tfm-library/` IS READ-ONLY. NO EXCEPTIONS BUT ONE.

`tfm-library/` is a **pinned git submodule** — a snapshot of the shared TFM
literature library that every one of these projects consumes. This repository
does **not** track its contents.

It is here so that you can answer *"what does the literature say?"* and *"how
does the official implementation do this?"* **by reading and grepping files in
this repository** — offline, no web search, no recall from memory. Use it. A
claim you can point at a path for is worth more than a confident sentence.

**Never create, edit, move, or delete anything inside `tfm-library/`** — not to
fix a typo, not to add a note, not to "just" reformat. Anything you write there
is either silently lost when the pin moves or silently corrupts a resource shared
with every other project.

**The single exception** is `tfm-library/PROJECT_SPECIFIC.md`. That filename is
gitignored *by the library*, so it lives beside the literature, survives
`git submodule update`, and can never be pushed upstream. It is the only place
project-specific notes about the literature belong. Create it by copying
`tfm-library/PROJECT_SPECIFIC.template.md`.

**If a library document is wrong, do not patch it.** Report it to {{AUTHOR}} so
it is fixed in the library's own checkout and flows down to every consumer.

Never lint it, never format it, never run tests over it. Read
`tfm-library/AGENTS.md` for the full upstream contract.

### Citing the library

- Papers by path: `tfm-library/papers/<year>/<MM>_<Author>_<Title>.pdf`,
  full text at `tfm-library/papers/text/<year>/<same-name>.txt`.
- **Code dumps by symbol name, never by line number.** The dumps are
  re-snapshotted periodically and line numbers drift by thousands.
  `` `TabICL.txt`, `GraphSCM.__call__` `` — yes. `` `TabICL.txt:24994` `` — never.
- When a result depends on the literature, record the pinned commit.
  Current pin: **`<run: git submodule status>`**.

## 2. Follow the template

[`docs/TEMPLATE.md`](docs/TEMPLATE.md) defines the layout and the rules.
**Adhere to it.** The only reason to deviate is that the user has explicitly told
you to, and when you deviate you must **say so in your reply** — never silently.

In short: `src/` holds all importable logic, `scripts/` holds only runnables you
actually invoke, `config/` holds YAML, `docs/` holds documentation, `output/`
holds **everything** the code generates, `tests/` mirrors `src/`.

`python scripts/check.py` is the one command that says whether the repository is
healthy. Run it before you claim you are done.

## 3. Never commit data or checkpoints

`data/` holds datasets under varying licences and `checkpoints/` holds
multi-hundred-MB weights. Both are gitignored. Do not add them, do not
`git add -f` them, and do not paste raw rows into commits, issues, or docs.

## 4. Verify before you assert

This project's value is careful measurement. If a claim cannot be confirmed from
the library, the upstream source, or a primary reference, **say so** rather than
filling the gap plausibly. Distinguish:

- what a paper *evaluated* from what its code merely *supports*;
- what a mechanism *can represent* from how *often* it actually occurs;
- a library annotation from the primary source it summarises.

## 5. Do not run training, install packages, or push without asking

Cluster runs cost real VSC credits. Package installs change a shared environment.
Pushes are {{AUTHOR}}'s action. Ask first.

## 6. Windows PowerShell 5.1 — no `&&`

{{AUTHOR}} works in **Windows PowerShell 5.1**, which has **no `&&` operator**,
no ternary, and no `??`. Never hand over bash-chained commands. One command per
line, or `;` with `if ($?) { ... }`.

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — bash on Linux, normal POSIX syntax.
Keep the two straight.

## 7. Write both logs

Two files, two different jobs. Both newest-first, dates `DD-MM-YYYY`.

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — every substantive change: what, and
  why if it is not obvious.
- [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — every **failure** that cost
  more than a couple of minutes: what you tried, what happened, why, and what to
  do instead. Four lines. Write it even when the eventual fix worked — especially
  then, because the dead end is the expensive part.

## 8. Figures and notebooks

- A notebook contains **no `def` and no `class`.** Logic goes in `src/`; the
  notebook calls it. The compliance test fails on a `def ` in a notebook.
- A notebook's **last code cell prints a text summary** of everything it showed.
- Never pick a colour in a notebook. `src/visualize/style.py` owns every colour,
  and a name means the same colour in every figure. Register a new series name
  there, once.
- Save through `src/visualize/figures.FigureSaver` — PDF plus PNG, into that
  notebook's own folder, which it clears before drawing.
