# Andreas' repository template

The structure and rules every one of my research repositories follows. The template
repository IS this structure with the name left blank, so a new project is that repository
initialised — never a tree assembled from scratch.

    Author   Andreas Goethals <andreas.goethals@kuleuven.be>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (Genius login, wICE and Mindwell compute)
    Source   https://github.com/andreasgoethals/repo-template

**Generic and read-only.** Never edited from inside a repository — only at the source
above. Anything project-specific belongs in `README.md` or another `docs/` file. This
file names no dataset, no model, no experiment and no result: if a rule cannot be stated
without one, it is not a template rule.

A `tests/test_template_compliance.py` hashes this file against the source, so editing it
inside a repository **fails the test suite**. That is deliberate: a rule asking politely
not to be edited is not a rule.

This is the **one governing document**. Every rule and every piece of structure is
written here; `README.md`, `AGENTS.md` and the tests point back to it rather than
restating it. If two documents disagree, this one wins.

---

## Structure

```
config/                 one YAML per experiment. No subfolders.
data/
  raw/                  never modified, never committed
  processed/            generated cache
docs/                   ONLY .md files, names in CAPITALS
  TEMPLATE.md           this file, never edited here
  CHANGELOG.md          what changed, newest first
  AGENTS_MEMORY.md      what was tried and FAILED, newest first
  VSC.md                this project on the cluster
  ...                   more docs as the project needs
notebooks/              thin: all logic imported from src/
output/                 EVERYTHING the code generates
  All_Results.md        every notebook's text summary, alphabetical
  figures/
    CAPTIONS.md         one shared file for all notebooks
    <notebook>/         one PDF + one PNG per figure
  logs/
  manifests/
  results/
scripts/                only runnable project entry points
  slurm/                cluster job scripts
src/
  data/                 loading and preprocessing
  utils/                paths, config, logging, cleanup, notebook runner
  visualize/            all plotting
  ...                   more subfolders as the project needs (train/, eval/, models/)
tests/                  one file per src module
tfm-library/            submodule: literature + VSC documentation. READ-ONLY.
.github/workflows/      CI running scripts/check.py
.gitattributes
.gitignore
.gitmodules
AGENTS.md
CITATION.cff            optional
LICENSE
README.md
pyproject.toml
```

**The template repository has exactly one extra directory, `_template/`**, holding the
things that must not travel into a project: the initialiser, the template's own self-check,
and the cross-project submodule tool. Its presence is what distinguishes the template from a
project. **A project deletes it** — that is step one after "Use this template".

Extra top-level directories are otherwise allowed only if this file is updated at the source
to name them. Extra `src/` subfolders need no permission — that is the documented extension
point.

---

## Required files

**`README.md`** — what the project is, how to install it, how to run it, the layout.
It **ends** with the short "Based on the repository template" chapter and nothing after
it; everything above that chapter is the project's own. A fresh repository starts with
that chapter alone and grows upward.

**`AGENTS.md`** — the rules an AI agent follows here. Must state: adhere to this
template; deviate only when the user says so, and always say when you deviate;
`tfm-library/` is read-only; never commit data or weights; never install, train, or push
without asking; verify claims against a source rather than filling gaps plausibly; read
`docs/AGENTS_MEMORY.md` before starting and add to it after a failure.

**`LICENSE`** — MIT, covering only the project's own code:

```
MIT License

Copyright (c) <year> <name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Add a line stating that third-party code (submodules, vendored files, datasets) keeps its
own licence.

**`.gitattributes`**

```
* text=auto eol=lf
*.ipynb text
*.pdf binary
*.png binary
*.ckpt binary
```

**`.gitmodules`**

```
[submodule "tfm-library"]
	path = tfm-library
	url = <library repo url>
```

**The template repository carries the submodule as a real gitlink**, not only as this file.
`.gitmodules` records a *path and a URL*; the gitlink — a tree entry of mode `160000` holding a
commit SHA — is what records *which commit*. Without it, `git submodule update --init` in a
fresh fork does nothing, because there is no commit to check out.

That distinction is the whole reason the template carries it. A fork or a "Use this template"
copy inherits both, so a new project has a correctly wired, already-pinned submodule from its
first minute — one `git submodule update --init`, the same command a fresh clone of any project
needs. The alternative, letting each project run `git submodule add`, is a second and different
instruction, needs the URL to be right by hand, and is the step an agent gets wrong.

It costs the template nothing: a gitlink is a SHA. The library's ~749 MB is fetched only when
someone asks for it, and a plain `git clone` never does.

**`.vscode/settings.json`** and **`.vscode/extensions.json`** are committed; the rest of
`.vscode/` is ignored. They are not preferences — `python.analysis.extraPaths` is what lets an
editor resolve `from src.utils...` given the flat `src/` package root, and
`jupyter.notebookFileRoot` is what makes an interactive notebook run from the same directory as
the notebook runner. Both follow from this layout, so they belong to it.

**`.gitignore`** — never commit: raw data, checkpoints, generated caches, virtual
environments, tool caches, editable-install metadata, PDF figures, logs, per-run metrics.
Use globs, and **anchor with a leading slash** when a name should only match at the root
(a bare `figures/` also matches `output/figures/`).

**`pyproject.toml`** — Python **3.11–3.12**, `ruff` for linting, `pytest` for tests,
both in a `dev` extra. `tfm-library/` is excluded from ruff, from pytest collection and
from the installed package. Every non-obvious pin carries a comment saying **why**.

---

## Rules

### `tfm-library/` — the literature submodule

Every repository carries the shared TFM literature library at `tfm-library/`, as a
**pinned git submodule**. It is not optional and it is not per-project.

**What it is.** One curated knowledge base — the papers as PDFs with full-text
extractions, written per-paper summaries, a cross-paper synthesis, flat-text snapshots of
the upstream reference implementations, and the VSC documentation. It is maintained in
its own repository and consumed by every project.

**Why every project has it.** So that a human *or an agent* working in this repository
can answer "what does the literature actually say?" and "how does the official code
actually do this?" **from inside the repository, offline, by reading and grepping files**
— with no web search, no paywall, and no recall from memory. An agent that can read the
sources does not have to guess, and a claim it makes can be traced to a file path. That
is the entire purpose: it turns "I believe X" into "X, see `tfm-library/<path>`".

**Why a submodule rather than a copy.** A submodule pins one exact commit. A result
produced today is reproducible against the literature *as it stood* when it was produced,
and every project shares one consistently maintained copy instead of four drifting ones.

**READ-ONLY. No exceptions but one.**

- Never create, edit, move, rename or delete anything inside `tfm-library/` — not to fix
  a typo, not to add a note, not to reformat. The consuming repository does not track its
  contents, so anything written there is either silently lost when the pin moves or
  silently corrupts a resource four projects share.
- **The single exception** is `tfm-library/PROJECT_SPECIFIC.md`, which the library
  gitignores on purpose. It is the only place project-specific notes about the literature
  belong. Create it by copying `tfm-library/PROJECT_SPECIFIC.template.md`.
- If a library document is wrong, **do not patch it here.** Report it so it is fixed in
  the library's own checkout and flows down to every consumer.
- Never lint, format, or test it. `pyproject.toml` excludes it from ruff and pytest.
- Never let cleanup touch it — it is a protected path in the cleanup helper.

**Citing it.**

- Papers by path: `tfm-library/papers/<year>/<MM>_<Author>_<Title>.pdf`, full text at
  `tfm-library/papers/text/<year>/<same-name>.txt`.
- **Code dumps by symbol name, never by line number.** The dumps are re-snapshotted
  periodically and line numbers drift by thousands. `` `TabICL.txt`, `GraphSCM.__call__` ``
  — yes. `` `TabICL.txt:24994` `` — never.
- When a result depends on the literature, **record the pinned commit** next to the
  result and in `AGENTS.md`.

**Commands** (run in the consuming project, never inside the folder; PowerShell has no
`&&`, so one command per line):

```
git submodule update --init            # after a fresh clone — the folder starts empty
git submodule status                   # which commit this project is pinned to
git submodule update --remote tfm-library
git add tfm-library
git commit -m "Bump tfm-library pin"
```

The pin is **not** recorded until the `git add` and commit. Between the two,
`git submodule status` shows a leading `+`; that is normal. `scripts/update_tfm_library.py`
does all of this, reports first and changes nothing without `--update`.

### `docs/`

Only `.md` files, names in CAPITALS.

- **`TEMPLATE.md`** — this file. Shared across repositories, never edited from inside one.
- **`VSC.md`** — reads `tfm-library`'s VSC documentation and turns it into a guide for
  *this* project: which partitions, walltime limits, how to submit, how to resume a job
  that outlives the walltime, and where files go on the two storage tiers.
- **`CHANGELOG.md`** — one chapter per date, `DD-MM-YYYY`, **newest at the top**. Terse:
  what changed, and why if it is not obvious. All rules and rule changes are recorded here.
- **`AGENTS_MEMORY.md`** — one chapter per date, `DD-MM-YYYY`, **newest at the top**. Not
  what changed — **what was tried and did not work.** Dead ends, wrong assumptions,
  approaches that looked right and failed, and the cheap check that would have caught each
  one. The changelog records the road taken; this records the roads closed, so nobody pays
  for the same mistake twice. Every entry has exactly four lines: **Tried**, **Result**,
  **Why**, **Instead**. An agent reads this file *before* starting and appends to it after
  any failure that cost more than a couple of minutes.

### `output/`

**Everything the code generates goes under `output/`**, locally and on the cluster. One
root, so "what did this run produce" and "what can I delete" have one answer. Nothing
generated is written anywhere else — not next to a notebook, not into `src/`, not into a
new top-level folder. The compliance test enforces this.

- **`output/results/`** — the **fine-grained** results: one row per prediction, per-fold
  scores, anything large. On the cluster this directory lives on **project storage**, not
  `$VSC_DATA`, because per-row predictions across every dataset and model run to gigabytes
  and `$VSC_DATA` is small and backed up. Everything else under `output/` stays on
  `$VSC_DATA`.
- **`output/All_Results.md`** — every notebook's printed text summary, concatenated in
  alphabetical notebook order.
- **`output/figures/CAPTIONS.md`** — **one** file for all notebooks, grouped per notebook,
  figures in the order they appear in that notebook. Each entry gives the figure's name,
  then the caption underneath.
- **Captions are pure description.** What is plotted, on what axes, from how much data.
  No interpretation — exactly what would sit under the figure in a journal paper.

### Notebooks

- **All logic lives in `src/`.** A notebook only calls it. A notebook contains no `def`
  and no `class` — if you need one, it belongs in `src/` where it can be imported and
  tested.
- Every notebook **ends by printing a text summary** of everything it showed.
- Rerunning a notebook **deletes the figures that same notebook produced before**.
- Every figure is saved as **PDF** (high DPI, for the paper) *and* rendered as **PNG**
  (lower DPI, small enough to commit).
- There is **one file that reruns every notebook**, in parallel, and regenerates
  `CAPTIONS.md` and `All_Results.md`.
- **One shared visual style across every notebook**, in a single `src/visualize/style.py`:
  the same fonts, sizes, grid and — most importantly — the **same colours meaning the same
  thing in every figure**. A reader should never have to re-learn the legend, and figures
  from different notebooks must sit together in one paper without looking like they came
  from different projects. The mapping from a name to a colour is declared **once**, in
  that module; a notebook never picks a colour itself.
- **Figures are saved by the notebook itself**, not only by the runner, so an interactive
  run in Jupyter produces the same files. A notebook deletes **its own** figures — never
  another notebook's — and does so **before** it draws anything.

### `scripts/`

Only real, runnable entry points for the project's main experiments, plus `slurm/`.
Anything importable belongs in `src/` — a module in `scripts/` cannot be imported or
tested. Every `.py` file directly in `scripts/` therefore has an
`if __name__ == "__main__":` block; the compliance test checks it.

There must be a **Python script, runnable locally and on the cluster, that deletes all
output from a previous run.** It lists by default and deletes only when asked. Raw data
and downloaded model weights can never be deleted by it.

There must be **one command that answers "is this repository healthy?"** —
`scripts/check.py`, which runs ruff, pytest and an import check of every `src` module and
prints one verdict. Nothing else needs to be remembered before a commit.

### `config/`

One YAML per experiment. No subfolders, no inheritance.

- Every knob with more than one value goes in a **`sweep:` block at the top**; the full
  cartesian product is run. Everything below it is a single value.
- **One short comment per knob**, saying what it is.

### `src/`

Always has `data/`, `utils/`, `visualize/`. Add more subfolders as the project needs
(`train/`, `eval/`, `models/`). Paths are built in **one** module, never by string
concatenation at the call site, and relative paths resolve against the repository root so
tools work from any directory.

### `tests/`

One file per `src` module, plus `test_template_compliance.py`. Tests never write outside
`tmp_path` or `output/`.

### Comments

Every non-obvious decision carries a short comment saying **why**, not what. A pin, a
fallback, an exclusion, a magic number, an ordering that matters: say what breaks if it
changes. Comments that restate the code are noise and are deleted.

---

## The compliance test

`tests/test_template_compliance.py` is how these rules stop being advice. It **fails**
when:

| # | Failure |
|---|---|
| 1 | `docs/TEMPLATE.md` differs from the template source (SHA-256 comparison against the hash the initialiser baked in, and against a root `TEMPLATE.md` if one exists) |
| 2 | a required directory or file from § Structure is missing |
| 3 | anything generated is written outside `output/` — a hard-coded write path in `src/` or `scripts/`, or a stray generated file in the working tree |
| 4 | a `.py` file directly in `scripts/` has no `if __name__ == "__main__":` block |
| 5 | a notebook contains a `def ` (or a `class `) — logic belongs in `src/` |
| 6 | a notebook's last code cell does not print a text summary |
| 7 | a `.gitignore` rule that should be root-anchored is not (a bare `figures/` also matches `output/figures/`) |

It is a **repository** test, not a code test: it reads files and never imports the
project. That keeps it runnable on a fresh clone before anything is installed.

Check 1 stays dormant while the repository is still the un-initialised template — there is
nothing to compare against until a project exists. It switches on the moment the initialiser
has run, and cannot be switched off again.

---

## VSC storage

Two locations. On **both**, everything lives inside a folder named after the project.

| tier | path | holds | backed up |
|---|---|---|---|
| **project storage** | `/lustre1/project/stg_00211/<ProjectName>/` | big files: datasets, checkpoints, generated caches, **`output/results/`** | no |
| **personal data** | `$VSC_DATA/<ProjectName>/` | the repo, and the rest of `output/` (figures, logs, manifests, the two `.md` summaries) | yes |

`$VSC_DATA` is small (75 GiB), so nothing large may go there. Project storage has a low
inode budget, so it wants few big files rather than thousands of small ones. `$VSC_SCRATCH`
exists but is purged after 30 days without access — working scratch only.

One module — `src/utils/paths.py` — resolves both tiers and is the only place a path is
built. Off-cluster both tiers collapse to the repository root, so the same code runs on a
laptop with no configuration.

---

## Starting a new repository

The template **is** a project with the name left blank. There is no copying step.

1. On GitHub, **Use this template** → new repository. Set the visibility you want. Clone it.
2. `python _template/init_project.py <ProjectName> --description "..."` — fills in every
   placeholder and bakes this file's SHA-256 into the compliance test.
3. Delete `_template/`. Nothing in the project imports from it.
4. `git submodule update --init` — the pin came with the repository; this fetches its content.
5. `python scripts/check.py` — must pass before the first commit.

**"Use this template", not a fork.** Two reasons, and the second is the bigger one:

- **A fork of a public repository can never be made private.** GitHub does not allow changing a
  fork's visibility. "Use this template" lets the new repository be private even when the
  template is public, which is what a project under embargo or with licence-restricted data
  needs.
- **The merge a fork promises does not work in practice.** The appeal is
  `git pull upstream main` to bring a rule change down. But the files a template change touches
  most are exactly the ones the initialiser rewrote per project — `README.md`,
  `pyproject.toml`, `src/utils/paths.py`, and the hash line in the compliance test — so every
  pull is a conflict in the files you least want to merge by hand. A fork also puts the
  template's commits at the root of the project's history and makes GitHub default a pull
  request to the *upstream*, so the obvious button proposes your project's work to the template.

Rule changes instead flow **from** the template, deliberately, with
`_template/sync_template_rules.py` — see below.

**An agent doing the initialising** follows `_template/INITIALISE.md`, which lists the facts to
ask for, the one command, and the files only a human judgement can fill.

There is exactly one `AGENTS.md`, one `docs/CHANGELOG.md` and one `docs/AGENTS_MEMORY.md`,
and they are the project's. The template ships them seeded rather than keeping a second set
of its own. A lesson that turns out to apply everywhere is **promoted** from a project's
`AGENTS_MEMORY.md` into the template's copy, so new projects start already knowing it.

### Changing a rule

1. Edit `docs/TEMPLATE.md` **in the template repository**, and record the change in its
   `docs/CHANGELOG.md`.
2. Add or update the check in `tests/test_template_compliance.py`, and add the matching
   violation to `VIOLATIONS` in `_template/check_template.py`.
3. `python _template/check_template.py` — the new violation must be **caught**. A rule whose
   check has only ever been seen to pass is not enforced.
4. Propagate it: `python _template/sync_template_rules.py --apply`. That copies the new
   `docs/TEMPLATE.md` into every project and re-bakes each one's SHA-256. It reports by default
   and writes only with `--apply`; it never commits and never pushes.
5. In each project: `git diff`, then `python scripts/check.py`, then a line in that project's
   `docs/CHANGELOG.md`.

Step 4 is what makes the hash check affordable, and step 5 is the point of it: a rule cannot
drift into a project silently, and it cannot be quietly ignored in one either. Files other than
`docs/TEMPLATE.md` are **reported** as differing rather than overwritten, because a project is
allowed to extend some of them and expected to delete others; take one deliberately with
`--also <path>`.
