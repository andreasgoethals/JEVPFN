# Andreas' repository template

The structure and rules every one of my research repositories follows.

    Author   Andreas Goethals <andreas.goethals@kuleuven.be>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (Genius login, wICE and Mindwell compute)
    Source   https://github.com/andreasgoethals/0.-Template

**This is the one governing document.** Every rule is here; `README.md`, `AGENTS.md` and the
tests point back to it rather than restating it. If two documents disagree, this one wins.

**Generic and read-only.** Edited only at the source above, never from inside a repository —
`tests/test_template_compliance.py` hashes it, so a local edit fails the test suite. That is
deliberate: a rule asking politely not to be edited is not a rule. Project-specific rules go in
`README.md` or a new `docs/<NAME>.md`. This file names no dataset, model, experiment or result:
if a rule cannot be stated without one, it is not a template rule.

---

## Retrofitting an existing repository

Hand this file to an agent in a repository that predates it. In order, stopping to report:

1. **Inventory.** List what exists against § Structure. Name what is missing, what is misplaced,
   and what exists under another name — move nothing yet.
2. **`output/`.** Create it and move every generated artefact under it. Usually the largest change
   and the one that breaks imports; do it first, then fix the call sites.
3. **`src/utils/paths.py`.** Build it, then replace every hard-coded path with a call to it. Grep
   for string literals containing `/`, `output`, `results`, `figures`, `logs`.
4. **The rest of `src/`** — see § Module contract.
5. **Notebooks.** Move every `def` and `class` into `src/`. Make each notebook clear its own
   figures, save through `FigureSaver`, and end by printing a text summary.
6. **`scripts/`.** Move anything importable into `src/`. Every remaining `.py` gets a `__main__`
   block. Add `check.py` and `clean_run.py`.
7. **`docs/`.** `.md` only, CAPITALS. Add `CHANGELOG.md`, `AGENTS_MEMORY.md`, `VSC.md`, and this
   file. Then `AGENTS.md`, `LICENSE`, `.gitattributes`, `.gitignore`, `.gitmodules`.
8. **`tfm-library/`.** Add the submodule if absent.
9. **`tests/test_template_compliance.py`** last, with this file's SHA-256 baked in, and make it
   pass. Copy it from the template rather than writing it fresh.
10. **Report** every deviation you left and why. Do not silently narrow the scope.

Deviate only when told to, and always say so.

---

## Structure

```
config/                 one YAML per experiment. Flat, no subfolders.
data/raw/               inputs. Never modified, committed, or deleted by a tool.
data/processed/         generated cache
docs/                   ONLY .md files, names in CAPITALS
  TEMPLATE.md             this file, never edited here
  CHANGELOG.md            what changed, newest first
  AGENTS_MEMORY.md        what was tried and FAILED, newest first
  VSC.md                  this project on the cluster
notebooks/              thin: all logic imported from src/
output/                 EVERYTHING the code generates
  All_Results.md          every notebook's printed summary, alphabetical
  figures/CAPTIONS.md     ONE captions file for all notebooks
  figures/<notebook>/     one PDF + one PNG per figure
  logs/  manifests/  results/  runs/
scripts/                only runnable entry points. slurm/ for cluster jobs.
src/data/               loading and preprocessing
src/utils/              paths, config, logging, cleanup, notebook runner
src/visualize/          all plotting
src/...                 more as needed (train/, eval/, models/)
tests/                  one file per src module
tfm-library/            submodule: literature + VSC docs. READ-ONLY.
.github/workflows/      CI running scripts/check.py
.vscode/                settings.json and extensions.json only
.gitattributes  .gitignore  .gitmodules
AGENTS.md  CITATION.cff  LICENSE  README.md  pyproject.toml
```

New `src/` subfolders need no permission — that is the extension point. Any other new top-level
directory requires this file to be updated at the source.

The template repository alone has `_template/`, holding what must not travel into a project: the
initialiser, the template's self-check, and the cross-project sync tools. **A project deletes it.**

**`README.md`** ends with the short "Based on the repository template" chapter and nothing after
it; everything above is the project's own. **`LICENSE`** is MIT in the author's name, plus a line
stating that third-party material (the submodule, datasets, downloaded weights, vendored files)
keeps its own licence. **`pyproject.toml`** targets Python **3.11–3.12**, with `ruff` and `pytest`
in a `dev` extra, and excludes `tfm-library/` from ruff, from pytest collection and from the
package. Every non-obvious pin carries a comment saying **why**.

```
# .gitattributes            # .gitmodules
* text=auto eol=lf          [submodule "tfm-library"]
*.ipynb text                	path = tfm-library
*.pdf *.png *.ckpt binary   	url = <library repo url>
```

`.vscode/settings.json` and `extensions.json` are committed; the rest of `.vscode/` is ignored.
They are not preferences: `python.analysis.extraPaths` is what lets an editor resolve
`from src.utils...` given the flat `src/` package root, and `jupyter.notebookFileRoot` is what
makes an interactive notebook run from the same directory as the notebook runner.

---

## Rules

### `tfm-library/` — the literature submodule

Every repository carries the shared TFM literature at `tfm-library/` as a **pinned git
submodule**. Not optional, not per-project.

**What it is.** One curated knowledge base: the papers as PDFs with full-text extractions,
per-paper summaries, a cross-paper synthesis, flat-text snapshots of the upstream reference
implementations, and the VSC documentation. Maintained in its own repository, consumed by all.

**Why every project has it.** So a human *or an agent* can answer "what does the literature say?"
and "how does the official code actually do this?" **from inside the repository, offline, by
reading and grepping files** — no web search, no paywall, no recall from memory. It turns "I
believe X" into "X, see `tfm-library/<path>`". That is the entire purpose.

**Why a submodule.** It pins one exact commit, so a result stays reproducible against the
literature *as it stood* when it was produced, and every project shares one maintained copy
instead of four drifting ones.

**READ-ONLY, one exception.** Never create, edit, move, rename or delete anything inside it — not
a typo fix, not a note, not a reformat. The repository does not track its contents, so anything
written there is either lost when the pin moves or corrupts a resource four projects share. The
exception is `tfm-library/PROJECT_SPECIFIC.md`, which the library gitignores for exactly this
purpose; create it from `PROJECT_SPECIFIC.template.md`. If a library document is wrong, report it
upstream — do not patch it here. Never lint, format or test it: `pyproject.toml` excludes it, and
cleanup treats it as protected.

**Citing it.** Papers by path: `tfm-library/papers/<year>/<MM>_<Author>_<Title>.pdf`, full text at
`papers/text/<year>/<same-name>.txt`. **Code dumps by symbol name, never by line number** — the
dumps are re-snapshotted and line numbers drift by thousands. Record the pinned commit next to any
result that depends on the literature.

**A repository carries a real gitlink**, not just `.gitmodules`. `.gitmodules` records a path and
URL; the gitlink (a tree entry of mode `160000`) records *which commit*. Without it,
`git submodule update --init` does nothing.

```
git submodule update --init          # after a clone; the folder is empty until you ask
git submodule status                 # which commit this project is pinned to
python scripts/update_tfm_library.py # bump the pin; reports first, --update to move it
```

`git submodule update --remote` moves the working tree but does **not** record the pin — a leading
`+` in `git submodule status` is that, not an error. The library is ~749 MB; a plain `git clone`
never fetches it.

### `output/` — one root for everything generated

**Everything the code generates goes under `output/`**, locally and on the cluster. Nothing
generated is written anywhere else: not beside a notebook, not into `src/`, not into a new
top-level folder. One root means "what did this run produce?" and "what can I delete?" each have
one answer. Enforced.

- **`output/results/`** — fine-grained results: one row per prediction, per-fold scores, anything
  large. On the cluster this alone lives on **project storage**; per-row predictions across every
  dataset and model reach gigabytes and `$VSC_DATA` is 75 GiB. Locally it is a plain subdirectory.
- **`output/All_Results.md`** — every notebook's printed summary, in alphabetical notebook order.
- **`output/figures/CAPTIONS.md`** — **one** file for all notebooks, grouped per notebook, figures
  in the order that notebook drew them.
- **Captions are pure description.** What is plotted, on what axes, from how much data. No
  interpretation — exactly what would sit under the figure in a journal.

### Notebooks

- **All logic lives in `src/`.** A notebook only calls it, and contains **no `def` and no
  `class`** — a function defined in a notebook cannot be imported or tested, so it gets copied
  into the next notebook and the two copies then diverge.
- Every notebook **ends by printing a text summary** of everything it showed. That text is what
  `All_Results.md` is built from; a notebook ending on a plot contributes nothing to it.
- **A notebook saves its own figures**, not the runner, so an interactive *Run All* produces
  exactly the same files. It clears **its own** figure folder — never another's — **before**
  drawing anything: a stale PDF beside a fresh one is how a paper ends up with a figure that no
  longer matches the code that made it.
- Every figure is saved as **PDF** (300 dpi, for the paper) *and* **PNG** (110 dpi, committed).
- **One file reruns every notebook**, in parallel, and rebuilds both summary documents.
- **One shared style, in `src/visualize/style.py`**: same fonts, sizes, grid, and above all **the
  same colours meaning the same thing in every figure**. A reader learns the legend once, and
  figures from different notebooks sit together in one paper. A notebook never picks a colour.

### `scripts/`

Only real, runnable entry points, plus `slurm/`. Anything importable belongs in `src/` — a module
in `scripts/` cannot be imported or tested, so something importable there is untestable by
construction. Every `.py` directly in `scripts/` therefore has a `__main__` block. Two are
required:

- **`clean_run.py`** — deletes output from a previous run, on a laptop and on the cluster. **Lists
  by default, deletes only when asked.** `data/raw/`, downloaded weights and `tfm-library/` can
  never be deleted by it, by construction rather than by a flag.
- **`check.py`** — the one command for "is this repository healthy?": ruff, then an import of
  every `src` module, then pytest. CI runs the same script, so the failure a reviewer sees is the
  one the author can reproduce. The import step exists because ruff parses files without importing
  them and pytest only imports what a test touches.

### `config/`

One YAML per experiment. Flat — no subfolders, no inheritance, no includes: a config file is read
top to bottom and that is the whole story.

- Every knob with more than one value goes in a **`sweep:` block at the top**; the full cartesian
  product is run. Everything below it is a single value.
- **One short comment per knob**, saying what it is.
- A run writes the fully expanded config it used into its own output directory. The YAML may have
  been edited since, and a sweep point is not in the YAML at all.

### `src/` and `tests/`

`src/` always has `data/`, `utils/`, `visualize/`. **Paths are built in one module and nowhere
else**, resolving against the repository root so tools work from any directory — a path assembled
at a call site with `"output/" + name` is correct on a laptop and wrong on the cluster.

`tests/` has one file per `src` module plus `test_template_compliance.py`, and never writes
outside `tmp_path` or `output/`.

### `docs/`

`.md` only, names in CAPITALS.

- **`VSC.md`** — reads `tfm-library`'s VSC documentation and turns it into a guide for *this*
  project: partitions, walltime limits, how to submit, how to resume a job that outlives the
  walltime, and where files go on the two tiers.
- **`CHANGELOG.md`** — one chapter per date, `DD-MM-YYYY`, **newest at the top**. What changed, and
  why if it is not obvious. All rules and rule changes are recorded here.
- **`AGENTS_MEMORY.md`** — same format, but **what was tried and did *not* work.** Dead ends, wrong
  assumptions, approaches that looked right and failed. The changelog is the road taken; this is
  the roads closed, so nobody pays twice. Four lines per entry — **Tried**, **Result**, **Why**,
  **Instead** — for any failure that cost more than a couple of minutes, including ones eventually
  fixed: the fix is one changelog line, the dead end is the hour. An agent reads it *before*
  starting. Never delete entries.

### `AGENTS.md` and comments

`AGENTS.md` must state: adhere to this template, deviate only when told and always say so;
`tfm-library/` is read-only; never commit data or weights; never install, train, or push without
asking; verify claims against a source rather than filling gaps plausibly; read
`docs/AGENTS_MEMORY.md` before starting and add to it after a failure; Windows PowerShell 5.1 has
no `&&`, so one command per line.

Every non-obvious decision carries a short comment saying **why**, not what: a pin, a fallback, an
exclusion, a magic number, an ordering that matters. Say what breaks if it changes. Comments that
restate the code are deleted.

---

## Module contract

What each required module must do. Reasons live in their docstrings.

| Module | Must provide |
|---|---|
| `src/utils/paths.py` | Every path in the project. Two VSC tiers, one resolver, repo-root-relative, collapsing to the repo off-cluster. `outputs_dir`, `results_dir`, `logs_dir`, `manifests_dir`, `figures_dir(nb)`, `captions_path`, `all_results_path`, `raw_dir`, `processed_dir`, `checkpoints_dir`, `config_path`, `ensure`, `resolve_writable`, `touch_tree`, `describe`. |
| `src/utils/config.py` | `load` → `sweep_axes` → `n_points` → `expand` (one flat dict per sweep point, deterministic order), `get` (dotted), `resolved_dump`. |
| `src/utils/logging_setup.py` | `setup()` once, `get_logger(__name__)` everywhere. Console **and** `output/logs/` — on the cluster stdout is a SLURM file that moves on requeue. |
| `src/utils/run_artifacts.py` | `find_artifacts`, `summarise`, `clean(dry_run=True)`, `protected_paths`. Cheap categories by default, expensive ones named explicitly. Deletes contents, not directories, so tracked `.gitkeep` markers survive. |
| `src/utils/run_notebooks.py` | `discover` (glob, alphabetical — never a hard-coded list), `run_one` in **separate processes** (matplotlib's figure registry is global), `write_captions`, `write_all_results`, `run_all`. |
| `src/visualize/style.py` | `apply()`, and `color(name)` as the only way to get a colour. A **validated** categorical order, semantic roles, sequential and diverging maps, journal column widths. |
| `src/visualize/figures.py` | `FigureSaver(notebook)` — clears its own folder on construction, writes PDF + PNG with a numbered prefix, records each caption in a manifest so `CAPTIONS.md` is rebuildable from disk. |
| `src/data/loaders.py` | Read inputs, cache the processed form. Never build a path; `data/raw/` is read-only; the cache marker is written **last**, so a run killed halfway leaves a cache correctly treated as absent. |

### The colour rule, concretely

`color(name)` resolves a **semantic role** (`baseline`, `proposed`, `observed`, `alternative`,
`highlight`, `annotation`), then a **status** (`good`/`warning`/`serious`/`critical`, never reused
as a series), then a **registered series name** → its fixed categorical slot. The project declares
its series names once, in that module, and **appends** — inserting repaints every figure after it
and invalidates any already in a paper. Colour follows the entity, never its rank: if a figure
drops a series, a plain cycler shifts every colour after it and the same model is blue in one
figure and orange in the next.

The categorical **order is a colour-vision-safety mechanism, not decoration** — reordering changes
which pairs are adjacent. Validate any change: adjacent-pair CVD ΔE ≥ 8 and normal-vision ΔE ≥ 15
(OKLab ×100) against the surface the figure renders on. Forms where every pair is visible at once
(scatter, bubble, small multiples) cap lower than neighbour-only forms (bars, lines, stacks). Raise
past the last distinguishable slot rather than generating another hue. One mode only: a figure for
a paper renders on white, so there is no dark variant to keep in sync.

---

## The compliance test

`tests/test_template_compliance.py` is how these rules stop being advice. It **fails** when:

| # | Failure |
|---|---|
| 1 | `docs/TEMPLATE.md` differs from the template source — SHA-256 against the baked-in hash, and against a root `TEMPLATE.md` if one exists |
| 2 | a required directory or file from § Structure is missing |
| 3 | anything generated is written outside `output/` — a stray generated file in the tree, or a hard-coded write path in `src/` or `scripts/` |
| 4 | a `.py` directly in `scripts/` has no `if __name__ == "__main__":` block |
| 5 | a notebook contains a `def ` or a `class ` |
| 6 | a notebook's last code cell does not print a text summary |
| 7 | a `.gitignore` rule that should be root-anchored is not — a bare `figures/` also matches `output/figures/` |

Plus: `docs/` holds only CAPITALISED `.md`; `config/` is flat; both dated logs are `DD-MM-YYYY`
newest-first; `README.md` ends with the template chapter.

It is a **repository** test, not a code test: it reads files and never imports the project, so it
runs on a fresh clone before anything is installed. Check 1 is dormant while the repository is
still the un-initialised template — there is nothing to compare against until a project exists —
and cannot be switched off afterwards. **A rule is not enforced until its check has been seen to
fail** on a repository that violates it; passing on a clean one proves nothing.

---

## VSC storage

Two locations. On **both**, everything lives inside a folder named after the project.

| tier | path | holds | backed up |
|---|---|---|---|
| **project storage** | `/lustre1/project/stg_00211/<Project>/` | big files: datasets, checkpoints, caches, **`output/results/`** | no |
| **personal data** | `$VSC_DATA/<Project>/` | the repo, and the rest of `output/` | yes |

`$VSC_DATA` is 75 GiB, so nothing large goes there. Project storage has a **low inode budget** —
few big files, not thousands of small ones — so logs and per-step metrics stay on `$VSC_DATA`.
`$VSC_SCRATCH` is purged after 30 days **without access**, and `mv` and timestamp-preserving
`rsync` do not count as an access: copy, then `paths.touch_tree()`. Compute nodes have no outbound
internet, so anything that downloads happens on a login node first. Any run that can exceed the
walltime must be resumable, writing its state pointer **last** so a job killed mid-write points at
the previous complete checkpoint.

---

## Starting a new repository

The template **is** a project with the name left blank. There is no copying step.

1. On GitHub, **Use this template** → new repository. Pick the visibility. Clone it.
2. `python _template/init_project.py <Name> --description "..."` — fills in every placeholder and
   bakes this file's SHA-256 into the compliance test.
3. Delete `_template/`.
4. `pip install -e ".[dev]"`, then `git submodule update --init`.
5. `python scripts/check.py` — must pass before the first commit.

An agent doing this follows `_template/INITIALISE.md`.

**"Use this template", not a fork.** A fork of a public repository can never be made private. And
the merge a fork promises does not work: the files a template change touches most are exactly the
ones the initialiser rewrote per project — `README.md`, `pyproject.toml`, `src/utils/paths.py`,
the hash line — so every pull conflicts where you least want to merge by hand.

### Changing a rule

1. Edit this file **at the source**; note it in the template's `docs/CHANGELOG.md`.
2. Update the check in `tests/test_template_compliance.py` and its violation in
   `_template/check_template.py`.
3. `python _template/check_template.py` — the new violation must be **caught**.
4. `python _template/sync_template_rules.py --apply` — copies this file into every project and
   re-bakes each hash. Reports by default; never commits.
5. Per project: `git diff`, `python scripts/check.py`, a line in its `CHANGELOG.md`.
