# `_template/` — the parts that do not travel

**If you are in a project: delete this folder.** It is the template's own machinery. Nothing
outside it depends on it, and nothing inside it is useful once the repository has become a
project.

```powershell
Remove-Item -Recurse -Force _template
```

**If you are in the template repository:** this file is the manual. It explains how the
template works, what every module in it does, and how to change a rule without breaking the
five projects downstream.

---

## The idea

**This repository is not a generator. It *is* a project — with the name left blank.**

Every folder, every module, every rule is already in its final place. A new project is this
repository with `{{PROJECT_NAME}}` filled in and this folder removed. There is no `skeleton/`
directory, no copying step, and no second copy of anything to keep in sync — which means the
`.gitignore`, the compliance test and the figure pipeline are all exercised *in place*, so a
mistake in them shows up here rather than in a project six months from now.

```
Use this template  (GitHub)                    pick the visibility; the submodule pin comes too
        |
        v
clone it, open in VS Code                      .vscode/settings.json is committed, so
        |                                      `from src...` resolves immediately
        v
point an agent at _template/INITIALISE.md      it asks for the four facts it needs
        |
        v
python _template/init_project.py CreditICL --description "..."
        |                                      fills in the blanks, bakes the hash
        v
Remove-Item -Recurse -Force _template          nothing in the project imports from it
        |
        v
git submodule update --init                    fetches the literature (749 MB)
        |
        v
python scripts/check.py                        must pass before the first commit
```

### Why not a fork

A fork's appeal is `git remote add upstream …` and then `git pull upstream main` to bring a rule
change down as a merge instead of a file copy. Two reasons not to, and the second is the bigger
one:

- **A fork of a public repository can never be made private.** GitHub does not allow changing a
  fork's visibility. "Use this template" lets the new repository be private even when the
  template is public — which a project under embargo, or one with licence-restricted data, needs.
- **The merge does not actually work.** The files a template change touches most are exactly the
  ones `init_project.py` rewrote per project: `README.md`, `pyproject.toml`,
  `src/utils/paths.py`, and the hash line in `tests/test_template_compliance.py`. Every pull is
  therefore a conflict in the files you least want to merge by hand. A fork also puts the
  template's commits at the root of the project's history, and GitHub defaults a pull request
  from a fork to the *upstream* — so the obvious button proposes the project's work to the
  template.

`sync_template_rules.py` does the propagation instead: one command, from the template, into
every project, with the hash re-baked. That removes the only real argument for forking.

## What is in here

| File | What it does |
|---|---|
| [`INITIALISE.md`](INITIALISE.md) | **The brief an agent follows** in a fresh fork: the facts to ask the user for, the one command, the files only judgement can fill, and what to report. Point an agent at this. |
| [`init_project.py`](init_project.py) | Fills in every `{{PLACEHOLDER}}` and writes `docs/TEMPLATE.md`'s SHA-256 into `tests/test_template_compliance.py`. Runs nothing else — no install, no git, no push, and it does not delete this folder for you. |
| [`check_template.py`](check_template.py) | Answers *"does this template still produce a healthy project?"* by making one in a temp directory and checking it — then injecting every rule violation into fresh copies and confirming the compliance test **fails** on each. Run it after any change here. |
| [`sync_template_rules.py`](sync_template_rules.py) | Pushes a changed rule down into every existing project: copies `docs/TEMPLATE.md` and re-bakes each project's hash. Reports by default. This is what replaces the merge a fork would have given you. |
| [`sync_tfm_library.py`](sync_tfm_library.py) | Bumps the `tfm-library/` pin across **every** project at once. Reports by default; `--only` / `--exclude` / `--pick` choose which. Never pushes. |
| [`_projects.py`](_projects.py) | Shared by both `sync_*` tools: finds the sibling projects and handles the selection flags. One copy, because two copies of a filesystem walk drift. |
| `README.md` | This file. |

Everything else in the repository is the project and travels with it.

### `init_project.py`

```powershell
python _template/init_project.py CreditICL
python _template/init_project.py CreditICL --dry-run
python _template/init_project.py CreditICL --repo-url https://github.com/me/CreditICL
```

Placeholders, all derived from the project name unless overridden:

| Placeholder | Becomes | Where it matters |
|---|---|---|
| `{{PROJECT_NAME}}` | `CreditICL` | the folder name on **both** cluster tiers, prose everywhere |
| `{{PACKAGE_NAME}}` | `crediticl` | `pyproject.toml` name, SLURM job name |
| `{{PROJECT_UPPER}}` | `CREDITICL` | the `CREDITICL_STAGING_ROOT` environment-variable prefix |
| `{{DESCRIPTION}}` | one sentence, from `--description` | `README.md`'s opening line, `pyproject.toml` description, `CITATION.cff` abstract — three places that must not drift apart |
| `{{AUTHOR}}` `{{EMAIL}}` | you | `LICENSE`, `pyproject.toml`, `CITATION.cff`, `AGENTS.md` |
| `{{YEAR}}` `{{DATE}}` `{{DATE_ISO}}` | today | `LICENSE`; `DD-MM-YYYY` for the logs; ISO only for `CITATION.cff` |
| `{{REPO_URL}}` `{{TFM_LIBRARY_URL}}` | the remotes | `pyproject.toml`, `.gitmodules` |
| `{{TEMPLATE_SHA256}}` | the hash | `tests/test_template_compliance.py` |

`{{REPO_URL}}` is **auto-detected from this checkout's `origin`** before falling back to a
name-based guess. That is what makes the fork workflow need no flag: after cloning a fork,
`origin` is already the right answer — and it stays right when the repository name differs from
the project name, which a guess built from the project name never is. SSH remotes are normalised
to https so the value that lands in `pyproject.toml` is clickable.

`init_project.py` also **reports the submodule state** — `MISSING` (no gitlink recorded),
`pinned, empty` (the normal state after a clone), or `populated` — and prints the command for
that specific case. Three situations that look identical from the outside and need different
fixes.

The regex is `(?<!\$)\{\{([A-Z][A-Z0-9_]*)\}\}` — upper case only, never after a `$`. That is
what keeps it away from GitHub Actions expressions like `${{ matrix.python-version }}`, which
look identical to a careless substitution. A leftover placeholder after a run is reported as a
**template bug**, because it means a key exists in a file that `derive()` does not produce.

### `check_template.py`

```powershell
python _template/check_template.py
python _template/check_template.py --fast     # skip the notebook run
python _template/check_template.py --keep     # leave the temp copies to inspect
```

Three things, in order:

1. `scripts/check.py` in a freshly initialised copy — ruff, every import, pytest.
2. `scripts/run_notebooks.py` in that copy, **and a check that it actually produced PDFs,
   `CAPTIONS.md` and `All_Results.md`**. A runner that exits 0 having drawn nothing is the
   failure mode worth guarding against.
3. Every rule from `docs/TEMPLATE.md`, injected one at a time into its own fresh copy, with the
   compliance test required to **fail**.

Step 3 is why the script exists. A compliance test that has only ever been seen to pass is
decoration. **Every rule you add to `docs/TEMPLATE.md` gets an entry in `VIOLATIONS`** — and if
you cannot make the test fail, the rule is not enforced, whatever the document says.

### `sync_template_rules.py`

```powershell
python _template/sync_template_rules.py                   # what is out of date, where
python _template/sync_template_rules.py --diff            # show what actually differs
python _template/sync_template_rules.py --apply           # docs/TEMPLATE.md + the hash
python _template/sync_template_rules.py --apply --also scripts/check.py
python _template/sync_template_rules.py --only CreditICL --apply
```

Two tiers, and the split is about **who owns a file once the project exists**:

- **Applied** — `docs/TEMPLATE.md`, plus its SHA-256 in `tests/test_template_compliance.py`.
  Overwriting is safe without asking, because the template owns that file verbatim and the
  compliance test already proves the project has not edited it: if it had, that project's suite
  would be red.
- **Reported** — every other file the template ships unchanged (`scripts/check.py`,
  `src/utils/run_artifacts.py`, the module tests, …). Shown as differing, never written, because
  a project is *allowed* to extend some of them and *expected* to delete others (the example
  notebook, the example config). Look with `--diff`, then take one deliberately with `--also`.
- **Never compared** — anything the initialiser rewrote, plus `src/utils/paths.py` and
  `src/visualize/style.py`, where "differs from the template" is the correct state.

The tool skips the template repository itself, writes files without committing, and prints the
review steps: `git diff`, `python scripts/check.py`, a line in that project's changelog.

### `sync_tfm_library.py`

```powershell
python _template/sync_tfm_library.py                        # report on all of them
python _template/sync_tfm_library.py --only CreditICL CreditPFN
python _template/sync_tfm_library.py --exclude TabPFN       # all but these
python _template/sync_tfm_library.py --pick                 # numbered list, choose
python _template/sync_tfm_library.py --update               # move the working trees
python _template/sync_tfm_library.py --update --commit      # ...and record the pins
```

Finds every git repository two levels under the projects directory — so `Projects/4. CreditICL/CreditICL`
is found — and keeps the ones whose `.gitmodules` declares `tfm-library`. Nothing is
hard-coded, so a new project appears the moment it has the submodule.

`--pick` prints the numbered list and reads a selection: nothing for all, `2,4` for only those
two, `-3` for everything except the third.

Three separate gears, on purpose. `--update` moves a working tree; `git submodule update
--remote` does **not** record the pin, and the whole reason the literature is pinned is that a
result stays reproducible against the sources as they stood when it was produced. Moving that
silently across five repositories is exactly what the pin exists to prevent. It commits with a
`-- tfm-library` pathspec so it can never sweep in unrelated staged work, refuses outright in a
repository that has other things staged, and never pushes.

---

## The rules

[`docs/TEMPLATE.md`](../docs/TEMPLATE.md) is the **one governing document**. Every rule and
every piece of structure is written there; `README.md`, `AGENTS.md` and the tests point back to
it rather than restating it. If two documents disagree, that one wins.

It is edited **only here, in the template repository.** Its SHA-256 is baked into every
project's compliance test, so a local edit downstream fails that project's test suite. That is
the mechanism, and it is deliberate: a rule asking politely not to be edited is not a rule.

### Changing a rule

1. Edit `docs/TEMPLATE.md` here, and note the change in `docs/CHANGELOG.md`.
2. Add or update the check in `tests/test_template_compliance.py`, and add its violation to
   `VIOLATIONS` in `check_template.py`.
3. `python _template/check_template.py` — the new violation must be **CAUGHT**.
4. `python _template/sync_template_rules.py --apply` — every project gets the new document and
   a re-baked hash.
5. In each project: `git diff`, `python scripts/check.py`, then a line in its `CHANGELOG.md`.

Step 4 is what makes the hash check affordable; step 5 is the point of it. A rule cannot drift
into a project silently, and it cannot be quietly ignored in one either.

---

## Reference — what every module does

The template ships working code, not stubs. This is what each piece is for and the reason it
is shaped the way it is. Function-level detail lives in the docstrings; this is the map.

### `src/utils/paths.py` — the resolver

**The only module that builds a path.** Everything else asks it. A path assembled at a call
site with `"output/" + name` is correct on a laptop and wrong on the cluster, and the failure
shows up as a full quota or an empty results directory hours into a job.

| Function | Returns |
|---|---|
| `on_vsc()` | whether `$VSC_DATA` is set, i.e. we are on a cluster node |
| `staging_root()` `data_root()` `scratch_root()` | the three roots, honouring overrides |
| `outputs_dir()` | **the** root for everything generated |
| `results_dir(*parts)` | the one part of `output/` on project storage — see below |
| `logs_dir()` `manifests_dir()` `figures_dir(nb)` | inside `outputs_dir()` |
| `captions_path()` `all_results_path()` | the two shared summary documents |
| `raw_dir()` `processed_dir()` `checkpoints_dir()` | inputs, cache, weights |
| `data_search_paths()` `find_input()` | read search order: **repo first**, then project storage |
| `config_path(name)` `notebooks_dir()` `library_dir()` | repository directories |
| `ensure(path)` | `mkdir -p`, returning the path |
| `resolve_writable(preferred, fallback)` | probes with a real write and falls back **loudly** |
| `touch_tree(path)` | refreshes access times against the scratch purge |
| `describe()` | every resolved root, for logging at job start |

Three decisions worth knowing:

- **`output/results/` is the one part of `output/` on project storage.** Per-row predictions
  across every dataset and model reach gigabytes and `$VSC_DATA` is 75 GiB — one sweep would
  fill it, and then every job that writes a log also fails. Off-cluster it is plain
  `output/results/`, so the split is invisible locally.
- **Off-cluster every tier collapses into the repository.** There is no `/lustre1` on a laptop,
  and pretending there is would mean two code paths — the one that only runs on the cluster
  being the one that breaks.
- **`resolve_writable` probes with a real create-and-delete.** `mkdir(exist_ok=True)` is not
  enough: a directory on a shared tier can exist and still be unwritable by you.

### `src/utils/config.py` — one YAML per experiment

`load` → `sweep_axes` → `n_points` → `expand`. `expand()` returns one flat dict per sweep
point, with the swept keys folded in at the top level so nothing downstream has to know whether
a knob came from the sweep or from the shared part. `resolved_dump()` writes what a run
actually used next to that run's output — the YAML may have been edited since, and a sweep
point is not in the YAML at all.

No inheritance and no includes, deliberately: a config file is read top to bottom and that is
the whole story. `get(cfg, "train.learning_rate")` is dotted access so a nested knob does not
need four `.get()` calls, each of which is a place to silently return `None`.

### `src/utils/logging_setup.py`

`setup()` once, `get_logger(__name__)` everywhere. Writes to stdout **and** to
`output/logs/`, because on the cluster stdout is a SLURM file that lands wherever the job
script put it — and after a requeue it is a *different* file.

### `src/utils/run_notebooks.py` — the runner

`discover()` → `run_one()` in parallel → `write_captions()` + `write_all_results()`.

- Notebooks are **discovered, not listed**, alphabetically. A hard-coded list is a list that
  silently stops covering a notebook someone added.
- **Separate processes, not threads.** matplotlib's figure registry is global; two notebooks in
  one interpreter would capture each other's figures, silently.
- **A flattened script, not a Jupyter kernel.** Nothing extra to install, identical on the
  cluster, and a traceback points at a line number instead of a cell index. Magics are stripped
  — a notebook that needs one cannot be executed non-interactively at all.
- **The runner does not save figures.** The notebooks do. See below.

### `src/visualize/figures.py` — `FigureSaver`

One folder per notebook; PDF (300 dpi) plus PNG (110 dpi); the folder cleared on construction,
**before** anything is drawn, and only ever that notebook's own; a numbered filename prefix so
alphabetical order is drawing order; a `_figures.json` manifest holding each figure's caption.

**Why the notebook saves its own figures.** A runner that captures them on the notebook's
behalf only works inside the runner — *Run All* in Jupyter, which is where figures are actually
iterated on, then produces nothing. Both paths run the same code here.

**Why captions are passed at `save()`** instead of kept in a central registry: they live next
to the figure they describe and cannot go stale when it is renamed. A figure saved without one
is listed in `CAPTIONS.md` with a loud placeholder rather than skipped.

### `src/visualize/style.py` — the shared look

`apply()` once per notebook. Then `color(name)` for every colour, and never a literal.

- **`COLORS`** — semantic roles that carry a *meaning*: `baseline`, `secondary_baseline`,
  `proposed`, `observed`, `alternative`, `highlight`, `annotation`. `baseline` is achromatic on
  purpose, so it sits outside the categorical palette: a reference condition should recede, and
  greying it says "this is the thing being improved on" without spending a hue.
- **`STATUS`** — `good` / `warning` / `serious` / `critical`, never reused as a series colour.
- **`SERIES`** — the eight categorical slots, in a **fixed order**.
- **`register_series()` + `color()`** — the mechanism that makes a legend mean one thing
  project-wide. Colour follows the entity, never its rank: if a figure drops a series,
  matplotlib's cycler would shift every colour after it and the same model would be blue in one
  figure and orange in the next. Declare the project's names once, in this module, and append —
  never insert, because inserting repaints every figure after it and silently invalidates the
  ones already in the paper.
- `sequential_cmap()` (one hue, magnitude), `diverging_cmap()` (two hues, neutral grey midpoint,
  for signed quantities), `ORDINAL` (discrete ordered steps with visible gaps).
- `figsize()`, `WIDTH_SINGLE`, `WIDTH_DOUBLE` — draw at final width. A figure scaled afterwards
  has the wrong font size: 8 pt text squeezed to 70 % arrives as 5.6 pt.
- `legend()`, `despine()`, `annotate_reference()`, `palette_table()`.

**The palette is computed, not chosen by eye.** On a white surface the categorical order clears
the OKLCH lightness band, a chroma floor, colour-vision-deficiency separation of every
*adjacent* pair (worst ΔE 9.2 deutan, OKLab ×100, target ≥ 8) and a normal-vision floor on
adjacent pairs (worst ΔE 20.8, floor 15). Three slots sit below 3:1 contrast against white
(aqua 2.8, magenta 2.7, yellow 2.2), which is why `legend()` is not optional and why bars carry
direct labels — identity must never rest on colour alone.

**The order is the safety mechanism, not decoration.** Reordering changes which pairs are
adjacent and can break the CVD gate. `MAX_SERIES_ALL_PAIRS = 4` because in scatter, bubble and
small-multiple forms every pair is on screen at once, not just neighbours, and only the first
four slots clear the floors under that condition — the fifth slot beside the second measures
12.9 to normal vision, genuinely hard to tell apart, colourblind or not. `color()` raises past
eight rather than generating a ninth hue, because a generated hue hides the fact that colour
has stopped working.

**One mode, deliberately:** a figure destined for a paper renders on white. There is no viewer
theme to follow, so there is no dark variant to keep in sync.

If you change a hex value or the order, **re-validate the whole set** before committing.

### `src/utils/run_artifacts.py` + `scripts/clean_run.py`

Lists by default, deletes with `--clean`, and `data/raw/`, `checkpoints/`, `tfm-library/` and
the repository's own directories are protected **by construction** — they are in
`protected_paths()`, not behind a flag, so no combination of arguments reaches them. The check
runs both directions, so passing a *parent* of something protected is refused too: deleting
`data/` to get at `data/processed` would take `data/raw` with it.

Cheap categories go by default; `results` and `processed` must be named, because someone typing
"clean up the logs" must not lose a week of finished runs. Directory *contents* are removed
rather than the directory, so the tracked `.gitkeep` markers survive — `rmtree` on `output/logs`
removes a directory git expects to exist.

### `src/data/loaders.py`

A shape to replace, with a contract to keep: never build a path; `data/raw/` is read-only; a
cache counts as valid only once its marker exists, and the marker is written **last** so a run
killed halfway leaves a cache correctly treated as absent rather than silently reused.

### `scripts/`

Only runnable entry points — every `.py` file directly in `scripts/` has an
`if __name__ == "__main__":` block, and the compliance test checks it. Something importable
there is untestable by construction and belongs in `src/`.

| Script | What it is for |
|---|---|
| `check.py` | **the** one command for "is this repository healthy?" — ruff, then an import of every `src` module, then pytest. CI runs this exact script, so the failure a reviewer sees is the one the author can reproduce. The import step exists because ruff parses files without importing them and pytest only imports what a test touches. |
| `run_notebooks.py` | run all notebooks, rebuild both summary documents. `--summaries-only` rebuilds them from disk after an interactive session. |
| `clean_run.py` | list, then delete. |
| `update_tfm_library.py` | this project's own pin, three gears, never pushes. |
| `slurm/job.slurm` | a job template with the SIGUSR1 walltime-resume pattern. |

### `tests/`

One file per `src` module, plus `test_template_compliance.py` — which reads files and **never
imports the project**, so it runs on a fresh clone before anything is installed. `conftest.py`
puts the repository root on `sys.path` (so the suite works with or without an editable install),
forces the `Agg` backend, and provides `isolated_output`, which redirects both cluster tiers
into `tmp_path` by setting the environment variables — exercising the branch that otherwise only
ever runs in production.

---

## The template repository is not installable, and that is fine

`pyproject.toml` has `name = "{{PACKAGE_NAME}}"`, which is not a valid PEP 508 name, so
`pip install -e .` fails **here**. It is not a package; it is a template. Everything that
matters still works, because `tests/conftest.py` puts the root on `sys.path`:

```powershell
python scripts/check.py
```

The compliance test's hash check **skips** while `src/utils/paths.py` still holds
`{{PROJECT_NAME}}` — there is nothing to compare against until a project exists. As soon as
`init_project.py` has run, it is enforced. `.github/workflows/check.yml` uses the same signal to
pick its install step, which is why one workflow file serves both the template and every
project made from it.

## Keep it project-agnostic

No dataset, no model, no experiment, no result — anywhere. If a rule cannot be stated without
naming one, it is not a template rule and it belongs in a project.

The seeded content in `docs/AGENTS_MEMORY.md` is the deliberate exception: those entries are
real dead ends, kept because they apply to every project. When a project hits one that turns
out to be universal, **promote it** — copy the entry here and every new project starts already
knowing.
