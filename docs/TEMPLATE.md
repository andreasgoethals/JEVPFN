# Andreas' repository template

The structure and rules every one of my research repositories starts from.

    Author   Andreas Goethals <andreas.goethals@kuleuven.be>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (Genius login, wICE and Mindwell compute)
    Source   https://github.com/andreasgoethals/0.-Template

**A starting point, not a contract.** A project that grows past it is fine, and deviating where
the work needs it is fine — deviate *on purpose* rather than by drift, and say so when you do.
What is written here is what has been worth having in every project so far. Anything
project-specific belongs in `README.md` or another `docs/` file, not here.

**Everything in the repository is explained below, one chapter per folder.**

```
├── config/                  YAML configuration
├── data/
│   ├── raw/                 inputs. Never modified, never committed.
│   └── processed/           generated cache, rebuildable
├── docs/                    ONLY .md files, names in CAPITALS
│   ├── TEMPLATE.md          this file
│   ├── CHANGELOG.md         what changed, newest first
│   ├── AGENTS_MEMORY.md     the runs done, and the dead ends
│   └── VSC.md               how THIS project runs on the VSC cluster
├── notebooks/               thin: all logic imported from src/
├── output/                  EVERYTHING the code generates
│   ├── All_Results.md       every notebook's printed summary
│   ├── figures/
│   │   ├── CAPTIONS.md      one shared captions file
│   │   └── <notebook>/      one PDF per figure
│   ├── logs/                the debugging record
│   ├── manifests/           what a run did
│   └── results/             the numbers a run produced
├── scripts/                 the experiments themselves
│   └── slurm/               the .slurm and .sh files that submit them
├── src/
│   ├── data/loaders.py      loading and preprocessing
│   ├── utils/
│   │   ├── paths.py         every path in the project
│   │   ├── clean_run.py     wipe the previous run
│   │   ├── run_notebooks.py run every notebook, rebuild both summaries
│   │   ├── update_tfm_library.py
│   │   ├── logging_setup.py
│   │   └── config.py        reading config/
│   ├── visualize/
│   │   ├── style.py         A4 sizes and print settings; the look is the project's
│   │   └── figures.py       FigureSaver
│   └── ...                  more as the project needs (train/, eval/, models/)
├── tests/
├── tfm-library/             submodule: literature + VSC docs. READ-ONLY.
├── .vscode/settings.json
├── .gitattributes  .gitignore  .gitmodules
├── AGENTS.md                the rules for every agent
├── CLAUDE.md                one line: `@AGENTS.md`
├── LICENSE
├── README.md
└── pyproject.toml
```

New `src/` subfolders need no permission — that is the obvious place to grow.

---

## `config/`

The project's YAML configuration. **Flat, or in folders and subfolders — however the project wants
it.** The template has no opinion beyond "configuration lives here".

`example.yaml` is a shape to copy; delete it once there are real configs.

---

## `data/`

- **`raw/`** — the inputs. Never modified, never committed (licences, size), and never deleted by
  any tool in the repository.
- **`processed/`** — the generated cache. Rebuildable from `raw/`, so losing it costs time and
  nothing else.

Both are gitignored except a tracked `.gitkeep`, so a fresh clone still has the directories.

A project that needs model weights adds **`checkpoints/`** at the root. `paths.checkpoints_dir()`
already resolves it to project storage on the cluster, `.gitignore` already excludes it, and
`clean_run` already refuses to touch it — weights are either downloaded from upstream or cost a
training run to reproduce.

---

## `docs/`

`.md` files only, names in CAPITALS, so a documentation file is recognisable from its name.

### `TEMPLATE.md`

This file. Copied in from the template. Generic rule changes belong at the source, not here.

### `CHANGELOG.md`

One chapter per date, `DD-MM-YYYY`, **newest at the top**. One bullet per change, **as short as
possible** — what changed, plus the why only when it is not obvious from the what. Most fit on one
line; a genuinely large change can take more, but no longer than it has to be. The detail belongs in
the commit, and a changelog nobody skims is a changelog nobody reads.

### `AGENTS_MEMORY.md`

What is worth carrying between sessions. The changelog records edits to the repository; this
records **experience**, and it has two halves:

- **Runs** — a table, one row per cluster run: date, the config or arm, the outcome (`done` /
  `walltime` / `OOM` / `crashed` / `diverged`), and one line of notes with the headline number or
  the output path. Projects iterate over dozens of runs, several of which do not work, and
  *"have we already tried that configuration?"* is a question nobody should answer by re-reading
  `output/results/`. A table because it stays scannable at forty rows.
- **Dead ends** — four lines each: **Tried**, **Result**, **Why**, **Instead**. For anything that
  cost more than a couple of minutes, including what you eventually fixed: the fix is one changelog
  line, the dead end was the hour.

Newest first, `DD-MM-YYYY`, and **short** — one line per run, four per dead end. An agent reads it
*before* starting and never deletes an entry: a run you would otherwise repeat and a dead end you
already paid for are both evidence.

It ships empty. Filling it is the project's job.

### `VSC.md`

How **this** project runs on the KU Leuven VSC cluster: which partitions and walltime limits it
uses, how to submit, how to resume a job that outlives the walltime, and how to get results back.
Built from the VSC documentation in `tfm-library/`, but written about this project.

**Storage — two tiers.** On both, everything lives inside a folder named after the project.

| tier | path | holds |
|---|---|---|
| **project storage** | `/lustre1/project/stg_00211/<Project>/` | big files: datasets, checkpoints, caches, **`output/results/`** |
| **personal data** | `$VSC_DATA/<Project>/` | the repository, and the rest of `output/` |

**Both are backed up.** They differ in size and in convenience: `$VSC_DATA` is only 75 GiB but you
can browse it directly, while project storage is large but has to be pulled down locally first
(PowerShell, `scp`/`rsync`) before you can look at anything in it. So the big, rarely-read things
go to project storage and everything you actually want to open stays on `$VSC_DATA`. Project
storage also has a **low inode budget** — few big files, not thousands of small ones.

`$VSC_SCRATCH` is purged after 30 days **without access**, and `mv` and timestamp-preserving
`rsync` do not count as an access. Compute nodes have **no outbound internet**, so anything that
downloads happens on a login node first. Any run that can exceed the walltime must be resumable,
writing its state pointer **last** so a job killed mid-write points at the previous complete
checkpoint.

---

## `notebooks/`

Thin. A notebook presents a finished computation; it does not perform one.

- **All logic lives in `src/`.** A notebook only calls it, and contains **no `def` and no
  `class`** — a function defined in a notebook cannot be imported or tested, so it gets copied into
  the next notebook and the copies diverge.
- Every notebook **ends by printing a text summary** of everything it showed, section by section,
  in the same order as its sections. That text is what `All_Results.md` is built from.
- **A notebook saves its own figures**, not the runner, so an interactive *Run All* produces
  exactly the same files. It clears **its own** figure folder — never another's — **before**
  drawing: a stale PDF beside a fresh one is how a paper ends up with a figure that no longer
  matches the code that made it.
- **A notebook never picks a colour or a figure size.** `src/visualize/style.py` owns both.
- Figures are **displayed inline** in the notebook, so the notebook itself is readable; the PDF on
  disk is what the paper uses.

`example_analysis.ipynb` is the pattern to copy. Delete it once a real notebook exists.

---

## `output/`

**Everything the code generates goes under `output/`**, locally and on the cluster. Not beside a
notebook, not into `src/`, not into a new top-level folder. One root means "what did this run
produce?" and "what can I delete?" each have one answer — and it is what makes
`src/utils/clean_run.py` possible at all.

### `logs/`

**`.log` files, and nothing else.** The debugging record, and the most important thing in here.

Runs happen on the supercomputer, where an agent cannot watch the job: the log is the only thing it
can read afterwards. So **log generously** — shapes, seeds, resolved paths, timings, intermediate
statistics, anything you would want when a run behaves oddly. Iterating on a run means reading the
log with an LLM, and what was not printed cannot be diagnosed.

One format, one extension, so "read the logs" is unambiguous. SLURM's own `.out`/`.err` files are a
different thing — SLURM writes them wherever the job script points, and after a requeue it is a
*different* file — which is exactly why the code writes its own `.log` here instead of relying on
them.

### `manifests/`

What a run actually did: the **resolved** configuration it used, progress or per-epoch records, the
environment. Small, and the answer to "what produced this?" six months later — the YAML on disk may
have been edited since.

### `results/`

The numbers a run produced: scores, per-fold metrics, per-row predictions. On the cluster this one
directory lives on **project storage**, because per-row predictions across every dataset and model
reach gigabytes and `$VSC_DATA` is 75 GiB. Locally it is a plain subdirectory.

### `figures/<notebook>/`

**One PDF per figure, nothing else.** Vector, with text embedded as TrueType so journal systems
accept it. No PNG on disk: the notebook displays its figures inline, so there is no second raster
copy to go stale. Sized for A4 — see `src/visualize/`.

### `figures/CAPTIONS.md`

**One** file for all notebooks, grouped per notebook, figures in the order that notebook drew them.

**These captions are the paper's captions.** They are written to be pasted under the figure in the
manuscript, so they are **pure description**: what is plotted, on what axes, from how much data. No
interpretation, no conclusion — the argument goes in the body text, and a caption that argues has
to be rewritten when the argument changes.

### `All_Results.md`

Every notebook's printed summary, in one file, because that one file answers "what did this project
find?". Its shape is fixed:

1. **One block per notebook, sorted alphabetically by notebook name.** A notebook whose name comes
   earlier in the alphabet appears earlier in the file.
2. **Each block is that notebook's printed summary, verbatim** — not a rewrite. The moment this
   file paraphrases, the two disagree and the notebook wins, but this is the file anybody reads.
3. **That summary follows the notebook's own section order**, so the file and the notebook read the
   same way round.

---

## `scripts/`

**The experiments themselves** — the things you actually submit and that run for hours on the
supercomputer: pretraining, a sweep, an evaluation pass, plus the `.slurm` and `.sh` files that
submit them.

The split is by *what a thing is for*, not by whether it happens to be runnable:

- **An experiment** — the work the paper is about — belongs here. It is invoked, watched, and
  charged to a credit account.
- **A utility** — cleaning up, running the notebooks, bumping the submodule pin — belongs in
  `src/utils/`, invoked with `python -m`. None of it is the experiment, and putting it here would
  bury the two files that matter among eight that do not.

`slurm/job.slurm` is a template with the walltime-resume pattern; copy it per experiment rather
than parameterising one script into unreadability. Everything in `slurm/` is **bash on Linux**, so
normal POSIX syntax — unlike everything you type locally.

---

## `src/`

All importable logic. Always has `data/`, `utils/`, `visualize/`; add more as the project needs.

**Most of these files ship empty**, as markers for something every project has but no two projects
implement the same way. A file is filled in only when the behaviour is genuinely identical
everywhere — and then it is identical, so do not rewrite it.

### `src/utils/`

- **`paths.py`** — *filled in.* **Every path in the project comes from here and nowhere else.** Two
  VSC tiers, one resolver, repo-root-relative, collapsing to the repository off-cluster so the same
  code runs on a laptop with no configuration. A path assembled at a call site with
  `"output/" + name` is correct on a laptop and wrong on the cluster.
- **`clean_run.py`** — *filled in.* `python -m src.utils.clean_run` lists what the previous run
  produced; `--clean` wipes it. Clears the **whole `output/` tree on both storage tiers** in one
  invocation, locally or on the cluster, and leaves the tracked `.gitkeep` markers so the next run
  has somewhere to write. `--processed` additionally clears `data/processed/`; that is opt-in
  because rebuilding the cache can cost far more than re-running the notebooks. `data/raw/`,
  weights and `tfm-library/` are never touched.
- **`run_notebooks.py`** — *filled in.* `python -m src.utils.run_notebooks` runs every notebook in
  parallel, in separate processes (matplotlib's figure registry is global), then rebuilds
  `CAPTIONS.md` and `All_Results.md`. Notebooks are **discovered**, alphabetically — a hard-coded
  list silently stops covering a notebook someone added.
- **`update_tfm_library.py`** — *filled in.* Bumps this project's submodule pin; see below.
- **`logging_setup.py`** — *filled in, minimal.* One logger, writing to the console **and** to
  `output/logs/`. On the cluster stdout is a SLURM file that moves on requeue; a log the code
  controls survives the job that produced it.
- **`config.py`** — *empty.* How a project reads its configuration is its own business. The one ask:
  write the **resolved** config into `output/manifests/`.

### `src/visualize/`

- **`style.py`** — *partly filled in.* The template fixes the part that follows from the output
  medium — the **A4 widths** and the print settings — and leaves the *look* to the project. Two
  projects plotting different things have no reason to look alike; what matters is that **every
  notebook inside one project shares one style**, defined here and nowhere else. A notebook never
  picks a colour or a size itself. Fill in `_PROJECT_RC` and whatever palette the project needs. If
  a new project is close to an existing one, copying that project's `style.py` is the fastest start.
- **`figures.py`** — *filled in.* `FigureSaver`: one folder per notebook, one PDF per figure with a
  numbered prefix, the folder cleared on construction, and each caption recorded in a manifest so
  `CAPTIONS.md` is rebuildable from disk without re-executing anything.

**Everything is sized for A4, because everything here is for a paper.** A figure is drawn at the
width it will occupy on the page — `WIDTH_FULL` (160 mm text block), `WIDTH_HALF`, `WIDTH_THIRD` —
and never rescaled in the document afterwards, because rescaling carries the text with it: 9pt
squeezed to 70 % arrives as 6.3pt, under the ~7pt floor for print. So the point sizes in `style.py`
are the point sizes **on the printed page**, and height is clamped to half a page so the caption
and the surrounding paragraphs still fit.

Two things worth deciding when you fill in the look: give a colour a **meaning** and keep it, so a
reader who has seen one figure can read the next without the legend; and check the palette is
readable **printed, and in greyscale**, because a paper gets photocopied.

### `src/data/`

- **`loaders.py`** — *empty.* Loading and preprocessing, which is project-specific by definition.
  Two asks: never build a path (use `paths.py`), and write a cache's marker file **last**, so a run
  killed halfway leaves a cache correctly treated as absent rather than silently reused.

---

## `tests/`

**Not one test per module — as many as the work needs.** Test what would be expensive to get wrong,
skip what is obvious. Tests never write outside `tmp_path` or `output/`.

`conftest.py` puts the repository root on `sys.path` (so the suite runs without an editable
install), forces matplotlib's `Agg` backend, and provides `isolated_output`, which redirects both
cluster tiers into `tmp_path` — exercising the branch that otherwise only runs in production.

The template ships tests for the files it ships filled in — `paths.py`, `clean_run.py`,
`run_notebooks.py`, `figures.py`, `style.py`. Those modules are identical in every project, so the
tests are too. Each says so in a comment at the top. Keep them; add your own beside them.

**Nothing runs them automatically.** There is no CI and no pre-commit hook in the template, so an
agent has to be told:

```
python -m pytest -q
```

Run it before delivering, and `python -m src.utils.run_notebooks` too if you touched a notebook or
anything under `src/visualize/`.

---

## `tfm-library/`

The shared TFM literature, as a **pinned git submodule**. In every project.

**What it is.** One curated knowledge base: the papers as PDFs with full-text extractions,
per-paper summaries, a cross-paper synthesis, flat-text snapshots of the upstream reference
implementations, and the VSC documentation. Maintained in its own repository, consumed by all.

**Why every project has it.** So a human *or an agent* can answer "what does the literature say?"
and "how does the official code actually do this?" **from inside the repository, offline, by
reading and grepping files** — no web search, no paywall, no recall from memory. It turns "I
believe X" into "X, see `tfm-library/<path>`". That is the point.

**Why a submodule.** It pins one exact commit, so a result stays reproducible against the
literature *as it stood*, and every project shares one maintained copy instead of several drifting
ones.

**READ-ONLY, one exception.** Never create, edit, move or delete anything inside it — anything
written there is lost when the pin moves, or corrupts a resource every project shares. The
exception is `tfm-library/PROJECT_SPECIFIC.md`, which the library gitignores for exactly this
purpose. If a library document is wrong, report it upstream rather than patching it here. Never
lint, format or test it.

**Citing it.** Papers by path (`tfm-library/papers/<year>/…`, full text under `papers/text/`).
**Code dumps by symbol name, never by line number** — the dumps are re-snapshotted and line numbers
drift by thousands. Record the pin next to any result that depends on the literature.

**A repository carries a real gitlink**, not just `.gitmodules`: that file records a path and a
URL, the gitlink (a tree entry of mode `160000`) records *which commit*. Without it,
`git submodule update --init` does nothing.

```
git submodule update --init                    # after a clone; empty until you ask (~749 MB)
git submodule status                           # which commit this project is pinned to
python -m src.utils.update_tfm_library         # bump the pin; reports first, --update to move it
```

`git submodule update --remote` moves the working tree but does **not** record the pin — a leading
`+` in `git submodule status` is that, not an error.

---

## Root files

- **`README.md`** — the project's own front page. It **ends** with the short "Based on the
  repository template" chapter and nothing after it; everything above is the project's.
- **`CLAUDE.md`** — one line, `@AGENTS.md`. Claude Code reads `CLAUDE.md` and not `AGENTS.md`,
  while every other agent (Codex, Cursor, Copilot, Windsurf) reads `AGENTS.md`. So `AGENTS.md` is
  the single source of truth and this file imports it — nothing duplicated, nothing to keep in sync.
  A symlink also works, but not on Windows without Administrator rights, so the import is the
  portable form. Claude-specific instructions, if ever needed, go below the import.
- **`AGENTS.md`** — the rules an agent follows here: follow this template and say when you deviate;
  `tfm-library/` is read-only; never commit data or weights; never install, train or push without
  asking; verify claims against a source rather than filling gaps plausibly; read
  `docs/AGENTS_MEMORY.md` first and add to it after a failure; Windows PowerShell 5.1 has no `&&`,
  so one command per line.
- **`LICENSE`** — MIT in the author's name, plus a line stating that third-party material (the
  submodule, datasets, downloaded weights) keeps its own licence.
- **`pyproject.toml`** — Python **3.11–3.12**, `ruff` and `pytest` in a `dev` extra, `tfm-library/`
  excluded from ruff, pytest collection and the package. Every non-obvious pin says **why**.
- **`.gitignore`** — never commit raw data, weights, caches, environments, tool caches or figures.
  **Anchor a rule with a leading slash** when a name should only match at the root: a bare
  `figures/` also matches `output/figures/`.
- **`.gitattributes`** — `* text=auto eol=lf`. The dev machine is Windows and every cluster job is
  Linux bash; a CRLF `.slurm` file fails with a bare `$'\r': command not found` that reads like a
  script bug and is not one.
- **`.gitmodules`** — declares `tfm-library`.
- **`.vscode/settings.json`** — committed, and not a preference: `python.analysis.extraPaths` is
  what lets an editor resolve `from src.utils…` given the flat `src/` package root, and
  `jupyter.notebookFileRoot` makes an interactive notebook run from the same directory as the
  notebook runner.

---

## Comments

Every non-obvious decision carries a short comment saying **why**, not what: a pin, a fallback, an
exclusion, a magic number, an ordering that matters. Say what breaks if it changes. Comments that
restate the code are noise.
