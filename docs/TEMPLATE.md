# Andreas' repository template

The structure and rules every one of my research repositories starts from.

    Author   Andreas Goethals <andreas.goethals@kuleuven.be>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (wICE login service, wICE and Mindwell compute)
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
├── output/                  generated reports, logs and results
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
│   │   ├── atomic.py        replace complete text reports and metadata
│   │   ├── clean_run.py     wipe the previous run
│   │   ├── run_notebooks.py run every notebook, rebuild both summaries
│   │   ├── update_library.py
│   │   ├── logging_setup.py
│   │   └── config.py        reading config/
│   ├── visualize/
│   │   ├── style.py         A4 sizes and print settings; the look is the project's
│   │   └── figures.py       FigureSaver
│   └── ...                  more as the project needs (train/, eval/, models/)
├── tests/
├── _template/               setup/adoption guide and initializer; remove in new projects
├── vsc-library/             submodule: HPC/VSC guides and templates. READ-ONLY.
├── .vscode/settings.json
├── .github/dependabot.yml   proposes VSC submodule pin updates
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

Newest first, `DD-MM-YYYY`, and **as short as possible without losing useful evidence**. An agent reads it
*before* starting and never deletes an entry: a run you would otherwise repeat and a dead end you
already paid for are both evidence.

The run table starts empty. Record the new project's runs and dead ends as work begins.

### `VSC.md`

How **this** project runs on the KU Leuven VSC cluster: which partitions and walltime limits it
uses, how to submit, how to resume a job that outlives the walltime, and how to get results back.
Built from the VSC documentation in `vsc-library/`, but written about this project.

**Storage — two tiers.** On both, everything lives inside a folder named after the project.

| tier | path | holds |
|---|---|---|
| **project storage** | `<configured staging root>/<Project>/` | big files: datasets, checkpoints, caches, **`output/results/`** |
| **personal data** | `$VSC_DATA/<Project>/` | the repository, and the rest of `output/` |

**Personal DATA is backed up; verify the allocation-specific project-storage protection policy.** They differ in size and in convenience: `$VSC_DATA` is only 75 GiB but you
can browse it directly, while project storage is large but has to be pulled down locally first
(PowerShell, `scp`/`rsync`) before you can look at anything in it. So the big, rarely-read things
go to project storage and everything you actually want to open stays on `$VSC_DATA`. Project
storage also has a **low inode budget** — few big files, not thousands of small ones.

Set `<PROJECT_UPPER>_STAGING_ROOT` (or `VSC_STAGING_ROOT`) to the allocated storage
directory; the resolver appends the project name. On VSC, missing configuration raises an
error rather than guessing an allocation. Locally, unset roots use the repository;
relative overrides are resolved against its root.
Set cluster overrides inside the job or its sourced project environment file: Leuven does
not propagate arbitrary exports from the submission terminal. Use a login-shell Bash script
(`#!/bin/bash -l`) and load the target node's compatible modules explicitly.

`$VSC_SCRATCH` is purged after 30 days **without access**, and `mv` and timestamp-preserving
`rsync` do not count as an access. Do not rely on outbound internet from compute nodes;
stage dependencies using the current VSC guidance. Any run that can exceed the walltime must be resumable,
writing its state pointer **last** so a job killed mid-write points at the previous complete
checkpoint.

#### Speed-first execution: Open OnDemand and Slurm

**Optimize elapsed time to verified results, not minimum token usage or compute credits.**
Use the VSC's parallel capacity, especially for CPU tests, preprocessing and evaluation.
Extra orchestration and resources are worthwhile when they shorten turnaround within the
approved budget. This is a project preference, not permission to submit unapproved work.

**Open OnDemand is an interface, not extra compute capacity or queue priority.** Its login
shell is a shared login node; interactive apps run inside scheduled allocations. Use an editor
session for editing/debugging and submit long campaigns as batch jobs, rather than running
them serially inside that session. End unused allocations; closing a browser tab is not a
cancellation. See [VSC's interactive-app guidance](https://docs.vscentrum.be/compute/portal/ondemand/interactive-apps.html).

- **Split independent CPU work by default.** Prefer ten balanced roughly two-hour jobs over
  one twenty-hour job when the work is separable. Partition datasets, folds, seeds, configs
  or test IDs exactly once; preserve scientific splits and seeds. Check the shard union for
  missing/duplicate work and merge only validated results. Balance by measured duration,
  not just item count. Explain any necessary serial stage. Two hours is an example, not a
  fixed target or a promise about queue time.
- **Submit ready work together after a representative preflight.** Prefer arrays for similar
  resource needs; separate CPU/GPU/memory classes. `--array=0-9%10` makes ten tasks eligible
  concurrently; the application must use `SLURM_ARRAY_TASK_ID` to select its shard. Do not
  launch the whole workload ten times. Choose the highest useful concurrency allowed by
  the account/QoS and measured I/O/RAM limits, not an arbitrary small cap. Array elements
  count toward job limits; `%N` limits running tasks, not submitted ones. Submit bounded
  waves if the submission limit is reached. [Slurm arrays](https://slurm.schedmd.com/job_array.html).
- **Minimize queue + setup + execution + merge time.** Request realistic walltime with
  measured setup/checkpoint margin, not the partition maximum. Shorter suitable jobs can
  backfill gaps; never under-request time or memory just to queue sooner. Batch very short
  work items together so startup and filesystem overhead do not dominate. Compare eligible
  wICE/Mindwell CPU partitions using actual availability and compatible environments, not
  hardware names alone. [Backfill](https://slurm.schedmd.com/sched_config.html),
  [VSC task granularity](https://docs.vscentrum.be/compute/jobs/worker_or_atools.html).
- **Use allocated CPUs effectively.** A Python process with workers normally needs one Slurm
  task and appropriate `--cpus-per-task`; extra `--ntasks` does not parallelize serial code.
  Bound worker processes and BLAS/OpenMP threads together to that CPU budget; avoid nested
  full-core pools. Request measured host RAM: excess memory can reserve extra cores.
  GPUs/multiple nodes help only software that actually uses them.
- **Prepare once, then fan out safely.** Stage immutable inputs on the native filesystem
  (Lustre for wICE, GPFS for Mindwell); use job-local scratch where appropriate. Keep heavy
  CPU preparation off GPU allocations. Give shards distinct logs, temporary files and
  outputs; never let them race on one SQLite database, cache or checkpoint. Merge through
  one writer after successful same-cluster `afterok` dependencies. Verify cross-cluster
  hand-offs explicitly. Preserve completed expensive work and retry only missing/failed
  shards. [VSC storage guidance](https://docs.vscentrum.be/leuven/tier2_hardware/parallel_filesystems.html).
- **Measure and adjust.** Record queue/run time, CPU utilization, peak RAM, I/O waits,
  throughput, shard counts and merge time. Increase concurrency while it reduces total
  turnaround; fix a serial merge or shared-I/O bottleneck rather than just adding jobs.
  Pending jobs are not failures: inspect the reason, do not resubmit duplicates or poll
  Slurm in a tight loop. Check credits and current limits before a large submission.

Useful read-only checks from a login shell (select the actual cluster):

```bash
sinfo --clusters=wice
squeue --clusters=all --me
sacctmgr show qos normal,long,debug,interactive format=Name,MaxSubmitJobsPerUser,MaxTRESPerUser
scontrol --clusters=wice show config | grep MaxArraySize
sacct --clusters=all -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ExitCode
sam-balance
```

Inspect job-step rows for memory/CPU statistics. Account-specific limits may further restrict
the QoS ceilings; do not hard-code a universal job count. Record the chosen shard map,
concurrency, resources, measured timings and any serial bottleneck in the project's `docs/VSC.md`.
Site rules and commands: [Leuven Slurm](https://docs.vscentrum.be/leuven/slurm_specifics.html),
[credit accounting](https://docs.vscentrum.be/leuven/slurm_accounting.html).
Reviewed 28-10-2026 against official sources and VSC Library pin
`1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87`; recheck live limits before submitting.

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

**Reports, logs and results go under `output/`**, locally and on the cluster. Not beside a
notebook, not into `src/`, not into a new top-level folder. One root means "what did this run
produce?" and "what can I delete?" each have one answer — and it is what makes
`src/utils/clean_run.py` possible at all. Rebuildable caches belong in `data/processed/`;
weights and costly resumable state belong in `checkpoints/`, outside routine cleanup.

### `logs/`

**`.log` files, and nothing else.** The debugging record, and the most important thing in here.

Runs happen on the supercomputer, where an agent cannot watch the job: the log is the only thing it
can read afterwards. So **log generously** — shapes, seeds, resolved paths, timings, intermediate
statistics, anything you would want when a run behaves oddly. Iterating on a run means reading the
log with an LLM, and what was not printed cannot be diagnosed.

The example Slurm script also directs stdout and stderr here as `.log` files and appends across
restarts. Create the directory before submission. Application logs should still record the run
identity, resolved configuration and progress independently of scheduler output.

### `manifests/`

What a run actually did: the **resolved** configuration it used, progress or per-epoch records, the
environment. Small, and the answer to "what produced this?" six months later — the YAML on disk may
have been edited since.

### `results/`

The numbers a run produced: scores, per-fold metrics, per-row predictions. On the cluster this one
directory lives on **project storage**, because per-row predictions across every dataset and model
reach gigabytes and `$VSC_DATA` is 75 GiB. Locally it is a plain subdirectory.

### `figures/<notebook>/`

**One PDF per figure**, plus `FigureSaver`'s `_figures.json` caption metadata and retained
`_stdout.txt` summary. A `_failure.txt` marks an incomplete runner execution. Vector, with text embedded as TrueType so journal systems
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

`slurm/job.slurm` is a minimal launcher requiring an explicit compute command and project-measured
CPU/memory/walltime settings; it does not automatically shard work. Checkpointing and
requeue need a tested project-specific protocol; see `docs/VSC.md`. Everything in `slurm/` is **bash on Linux**, so
normal POSIX syntax — unlike everything you type locally.

---

## `src/`

All importable logic. Always has `data/`, `utils/`, `visualize/`; add more as the project needs.

`config.py` and `data/loaders.py` are placeholders. The other modules provide reusable defaults;
adapt them when the project's needs differ. `__init__.py` files mark Python packages.

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
  weights and library submodules are never touched. Preview lists individual files and validates
  all targets before deletion; links and Windows junctions are refused. Stop writers before
  cleaning. Keep paid responses and costly/resumable artifacts outside disposable targets.
- **`atomic.py`** — writes UTF-8 text to a sibling temporary file, then replaces the destination.
  Reports and figure metadata use it so interrupted writes do not expose truncated files.
  It provides neither locking nor a multi-file transaction.
- **`run_notebooks.py`** — *filled in.* `python -m src.utils.run_notebooks` runs every notebook in
  parallel, in separate processes (matplotlib's figure registry is global), then rebuilds
  `CAPTIONS.md` and `All_Results.md`. Notebooks are **discovered**, alphabetically — a hard-coded
  list silently stops covering a notebook someone added.
  The lightweight runner supports top-level Python notebooks, not Jupyter magics or shell
  commands (except `%matplotlib inline`, ignored headlessly). It captures the final nonempty
  code cell's stdout, keeps it for rebuilding, and marks failed/incomplete runs in both reports.
  Partial reruns preserve other discovered notebooks' last reports. Use a project-specific
  Jupyter runner if kernel features or nested notebooks are needed.
- **`update_library.py`** — *filled in.* Bumps this project's submodule pin; see below.
- **`logging_setup.py`** — *filled in, minimal.* One logger, writing to the console **and** to
  `output/logs/`. Scheduler stdout follows the submission settings; a log the code
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

The supplied tests cover path resolution, cleanup boundaries, atomic replacement, notebook
reports, PDF figures, A4 sizing, library updates and initialization. Their header comments
explain the intent. Keep relevant tests and adapt them alongside project changes.

**Nothing runs them automatically.** There is no CI and no pre-commit hook in the template, so an
agent has to be told:

```
python -m pytest -q
```

Run it before delivering, and `python -m src.utils.run_notebooks` too if you touched a notebook or
anything under `src/visualize/`.

---

## Shared library submodules

The template includes **VSC Library only**, at `vsc-library/`. Initialize it after cloning:

```powershell
git submodule update --init -- vsc-library
git submodule status
```

Libraries are independent repositories. Add only those the project needs:

| Library | Repository | Mountpoint |
|---|---|---|
| VSC infrastructure | [VSC Library](https://github.com/andreasgoethals/VSC_Library) | `vsc-library` |
| Tabular foundation models (optional) | [TFM Library](https://github.com/andreasgoethals/TFM_Library) | `tfm-library` |
| Decision models (optional) | [DM Library](https://github.com/andreasgoethals/DM_Library) | `dm-library` |

**Purpose:** TFM and DM provide paper texts, summaries, syntheses and upstream implementation
snapshots for agents to inspect in place. VSC provides infrastructure documentation and Slurm
examples. They are references, not project runtime packages; read them, do not install them.

From the project root, run only the additions you need and which are not already registered:

```powershell
# Included in new template projects; add only when missing from an existing project.
git submodule add -b main https://github.com/andreasgoethals/VSC_Library.git vsc-library
# Independent opt-ins, not a requirement to add both.
git submodule add -b main https://github.com/andreasgoethals/TFM_Library.git tfm-library
git submodule add -b main https://github.com/andreasgoethals/DM_Library.git dm-library
```

For an already registered library, use `git submodule update --init -- <mountpoint>` instead.
Then review `.gitmodules` and `git diff --submodule=log`, and publish only the intended paths
when authorized. For example, after adding VSC Library:

```powershell
git add -- .gitmodules vsc-library
git commit --only -m "Add VSC Library reference" -- .gitmodules vsc-library
git push
```

Substitute the chosen mountpoint for optional libraries. Adding one library never installs
the others. Keep added mountpoints excluded from packaging,
linting and test discovery; the three paths above are already excluded here.

The `.gitmodules` registration records the URL; the committed gitlink records the
exact library revision. Read shared documents and snapshots in place, cite source
paths and symbols, and record the pin beside results that depend on them.

**Consumer copies are read-only.** Never edit, refresh, lint, format or test inside
them. VSC has no local-file exception: put project-specific operations in
`docs/VSC.md`. TFM and DM permit only their ignored `PROJECT_SPECIFIC.md`, created
from their own `PROJECT_SPECIFIC.template.md`. Corrections belong in the canonical
library checkout. These ignored local notes are not carried by the parent repository;
keep durable shared project decisions in its own `docs/`.

The owner's canonical checkouts live together in the local `0. Literature`
workspace. Its `scripts/maintain.py` runs each library's maintenance in parallel.
After reviewing, committing and pushing library changes, run from that workspace:

```powershell
python scripts/propagate_to_downstream.py --dry-run
python scripts/propagate_to_downstream.py --push
```

Propagation discovers existing consumers through their submodule URLs, including
this template. It pins the exact clean published library commit, commits only the
submodule path and optionally pushes. It never adds an unregistered library.
A new consumer must be inside the searched Projects tree or supplied through
`--root`. The template's daily Dependabot configuration proposes VSC pin updates
on GitHub; proposals require review and merging.

For a single consumer, the local helper reports the current pin by default:

```powershell
python -m src.utils.update_library
python -m src.utils.update_library --library vsc-library --update --commit
git push
```

Without that helper, use `git submodule update --init -- <mountpoint>` to initialize the
recorded pin, then `git submodule update --remote -- <mountpoint>` to advance it, review
`git diff --submodule=log`, and commit only that mountpoint when authorized. Never overwrite
local submodule changes. Pin updates are distinct from editing/refreshing library content.
Review the library changelog before updating. Publishing a library alone does not
change a consumer's committed pin; run propagation or merge an update proposal.

---

## Root files

- **`README.md`** — the project's own front page. It **ends** with the short "Based on the
  repository template" chapter and nothing after it; everything above is the project's.
- **`CLAUDE.md`** — one line, `@AGENTS.md`. Claude Code reads `CLAUDE.md` and not `AGENTS.md`,
  while Codex reads `AGENTS.md` directly. Other tools may need explicit configuration. `AGENTS.md` is
  the single source of truth and this file imports it — nothing duplicated, nothing to keep in sync.
  A symlink also works, but not on Windows without Administrator rights, so the import is the
  portable form. Claude-specific instructions, if ever needed, go below the import.
- **`AGENTS.md`** — the rules an agent follows here: follow this template and say when you deviate;
  library submodules are read-only; never commit data or weights; never install, train or push without
  asking; verify claims against a source rather than filling gaps plausibly; read
  `docs/AGENTS_MEMORY.md` first and add to it after a failure; Windows PowerShell 5.1 has no `&&`,
  so one command per line.
- **`LICENSE`** — MIT in the author's name, plus a line stating that third-party material (the
  submodule, datasets, downloaded weights) keeps its own licence.
- **`pyproject.toml`** — Python **3.11–3.12**, `ruff` and `pytest` in a `dev` extra, library submodules
  excluded from ruff, pytest collection and the package. Every non-obvious pin says **why**.
- **`.gitignore`** — never commit raw data, weights, caches, environments, tool caches or figures.
  **Anchor a rule with a leading slash** when a name should only match at the root: a bare
  `figures/` also matches `output/figures/`.
- **`.gitattributes`** — `* text=auto eol=lf`. The dev machine is Windows and every cluster job is
  Linux bash; a CRLF `.slurm` file fails with a bare `$'\r': command not found` that reads like a
  script bug and is not one.
- **`.gitmodules`** — declares `vsc-library`; optional libraries are added explicitly.
- **`.vscode/settings.json`** — committed, and not a preference: `python.analysis.extraPaths` is
  what lets an editor resolve `from src.utils…` given the flat `src/` package root, and
  `jupyter.notebookFileRoot` makes an interactive notebook run from the same directory as the
  notebook runner.

---

## Comments

Every non-obvious decision carries a short comment saying **why**, not what: a pin, a fallback, an
exclusion, a magic number, an ordering that matters. Say what breaks if it changes. Comments that
restate the code are noise.

## `_template/`

Template-only setup, removed after initialization: `README.md` explains creating or adopting
a project; `init_project.py` substitutes names and metadata after validating all replacements.
It skips submodules, linked directories and generated files and performs no installs or Git writes.

## `.github/`

`dependabot.yml` proposes daily updates to the registered VSC Library pin. It does not
run tests or merge updates; review proposals before accepting them.
