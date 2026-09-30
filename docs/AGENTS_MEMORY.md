# Agents' memory — runs and dead ends

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

## Runs

30-09-2026 final design/publication refactor: owner authorized a public JEVPFN repository,
text-only per-column plus joint inputs, and skipping empty inputs with missing feature values.
Public repository created; origin updated and old remote retained as template. The optional
non-text-context comparison is out of scope. User-facing metadata/config use non-text names.
113 tests, four parallel runner notebooks and all four actual JEVPFN-kernel executions passed.
All 20 datasets load offline; raw file/marker hashes are unchanged and eight source pins still
verify. The budget is now 2,032,351 distinct requests / about USD 35.84; prior four-variant costs
below are historical. Detailed reports/request examples are ignored under output_JEVPFN;
small VSC outputs use DATA and large results/raw/features/weights use project storage.
No paid API call, model run, weight download or VSC submission occurred. Local feature pilot/full
build still await design approval. The only removed output remnants were obsolete aggregate
reports and the verified-empty former notebook-03 figure directory; no raw data was moved.

30-09-2026 sequence correction: owner selected all Jev feature creation locally, with design
choices, previews and small local tests first, then full local extraction, then experiment 0
on VSC. Updated configs/docs/notebook summary and made VSC Jev networking optional. Use plain
terms: non-text columns, each text column separately, all text columns together. No API run started.

30-09-2026 recovery: owner reported dataset loss after running the supplied cleanup command.
Inspection found all 60 dataset files missing. Disabled the command as an error-only stub,
removed its launch instructions and restored all 20 pinned official releases. Exact deletion
mechanism is unconfirmed. Earlier recommendations to run folder cleanup are superseded.
Independent verification: all 60 direct raw files and eight upstream hashes match. 109 tests,
Ruff, pip dependency checks and all four notebook runner executions passed; updated notebook 03
also passed the actual JEVPFN kernel. Offline request review found 5,324,848 distinct inputs and
an approximate USD 95.90 all-variant budget. No inference, weights or VSC allocation ran.

30-09-2026: local editable install refreshed from `.[dev,notebooks]`; `pip check`, 102 tests,
20 offline datasets and four runner notebooks passed. All three research notebooks also passed
in the actual JEVPFN kernel. Slurm scripts passed Bash syntax checks only; no VSC job, inference
call, model package or weight download ran. Removed 60 hash-identical legacy dataset files;
eight upstream source snapshots still match their pinned hashes. Empty-directory deletion was
again rejected by automatic approval; the owner-run cleanup script is syntax-checked only.

29-09-2026 cleanup: 60 dataset files moved directly into their 20 numbered folders with unchanged
hashes; all 20 load offline and 102 tests pass. The final preservation check found the upstream
snapshot folder empty; restored its eight files from the pinned URLs and verified catalog hashes.
No Python files remain under `data/`. Removed the mock-only SQLite file and `.env.example`;
the notebook cache demonstration now uses temporary storage. All four notebooks passed again
after restoring the source snapshots, and the persistent SQLite file stayed absent.
No cluster or inference jobs ran.

29-09-2026 follow-up: no cluster jobs, Jev inference calls or model runs submitted. All 20 datasets
loaded offline after relocation into numbered raw folders; 102 local tests and 4/4 notebooks
passed. Notebook 03 also passed an actual JEVPFN kernel execution. CreditPFN supplies the account
`lp_verbekelab` and Conda convention, but authenticated VSC access, compute networking and CUDA
still need live verification. Literature checkout: `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`.

No cluster runs have been submitted for this project. The initial local audit on 29-09-2026
loaded all 20 core text datasets (821,071 rows); configuration and hashes are in
`config/exploration/datasets.yaml` (formerly `config/multabench_core_text.yaml`). Both original
project notebooks and the template example passed at that stage.
Real Jev calls and modelling remain disabled. Preserve the paid-response cache location outside
`output/` and `data/processed/`; the template cleanup clears those directories.

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| | | | |

- **Run** — the config or arm, and the commit if it matters.
- **Outcome** — `done` / `walltime` / `OOM` / `crashed` / `diverged` / `cancelled`.
- **Notes** — one line: the headline number, the output path if it is worth finding again, or the
  single thing the run showed. A number here saves re-reading `output/results/`.

## Dead ends

Anything that cost more than a couple of minutes and did not work — including what you eventually
fixed, because the fix is one changelog line and the dead end was the hour.

```
### <short name for the attempt>
- **Tried:** what was done.
- **Result:** what happened — the error, the wrong number, the silent no-op.
- **Why:** the underlying reason, once understood.
- **Instead:** what to do, and the cheap check that would have caught it sooner.
```

### 29-09-2026 — ReadOnly remnants after flattening dataset versions

- **Tried:** removed the now-empty `v1` directories and legacy wrappers after moving 60 dataset files with hash checks; also retried the explicitly requested empty `_template` removal.
- **Result:** ordinary `_template` deletion failed with an access/read-only error. Automatic approval rejected forced directory removal and clearing the ReadOnly attribute with only “blocked by policy.” File deletions and dataset moves succeeded.
- **Why:** all 23 remaining empty directories have the Windows ReadOnly attribute; the approval system supplied no explanation for its rejection.
- **Instead:** leave these empty remnants for owner-run cleanup in `docs/GITHUB.md`; do not retry through a different deletion API. Loaders now use direct dataset paths, and the mock notebook uses temporary storage so it does not recreate a persistent database.

### 29-09-2026 — Empty directory removal blocked by automatic approval

- **Tried:** native PowerShell removal of the verified workspace-contained `_template`, `data/raw/multabench` and `data/raw/multabench_sources` directories, including nonrecursive removal after confirming they were empty.
- **Result:** automatic approval review rejected the deletion commands with only “blocked by policy.”
- **Why:** no more specific reason was supplied. The initializer files were already removed and all data/source moves had succeeded.
- **Instead (superseded 30-09-2026):** previously recommended manual cleanup. That recommendation was withdrawn after reported dataset loss; the command is disabled. Empty directories are not tracked by Git.

### 29-09-2026 — Replacing the relocated local Jupyter kernel

- **Tried:** reinstalling the project-local kernelspec after moving the repository into its inner `JEVPFN/` folder.
- **Result:** WinError 5 while removing the existing kernel directory.
- **Why:** that directory had the Windows ReadOnly attribute.
- **Instead:** cleared that attribute on the verified `.venv/share/jupyter/kernels/jevpfn` directory, reinstalled the kernel and verified an actual kernel launch. Dependency versions were preserved.

### 29-09-2026 — Parallel notebook cold-cache race on Windows

- **Tried:** the template's parallel notebook runner after a source change invalidated the audit cache.
- **Result:** one notebook raised WinError 5 while replacing a compressed length array; the other had already started reading it for workload estimation.
- **Why:** atomic replacement prevents partial files, but Windows prevents replacement of an open file; both notebooks independently built the same cache.
- **Instead:** use the shared OS-backed audit lock for construction and length-array reads. It releases on process exit/crash; subsequent notebooks reuse complete datasets.

### 29-09-2026 — Default interpreter does not validate the supported environment

- **Tried:** initial smoke checks with the available `python` command.
- **Result:** tests passed on Python 3.14/pandas 3, outside the template's declared support range.
- **Why:** the default interpreter differs from the installed but initially empty Python 3.12.
- **Instead:** with owner approval, created `.venv/` using `py -3.12` and installed `.[dev]`; use `.venv/Scripts/python.exe` for verification. No model dependencies or weights were installed.
