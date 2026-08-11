# Agents' memory — what was tried and did **not** work

**Read this before starting work in this repository.** Not what changed — that is
[`CHANGELOG.md`](CHANGELOG.md). This is the list of dead ends: approaches that looked
right, were tried, and failed. It exists so nobody pays twice for the same mistake.

**One chapter per date, `DD-MM-YYYY`, newest at the top.**

Every entry is exactly four lines:

```
### <short name for the attempt>
- **Tried:** what was actually done.
- **Result:** what happened — the error, the wrong number, the silent no-op.
- **Why:** the underlying reason, once understood.
- **Instead:** what to do, and the cheap check that would have caught it sooner.
```

Write an entry for **any failure that cost more than a couple of minutes** — including
failures you eventually fixed. Especially those: the fix is one line in the changelog,
the dead end is the hour. If the same entry would apply to every project, say so in the
reply so it can be **promoted** into the template's copy of this file, and every project
started afterwards begins already knowing it.

Do not delete entries. A dead end that is no longer reachable is still evidence. If an
entry stops being true, add a new dated entry saying so and leave the old one.

---

## {{DATE}}

*The entries below this date came with the template. They are real dead ends from building
it, kept because every project can hit them. Add yours above them.*

### Monkey-patching `plt.savefig` inside the notebook runner
- **Tried:** having the notebook runner wrap matplotlib so it captured and saved every
  figure a notebook drew, instead of the notebook saving its own.
- **Result:** `python scripts/run_notebooks.py` produced every figure correctly, and
  *Run All* in Jupyter produced none. The bug was invisible from the runner, which is
  the only place it was ever tested.
- **Why:** the patch only exists in the runner's process. An interactive kernel never
  sees it, so the two execution paths silently produced different artefacts — the worst
  kind of difference, because the interactive one is where figures are actually iterated.
- **Instead:** the notebook saves its own figures through
  `src.visualize.figures.FigureSaver`; the runner only executes notebooks and collects
  what they wrote. Cheap check: run one notebook interactively and confirm the same files
  appear as from the runner.

### Unanchored directory rules in `.gitignore`
- **Tried:** a bare `figures/` and `results/` in `.gitignore`, on the assumption that a
  rule names a top-level directory.
- **Result:** `output/figures/` and `output/results/` were ignored too, so a run's
  committed PNGs and small result summaries silently never reached git. Noticed only
  when a clone came up empty.
- **Why:** a gitignore pattern with no leading slash and no internal slash matches that
  name at **every** depth.
- **Instead:** anchor with a leading slash (`/checkpoints/`) whenever a name should match
  at the root only. `tests/test_template_compliance.py` now fails on an unanchored rule
  whose name also appears deeper in the tree. Cheap check:
  `git check-ignore -v <the path you expect to be tracked>`.

### Editing `docs/TEMPLATE.md` in place to record a project-specific rule
- **Tried:** adding a project-specific note to `docs/TEMPLATE.md`, since that is where
  the rules live.
- **Result:** `tests/test_template_compliance.py` failed on the hash comparison.
- **Why:** `TEMPLATE.md` is shared verbatim across every project. A local edit either
  gets overwritten on the next sync or quietly makes this repository's rules diverge from
  everyone else's.
- **Instead:** project-specific rules go in `README.md` or a new `docs/<NAME>.md`.
  Genuinely generic rules are changed at the template source and pulled down.

### `monkeypatch` on anything that runs inside a process pool
- **Tried:** `monkeypatch.setattr(run_notebooks, "notebooks_dir", ...)` and then calling
  `run_all()`, expecting the redirected directory to be used.
- **Result:** the notebook was reported "not found" even though it sat exactly where the
  patch pointed.
- **Why:** `run_all` executes each notebook through a `ProcessPoolExecutor`, and the child
  interpreter imports the module from scratch. **Monkeypatches do not cross a process
  boundary.** Environment variables do — which is why redirecting `output/` with `VSC_DATA`
  works in the same test.
- **Instead:** test `run_one` (same process) plus the two document writers; `run_all` is
  those three calls plus the pool. Cheap check: if a fixture patches an attribute, it cannot
  be relied on inside anything that spawns a process. Redirect with the environment instead.

### `matplotlib` rcParams that look like they took but did not
- **Tried:** `savefig.bbox = "standard"` as the opposite of `"tight"`, and
  `mpl.colormaps.register(..., force=True)` to make `style.apply()` re-runnable.
- **Result:** the first was silently normalised to `None`, so a test asserting
  `== "standard"` failed while the behaviour was correct all along. The second worked and
  emitted `UserWarning: Overwriting the cmap ...` on **every** call.
- **Why:** matplotlib accepts only `"tight"` or `None` for that key — `None` *is* the
  spelling for "use the declared figure size". And `force=True` suppresses the error, not the
  warning; it is meant for a deliberate one-off, not a function called on every notebook run.
- **Instead:** read the value back out of `mpl.rcParams` after setting it rather than assuming
  the write took, and `unregister` before registering. A warning that always fires is a
  warning nobody reads, and it hides the ones that matter.

### Reusing one scratch directory on Windows
- **Tried:** copy a tree to a fixed temp path, run pytest against it, `rmtree`, repeat.
- **Result:** `PermissionError: [WinError 5] Access is denied`, then `FileExistsError` on the
  next iteration because the failed cleanup left the tree behind.
- **Why:** Windows will not remove a directory while any process still holds a handle inside
  it, and a just-finished subprocess often still does.
- **Instead:** a unique `tempfile.mkdtemp()` per iteration. Never reuse a path you have just
  tried to delete on Windows, and prefer renaming to deleting when you only need it gone from
  where it was.
