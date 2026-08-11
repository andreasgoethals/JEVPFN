# `_template/` — delete this folder

You are looking at **the repository template**. It is not a generator: it *is* a project with the
name left blank, so a new project is this repository with `{{PROJECT_NAME}}` filled in and this
folder removed. Nothing outside `_template/` depends on anything inside it.

Every rule and every file is explained in [`docs/TEMPLATE.md`](../docs/TEMPLATE.md).

---

## Starting a new project

1. On GitHub, **Use this template** → new repository. Pick the visibility. Clone it.
   (Not a fork: a fork of a public repository can never be made private.)
2. Fill in the blanks, delete this folder, install, fetch the literature:

```powershell
python _template/init_project.py CreditICL --description "One sentence about the project."
```

```powershell
Remove-Item -Recurse -Force _template
```

```powershell
pip install -e ".[dev]"
```

```powershell
git submodule update --init
```

3. `python -m pytest -q` and `python -m src.utils.run_notebooks` should both pass.

Then fill in what a script cannot: `README.md` above its last chapter, the `TODO`s in
`docs/VSC.md` and `scripts/slurm/job.slurm`, and this project's own look in
`src/visualize/style.py` (the template fixes the A4 sizes; the colours and grid are yours — copying
a similar project's `style.py` is the fastest start). Leave a `TODO` rather than a plausible guess:
a wrong partition name fails at submit time, a `TODO` fails at read time.

### `init_project.py`

Replaces every `{{PLACEHOLDER}}` with a real value and prints the steps it deliberately does not
take. It installs nothing, runs no git command, pushes nothing, and does not delete this folder.
`--dry-run` shows the file list first.

| Placeholder | Becomes |
|---|---|
| `{{PROJECT_NAME}}` | `CreditICL` — the folder name on **both** cluster tiers, and all prose |
| `{{PACKAGE_NAME}}` | `crediticl` — `pyproject.toml`, the SLURM job name |
| `{{PROJECT_UPPER}}` | `CREDITICL` — the `CREDITICL_STAGING_ROOT` environment variable |
| `{{DESCRIPTION}}` | one sentence — `README.md`, `pyproject.toml` |
| `{{AUTHOR}}` `{{EMAIL}}` `{{YEAR}}` `{{DATE}}` | you, today |
| `{{REPO_URL}}` | auto-detected from `origin`, so it is right even if the repo name differs |

A leftover placeholder afterwards is a **template bug** — a key exists in a file that `derive()`
does not produce.

---

## Bringing the template to an existing repository

Copy [`docs/TEMPLATE.md`](../docs/TEMPLATE.md) into that repository's `docs/`, then give an agent
this:

```text
Read docs/TEMPLATE.md. It is the layout and the rules this repository should follow —
a starting point, not a contract. Bring the repository closer to it.

Work in this order, and stop after step 1 to show me the plan before moving anything:

1. Inventory. List what exists against the structure in TEMPLATE.md: what is
   missing, what is misplaced, what exists under another name.
2. output/. Create it and move every generated artefact under it. This is the
   biggest change and it breaks imports, so do it first, then fix the call sites.
3. src/utils/paths.py. Copy it from the template, set PROJECT_NAME, then replace
   every hard-coded path with a call to it. Grep for string literals containing
   "/", "output", "results", "figures", "logs".
4. The rest of src/utils/ and src/visualize/ — copy from the template, do not
   rewrite. scripts/ keeps only the real experiments and the slurm files; move
   every utility (cleanup, notebook runner, submodule pin) into src/utils/.
5. Notebooks. Move every def and class into src/. Make each notebook clear its own
   figures, save through FigureSaver, and end by printing a section-by-section
   text summary.
6. docs/: .md only, CAPITALS, plus CHANGELOG.md, VSC.md, and AGENTS_MEMORY.md
   (seed its Runs table from whatever cluster runs this project already has).
   Then AGENTS.md at the root, and a CLAUDE.md whose only line is "@AGENTS.md"
   (Claude Code reads CLAUDE.md, every other agent reads AGENTS.md).
7. tfm-library/ as a submodule if it is not there.

The template is at "C:\Users\U0152019\PhD Documents\Projects\0. Template" — copy
files from it rather than rewriting them.

Then run python -m pytest -q and report the result, plus every deviation you left
in place and why. Do not install, commit, or push without asking.
```

---

## Notes on the template itself

**It is not installable.** `pyproject.toml` has `name = "{{PACKAGE_NAME}}"`, not a valid PEP 508
name, so `pip install -e .` fails here. Tests still run — `tests/conftest.py` puts the root on
`sys.path`.

**After changing anything here**, initialise a throwaway copy and run it, rather than trusting that
it still works.

**Keep it project-agnostic.** No dataset, model, experiment or result anywhere. Several files under
`src/` ship deliberately empty; fill one in only when the behaviour is genuinely identical in every
project.
