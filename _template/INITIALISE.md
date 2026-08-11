# INITIALISE.md — read this first if you are an agent in a fresh fork

**You are here because someone made a new repository from the template ("Use this template")
and asked you to turn it into a real project.** This file is the whole job, in order. Follow it
top to bottom.

If `_template/` does not exist, this repository is already a project — stop, and read
[`AGENTS.md`](../AGENTS.md) instead.

---

## 0. Confirm you are in an un-initialised template

```powershell
python -c "import pathlib; print('{{PROJECT' + '_NAME}}' in pathlib.Path('src/utils/paths.py').read_text())"
```

`True` means not yet initialised. `False` means it already is — do not run step 2 again; go to
step 4 and fill in what is missing.

## 1. Get four facts from the user. Ask, do not invent.

| Fact | Used for | If you have to guess |
|---|---|---|
| **Project name**, e.g. `CreditICL` | the directory name on both cluster tiers, the package name, the `<NAME>_STAGING_ROOT` variable, every mention in prose | **do not guess.** Ask. It is baked into a dozen places and renaming it later means re-running this. |
| **One sentence** — what it is and what question it answers | `README.md`'s opening line, `pyproject.toml` description, `CITATION.cff` abstract | ask; a placeholder sentence survives for years |
| **Two or three paragraphs** — what it does, what it is compared against, what a result looks like | `README.md` § What this is | ask |
| **Series names** it will plot repeatedly (model names, arms, dataset groups) | `src/visualize/style.py`, so a name is the same colour in every figure | leave unregistered and say so; it is safe to add later, as long as you **append** |

The repository URL is detected from `origin` automatically. Pass `--repo-url` only if the
remote is wrong or absent.

## 2. Run the initialiser. One command.

```powershell
python _template/init_project.py CreditICL --description "One sentence about the project."
```

It fills in every `{{PLACEHOLDER}}` and writes `docs/TEMPLATE.md`'s SHA-256 into
`tests/test_template_compliance.py`. It installs nothing, commits nothing, pushes nothing and
deletes nothing. Use `--dry-run` first if you want to see the file list.

If it warns about **unsubstituted placeholders**, that is a template bug, not a user error —
report it and stop.

## 3. Delete this folder.

```powershell
Remove-Item -Recurse -Force _template
```

No project code imports anything from it, and a later rule change arrives through the
template's own `_template/sync_template_rules.py` — run from the template, writing into this
project — so there is nothing here worth keeping.

If it survives, say so in your report: the compliance test's template-hash check is keyed on
the project name rather than on this folder, so it *is* enforced either way, but a leftover
`_template/` in a project is confusing.

## 4. Fill in what a script cannot.

These are prose and judgement, which is why they are your job and not the initialiser's.

| File | What to write |
|---|---|
| `README.md` | Replace everything **above** the last chapter. Keep the "Based on the repository template" chapter at the bottom, unchanged. Delete the "Looking at the template itself?" note at the top. |
| `src/visualize/style.py` | `REGISTERED` — the project's series names, in slot order. **Append, never insert:** inserting repaints every figure after it and silently invalidates any already in a paper. |
| `docs/VSC.md` | The `TODO` markers: partition, walltime limit, credit account, Python module. Read the cluster documentation in `tfm-library/` for these — do not guess a partition name. |
| `scripts/slurm/job.slurm` | The same `TODO`s, plus the real entry point. |
| `config/example.yaml` | Copy it per experiment under a real name; delete the example once there is a real one. |
| `notebooks/example_analysis.ipynb` | The pattern to copy. Delete it once a real notebook exists. |
| `docs/CHANGELOG.md` | The first entry is already dated. Add a line for anything you changed here. |
| `AGENTS.md` § 1 | The current `tfm-library` pin — `git submodule status`. |

Leave a `TODO` in place rather than filling it with a plausible-looking guess. A wrong
partition name fails at submit time; a `TODO` fails at read time, which is cheaper.

## 5. The literature submodule

The pin ships with the template, so nothing needs adding. The folder is empty until someone
asks for its 749 MB:

```powershell
git submodule update --init
```

It is **READ-ONLY**. The only writable file inside it is `tfm-library/PROJECT_SPECIFIC.md`,
created from `PROJECT_SPECIFIC.template.md`. See `AGENTS.md` § 1 for the full contract.

If `init_project.py` reported the pin as **MISSING**, the template was published without its
gitlink — report that, and add it once with
`git submodule add https://github.com/andreasgoethals/TFM_Library.git tfm-library`.

## 6. Verify. Do not skip this.

```powershell
pip install -e ".[dev]"
```

```powershell
python scripts/check.py
```

Ruff, an import of every `src` module, and pytest — including
`tests/test_template_compliance.py`, whose template-hash check is now **live**. It must pass
before the first commit. If it does not, fix it; do not commit around it.

## 7. Report back, then stop.

Tell the user, in this order:

1. the project name, description and repo URL that were written in;
2. the `scripts/check.py` verdict, verbatim;
3. every `TODO` you left and why (the facts you did not have);
4. whether `_template/` was deleted;
5. the submodule state — pinned and empty, or populated.

**Then stop.** Do not `pip install` anything beyond `-e ".[dev]"`, do not run a training job,
and do not commit or push. Those are the user's calls — see `AGENTS.md` § 5.

---

## Things that will catch you out

- **Never edit `docs/TEMPLATE.md`.** Its hash is in the compliance test; an edit fails the
  suite. Project-specific rules go in `README.md` or a new `docs/<NAME>.md`.
- **A notebook may contain no `def` and no `class`**, and its last code cell must print a text
  summary. Both are enforced.
- **Everything generated goes under `output/`**, and paths are built only in
  `src/utils/paths.py`. A hard-coded write path fails the compliance test.
- **`.gitignore` rules naming a directory that occurs at more than one depth must be
  root-anchored.** A bare `figures/` also matches `output/figures/`.
- **Windows PowerShell has no `&&`.** One command per line, or `;` with `if ($?) { ... }`.
- **`docs/AGENTS_MEMORY.md` already has entries.** They are real dead ends that apply to every
  project. Read them before you debug anything — three of them cost an hour each the first time.
