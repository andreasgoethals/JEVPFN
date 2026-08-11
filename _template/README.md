# `_template/` — the parts that do not travel

**If you are in a project: delete this folder.** It is the template's own machinery. Nothing
outside it depends on it.

```powershell
Remove-Item -Recurse -Force _template
```

**If you are in the template repository:** this is the manual for the four tools in here. The
rules, the structure and what every module must do live in
[`docs/TEMPLATE.md`](../docs/TEMPLATE.md) — the one governing document, not repeated here.

---

## The idea

**This repository is not a generator. It *is* a project, with the name left blank.** Every folder
and module is already in its final place; a new project is this repository with
`{{PROJECT_NAME}}` filled in and this folder removed. No `skeleton/`, no copying step, no second
copy of anything to keep in sync — so the `.gitignore`, the compliance test and the figure
pipeline are all exercised *in place*, and a mistake in them shows up here rather than in a
project six months from now.

```
Use this template (GitHub)   →  pick the visibility; the submodule pin comes too
clone it, open in VS Code    →  .vscode/settings.json is committed, so `from src...` resolves
point an agent at INITIALISE.md
python _template/init_project.py CreditICL --description "..."
Remove-Item -Recurse -Force _template
git submodule update --init  →  fetches the literature (~749 MB)
python scripts/check.py      →  must pass before the first commit
```

### Why not a fork

- **A fork of a public repository can never be made private.** GitHub does not allow changing a
  fork's visibility. "Use this template" lets the new repository be private even when the template
  is public — which a project under embargo, or with licence-restricted data, needs.
- **The merge a fork promises does not work.** The files a template change touches most are
  exactly the ones `init_project.py` rewrote per project — `README.md`, `pyproject.toml`,
  `src/utils/paths.py`, the hash line — so every `git pull upstream main` conflicts where you
  least want to merge by hand. A fork also puts the template's commits at the root of the
  project's history and makes GitHub default a pull request to the *upstream*.

`sync_template_rules.py` does the propagation instead, which removes the only real argument for
forking.

---

## The four tools

| File | What it does |
|---|---|
| [`INITIALISE.md`](INITIALISE.md) | **The brief an agent follows** in a fresh repository: the facts to ask for, the one command, the files only judgement can fill, what to report. Point an agent here. |
| [`init_project.py`](init_project.py) | Fills in every `{{PLACEHOLDER}}` and bakes `docs/TEMPLATE.md`'s SHA-256 into the compliance test. Nothing else — no install, no git, no push, and it does not delete this folder for you. |
| [`check_template.py`](check_template.py) | *Does this template still produce a healthy project?* Makes one in a temp directory, checks it, then injects every rule violation into fresh copies and requires the compliance test to **fail**. Run after any change here. |
| [`sync_template_rules.py`](sync_template_rules.py) | Pushes a changed rule down into every existing project. Reports by default. |
| [`sync_tfm_library.py`](sync_tfm_library.py) | Bumps the `tfm-library/` pin across every project at once. Reports by default. |
| [`_projects.py`](_projects.py) | Shared by both `sync_*` tools: finds the sibling projects, handles the selection flags. One copy, because two filesystem walks drift. |

### `init_project.py`

```powershell
python _template/init_project.py CreditICL --description "One sentence about the project."
python _template/init_project.py CreditICL --dry-run
```

| Placeholder | Becomes | Where it matters |
|---|---|---|
| `{{PROJECT_NAME}}` | `CreditICL` | the folder name on **both** cluster tiers, prose everywhere |
| `{{PACKAGE_NAME}}` | `crediticl` | `pyproject.toml` name, SLURM job name |
| `{{PROJECT_UPPER}}` | `CREDITICL` | the `CREDITICL_STAGING_ROOT` environment variable |
| `{{DESCRIPTION}}` | one sentence | `README.md`, `pyproject.toml`, `CITATION.cff` — three places that must not drift |
| `{{AUTHOR}}` `{{EMAIL}}` | you | `LICENSE`, `pyproject.toml`, `CITATION.cff`, `AGENTS.md` |
| `{{YEAR}}` `{{DATE}}` `{{DATE_ISO}}` | today | `LICENSE`; `DD-MM-YYYY` for the logs; ISO only for `CITATION.cff` |
| `{{REPO_URL}}` `{{TFM_LIBRARY_URL}}` | the remotes | `pyproject.toml`, `.gitmodules` |
| `{{TEMPLATE_SHA256}}` | the hash | `tests/test_template_compliance.py` |

`{{REPO_URL}}` is **auto-detected from `origin`** before falling back to a name-based guess, so it
is right even when the repository name differs from the project name — which a guess never is.
SSH remotes are normalised to https so the value in `pyproject.toml` is clickable.

The substitution regex is `(?<!\$)\{\{([A-Z][A-Z0-9_]*)\}\}` — upper case only, never after a
`$`, which is what keeps it away from GitHub Actions expressions like
`${{ matrix.python-version }}`. A leftover placeholder is reported as a **template bug**: it means
a key exists in a file that `derive()` does not produce.

It also reports the **submodule state** — `MISSING` (no gitlink), `pinned, empty` (normal after a
clone), `populated` — and prints the command for that case. Three situations that look identical
from outside and need different fixes.

### `check_template.py`

```powershell
python _template/check_template.py            # the full sweep
python _template/check_template.py --fast     # skip the notebook run
python _template/check_template.py --keep     # leave the temp copies to inspect
```

1. `scripts/check.py` in a freshly initialised copy — ruff, every import, pytest.
2. `scripts/run_notebooks.py` in that copy, **and a check that it really produced PDFs,
   `CAPTIONS.md` and `All_Results.md`**. A runner that exits 0 having drawn nothing is the failure
   mode worth guarding against.
3. Every rule from `docs/TEMPLATE.md`, injected one at a time into its own fresh copy, with the
   compliance test required to **fail**.

Step 3 is why it exists. **Every rule you add to `docs/TEMPLATE.md` gets an entry in
`VIOLATIONS`** — if you cannot make the test fail, the rule is not enforced, whatever the document
says.

### `sync_template_rules.py`

```powershell
python _template/sync_template_rules.py                  # what is out of date, where
python _template/sync_template_rules.py --diff           # show what differs
python _template/sync_template_rules.py --apply          # docs/TEMPLATE.md + the hash
python _template/sync_template_rules.py --apply --also scripts/check.py
```

Three tiers, split by **who owns a file once the project exists**:

- **Applied** — `docs/TEMPLATE.md` and its SHA-256. Safe to overwrite unasked: the template owns
  it verbatim and the compliance test already proves the project has not edited it — if it had,
  that project's suite would be red.
- **Reported** — every other file the template ships unchanged. Shown as differing, never written,
  because a project may *extend* some (a new cleanup category) and is *expected* to delete others
  (the example notebook). Look with `--diff`, take one with `--also`.
- **Never compared** — anything the initialiser rewrote, plus `paths.py` and `style.py`, where
  "differs from the template" is the correct state.

It skips the template itself, writes without committing, and prints the review steps.

### `sync_tfm_library.py`

```powershell
python _template/sync_tfm_library.py                     # report on all of them
python _template/sync_tfm_library.py --only CreditICL CreditPFN
python _template/sync_tfm_library.py --exclude TabPFN
python _template/sync_tfm_library.py --pick              # numbered list, choose
python _template/sync_tfm_library.py --update            # move the working trees
python _template/sync_tfm_library.py --update --commit   # ...and record the pins
```

Finds every git repository two levels under the projects directory — so
`Projects/4. CreditICL/CreditICL` is found — and keeps those whose `.gitmodules` declares
`tfm-library`. Nothing hard-coded, so a new project appears the moment it has the submodule.
`--pick` reads a selection: nothing for all, `2,4` for only those, `-3` for all except the third.

Three separate gears on purpose. `--update` moves a working tree; `git submodule update --remote`
does **not** record the pin, and the reason the literature is pinned is that a result stays
reproducible against the sources as they stood. Moving that silently across five repositories is
what the pin exists to prevent. It commits with a `-- tfm-library` pathspec so it can never sweep
in unrelated staged work, refuses outright in a repository with other things staged, and never
pushes.

---

## Changing a rule

1. Edit `docs/TEMPLATE.md` here; note it in `docs/CHANGELOG.md`.
2. Update the check in `tests/test_template_compliance.py` and its violation in
   `check_template.py`.
3. `python _template/check_template.py` — the new violation must be **CAUGHT**.
4. `python _template/sync_template_rules.py --apply`.
5. Per project: `git diff`, `python scripts/check.py`, a line in its `CHANGELOG.md`.

## The template repository is not installable, and that is fine

`pyproject.toml` has `name = "{{PACKAGE_NAME}}"`, which is not a valid PEP 508 name, so
`pip install -e .` fails **here**. It is not a package; it is a template. Everything that matters
still works, because `tests/conftest.py` puts the root on `sys.path`:

```powershell
python scripts/check.py
```

The compliance test's hash check **skips** while `src/utils/paths.py` still holds
`{{PROJECT_NAME}}` — there is nothing to compare against until a project exists — and is enforced
the moment `init_project.py` has run. `.github/workflows/check.yml` uses the same signal to pick
its install step, which is why one workflow file serves both the template and every project.

## Keep it project-agnostic

No dataset, model, experiment or result anywhere. If a rule cannot be stated without naming one,
it belongs in a project.

The one exception is the seeded content in `docs/AGENTS_MEMORY.md`: real dead ends, kept because
every project can hit them. When a project hits one that turns out to be universal, **promote it**
here and every new project starts already knowing.
