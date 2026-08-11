# {{PROJECT_NAME}}

> {{DESCRIPTION}}

    Author   {{AUTHOR}} <{{EMAIL}}>
    Context  PhD research, KU Leuven — machine learning on tabular data

## What this is

Replace this with what the project does, what it is compared against, and what a result from it
looks like.

## Setup

Windows PowerShell — **one command per line, `&&` is a parser error there**:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -e ".[dev]"
```

The literature submodule is wired up already, but a clone leaves the folder empty until you ask for
its ~749 MB:

```powershell
git submodule update --init
```

## Running it

```powershell
python -m src.utils.run_notebooks
```

```powershell
python -m src.utils.clean_run
```

Add this project's own entry points here as they appear. On the cluster, see
[`docs/VSC.md`](docs/VSC.md).

---

## Based on the repository template

This repository was created from
[**andreasgoethals/0.-Template**](https://github.com/andreasgoethals/0.-Template).
[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template: it explains every folder and file here,
and it is a **starting point, not a contract** — this project may grow past it, and deviating where
the work needs it is fine as long as you say so. Generic rule changes belong at the source above.

*Keep this chapter, at the bottom, in every project that starts from the template. Everything above
it is that project's own.*
