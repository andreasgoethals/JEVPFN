# {{PROJECT_NAME}}

> **Looking at the template itself?** Read [`_template/README.md`](_template/README.md) for its
> tooling, and [`docs/TEMPLATE.md`](docs/TEMPLATE.md) for every rule. This file is a seed:
> replace everything above the last chapter with your project's own, this note included.

> {{DESCRIPTION}}

    Author   {{AUTHOR}} <{{EMAIL}}>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (Genius login, wICE and Mindwell compute)

## What this is

Two or three paragraphs. What the project does, what it is compared against, and what a result
from it looks like.

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

The literature submodule is already wired up, but a clone leaves the folder empty until you ask
for its ~749 MB:

```powershell
git submodule update --init
```

Then verify — ruff, an import of every module, and pytest:

```powershell
python scripts/check.py
```

On the cluster, see [`docs/VSC.md`](docs/VSC.md).

## Running it

```powershell
python scripts/run_notebooks.py
```

```powershell
python scripts/clean_run.py
```

Add this project's real entry points here as they appear, each with the one line saying what it
produces and where.

## Layout

| Path | What lives there |
|---|---|
| [`config/`](config/) | one YAML per experiment; the sweep block at the top |
| `data/raw/`, `data/processed/` | inputs (never committed) and the rebuildable cache |
| [`notebooks/`](notebooks/) | thin: every notebook imports its logic from `src/` |
| `output/` | **everything** the code generates: figures, logs, results, manifests |
| [`scripts/`](scripts/) | runnable entry points, plus `slurm/` job scripts |
| [`src/`](src/) | all importable logic |
| [`tests/`](tests/) | one file per `src` module |
| `tfm-library/` | the shared literature submodule — **read-only** |
| [`docs/TEMPLATE.md`](docs/TEMPLATE.md) | the layout and every rule. Governing; never edited here |
| [`docs/VSC.md`](docs/VSC.md) | how to run this on the KU Leuven cluster |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | what changed, newest first |
| [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) | what was tried and **failed**, newest first |
| [`AGENTS.md`](AGENTS.md) | the rules an AI agent follows here |

## The literature submodule

[`tfm-library/`](tfm-library/) is a **pinned git submodule** holding the shared TFM literature:
the papers with full-text extractions, per-paper summaries, a cross-paper synthesis, flat-text
snapshots of the upstream reference implementations, and the VSC documentation.

It is here so a question about the literature — or about how the official code actually does
something — is answerable **from inside this repository, by reading and grepping files**, with no
web search and nothing taken on memory. That matters most for agents: one that can read the
sources does not guess, and every claim it makes traces to a path.

**It is read-only**, and the pin is what keeps a result reproducible against the literature as it
stood. Bump it with `python scripts/update_tfm_library.py`, which reports first and changes
nothing until you pass `--update`. Full contract in [`docs/TEMPLATE.md`](docs/TEMPLATE.md).

---

## Based on the repository template

This repository was created from
[**andreasgoethals/0.-Template**](https://github.com/andreasgoethals/0.-Template) and follows it:
the layout, the `output/`-only rule for generated files, the shared figure style, the read-only
`tfm-library/` submodule, and the checks in `tests/test_template_compliance.py`.

[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template, copied in verbatim. It is the
**governing document** for this repository's structure and rules and is **never edited here** —
only at the source above. The compliance test hashes it, so a local edit fails the test suite.

Start a new project with **Use this template** on that repository — not a fork, so the new
repository can be private — then `python _template/init_project.py <ProjectName>`, delete
`_template/`, and run `python scripts/check.py`.

*Keep this chapter, at the bottom, in every project that starts from the template. Everything
above it is that project's own.*
