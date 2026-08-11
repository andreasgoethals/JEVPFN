# {{PROJECT_NAME}}

> **Looking at the template itself?** Read [`_template/README.md`](_template/README.md) —
> it explains how this repository works and what every module in it does. This file is a
> seed: replace everything above the last chapter with your project's own.

> {{DESCRIPTION}}

    Author   {{AUTHOR}} <{{EMAIL}}>
    Context  PhD research, KU Leuven — machine learning on tabular data
    Cluster  KU Leuven VSC (Genius login, wICE and Mindwell compute)

## What this is

Two or three paragraphs. What the project does, what it is compared against, and
what a result from it looks like.

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

Then populate the literature submodule. It is already wired up — the pin came with the
repository — but a clone leaves the folder empty until you ask for its 749 MB:

```powershell
git submodule update --init
```

Verify the repository is healthy — ruff, pytest, and an import of every module:

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

Add the project's real entry points here as they appear, each with the one line
that says what it produces and where.

## Repository layout

| Path | What lives there |
|---|---|
| [`config/`](config/) | one YAML per experiment; sweeps at the top of the file |
| `data/raw/` | inputs, never modified, never committed |
| `data/processed/` | generated cache, rebuildable |
| [`docs/`](docs/) | all documentation — see below |
| [`notebooks/`](notebooks/) | thin: every notebook imports its logic from `src/` |
| `output/` | **everything** the code generates: figures, logs, results, manifests |
| [`scripts/`](scripts/) | runnable entry points, plus `slurm/` job scripts |
| [`src/`](src/) | all importable logic |
| [`tests/`](tests/) | one file per `src` module |
| `tfm-library/` | the shared literature submodule — **read-only** |

Documentation:

| Doc | What it answers |
|---|---|
| [`docs/TEMPLATE.md`](docs/TEMPLATE.md) | the layout and every rule. Governing document; never edited here |
| [`docs/VSC.md`](docs/VSC.md) | how to run this project on the KU Leuven cluster |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | what changed, newest first |
| [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) | what was tried and **failed**, newest first |
| [`AGENTS.md`](AGENTS.md) | the rules an AI agent follows in this repository |

## The literature submodule

[`tfm-library/`](tfm-library/) is a **pinned git submodule** holding the shared
TFM literature: the papers as PDFs with full-text extractions, per-paper
summaries, a cross-paper synthesis, flat-text snapshots of the upstream reference
implementations, and the VSC documentation.

It is here so that a question about the literature — or about how the official
code actually does something — is answerable **from inside this repository, by
reading and grepping files**, with no web search and nothing taken on memory.
That matters most for AI agents: an agent that can read the sources does not have
to guess, and every claim it makes can be traced to a path.

**It is read-only.** Never edit anything inside it; the only writable file is
`tfm-library/PROJECT_SPECIFIC.md`, which the library gitignores for exactly this
purpose. A submodule pins one exact commit, so a result stays reproducible against
the literature as it stood when it was produced. Bump the pin with:

```powershell
python scripts/update_tfm_library.py
```

which reports first and changes nothing until you pass `--update`.

---

## Based on the repository template

This repository was created from
[**andreasgoethals/repo-template**](https://github.com/andreasgoethals/repo-template)
and follows it: the layout, the `output/`-only rule for generated files, the
shared figure style, the read-only `tfm-library/` submodule, and the checks in
`tests/test_template_compliance.py`.

[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template, copied in verbatim. It is
the **governing document** for this repository's structure and rules, and it is
**never edited here** — only at the source above. The compliance test hashes it,
so a local edit fails the test suite. If a rule needs to change, change it at the
source and pull it down.

Start a new project with **Use this template** on that repository — not a fork, so the new
repository can be private — then `python _template/init_project.py <ProjectName>`, then delete
`_template/`, then `python scripts/check.py`.

*Keep this chapter, at the bottom, in every project that starts from the
template. Everything above it is this project's own.*
