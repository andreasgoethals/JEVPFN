# JEVPFN

Can label-free Jev features extracted from text improve TabPFN-3.5 prediction?

Author: Andreas Goethals, KU Leuven. Initial collection: **20 core MulTaBench TEXT datasets**.
Public repository: [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).

**Current stage: exploration and feature design. No real Jev calls or modelling are enabled.**
All 20 datasets are available locally. Questions are fixed templates filled with task metadata;
no generative LLM creates questions. True row targets and label statistics never enter Jev.

## Agreed method

- Read each text column separately **and** read all available text columns together.
- Send **text and task metadata only** to Jev. Non-text features go to TabPFN later.
- Skip null/empty/whitespace-only inputs and store missing numeric features. Joint requests use
  available text and are skipped only when all text is missing.
- Cache identical requests once. With t nonempty text columns, both modes give t + 1 request
  slots per row before reuse; a joint request with one available column reuses that column's response.
- Create features locally after review and a small approved pilot. Then debug the predictive
  pipeline on VSC in experiment 0. No experiment makes fresh Jev calls.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | All 20 datasets, full column/target audits and text burden. |
| [02_jev_input_design](notebooks/02_jev_input_design.ipynb) | Exact deterministic requests, missing-input handling, cache demonstration and workload. |
| [03_feature_creation](notebooks/03_feature_creation.ipynb) | Feature-table plan, documented API-body previews and offline budget. No calls. |

The synthetic `example_analysis.ipynb` is preserved. Reusable logic is under `src/`.
Every notebook ends with a **complete printed report**, including every displayed table,
JSON preview, plotted value and figure caption. Reports may contain raw examples and are ignored by Git.

## Run locally

Work in the inner `JEVPFN/` repository. The existing `.venv` is Python 3.12 with prompt/kernel
name **JEVPFN**; no reinstall is required:

```powershell
Set-Location -LiteralPath 'C:\Users\U0152019\PhD Documents\Projects\5. JEVPFN\JEVPFN'
.\.venv\Scripts\python.exe -m jupyter lab
```

Select **Python 3.12 (JEVPFN)**. Run every notebook in parallel with one command:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 4
```

The runner uses separate processes, one numerical-library thread per process, saved logs and
run manifests. It preserves other notebooks' reports during a partial rerun. To rebuild reports
without rerunning anything, add `--summaries-only`. `--only 03_feature_creation` runs one notebook.

For a fresh clone, create the same environment from `pyproject.toml`:

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
.\.venv\Scripts\python.exe -m src.data.prepare
```

The `.venv` directory, shell prompt and Jupyter display name refer to the same environment.
Activation is optional; using its Python directly avoids PowerShell execution-policy changes.
The `jev` and `models` extras declare the future SDK and TabPFN/Torch dependencies. They are not
needed for these notebooks and no weights have been downloaded. On VSC use a separate Conda
**JEVPFN** environment with Python 3.12 and the same pyproject; select a compatible CUDA wheel
before installing `models`. See [VSC.md](docs/VSC.md).

## Files and outputs

```text
config/
  exploration/          audit, dataset catalog and preserved template example
  feature_creation/     text-only per-column and joint extraction design
  experiment_0/         VSC debugging
  experiment_1/         main experiment, disabled
  experiment_2/         representation comparison proposal, disabled
  experiment_3/         output-view comparison proposal, disabled
notebooks/              three research notebooks and one synthetic example
src/                    data/, jev/, cluster/, utils/, visualize/
data/raw/               20 numbered dataset folders, data.csv + metadata.json + download.json
data/jev_cache/        future durable responses and reusable feature tables
output_JEVPFN/
  allresults.md         complete reports from all notebooks
  captions.md           captions from all notebooks
  exploration/          figures/, reports/, logs/, manifests/, results/
  feature_creation/     figures/, reports/, logs/, manifests/, results/
  experiment_0/         same layout, created when used
  experiment_1/         same layout, created when used
  experiment_2/         same layout, created when used
  experiment_3/         same layout, created when used
scripts/slurm/          prepared CPU/GPU preflights; no submission performed
tests/, docs/          verification and research notes
tfm-library/           pinned, read-only literature submodule
```

Each phase also has `allresults.md` and `captions.md`. Each notebook's figure folder holds PDF
figures, its captions and a figure manifest. Future experiment folders are created when used;
empty placeholders are unnecessary. All generated output is ignored by Git.

On VSC, **small outputs** (figures, reports, logs, manifests) use
`$VSC_DATA/JEVPFN/output_JEVPFN/<phase>/`. **Large results** use
`$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/`. Raw data, Jev caches and weights
also use project storage. Unavailable project storage raises an error; it never silently fills
personal DATA. The wICE allocation inherited from CreditPFN is `/lustre1/project/stg_00211`;
confirm access and backup arrangements before use. Mindwell requires a confirmed native GPFS path.

## Reproducibility and next steps

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --offline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.jev.review
```

The catalog pins release versions, official text definitions and SHA-256 hashes. Keep all data
files directly in their numbered folders. Earlier folder-cleanup instructions are retired after
the reported data loss; the command is disabled. **Raw data and paid caches are never output cleanup.**

Next: agree score wording and long-text policy, inspect previews, approve a small local pilot,
implement and test resumable paid extraction, create/validate features locally, then experiment 0
on VSC. Final model inputs, splits, metrics and benchmark configuration remain open.
See [the research plan](docs/RESEARCH_PLAN.md), [feature design](docs/FEATURE_CREATION.md),
[design decisions](docs/DESIGN_NOTES.md), [data audit](docs/INITIAL_REPORT.md) and
[GitHub sync](docs/GITHUB.md). The [model review](docs/REVIEW_2026_09_30.md) preserves the
TabPFN text/Thinking research and per-dataset feature counts.

No credential is required now. `.env*`, credentials, datasets, paid responses, feature tables,
notebook report outputs and weights remain outside Git. Source notebooks have cleared outputs.
The library remains pinned to `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`.

Deliberate template adaptations: phase-specific config/output, complete ignored reports,
numbered raw folders, Python data logic under `src/data`, and removal of the inherited initializer
at the owner's request. Original Git history and useful source/style/test conventions are preserved.

---

## Based on the repository template

This repository was created from
[**andreasgoethals/0.-Template**](https://github.com/andreasgoethals/0.-Template).
[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template: it explains every folder and file here,
and it is a **starting point, not a contract** — this project may grow past it, and deviating where
the work needs it is fine as long as you say so. Generic rule changes belong at the source above.

*Keep this chapter, at the bottom, in every project that starts from the template. Everything above
it is that project's own.*
