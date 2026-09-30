# JEVPFN

Can label-free Jev features extracted from text improve tabular prediction?
TabPFN-3.5 is the main model; the proposed benchmark also includes other foundation models
and conventional baselines on the **20 core MulTaBench TEXT datasets**.

**Current phase: exploration and design. No paid calls or model experiments are enabled.**
The public repository is [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).

## Workflow

Exploration → local feature pilot and full feature creation → experiment 0 on VSC →
predictive experiments. The [research plan](docs/RESEARCH_PLAN.md) separates agreed choices
from the proposed experiments and remaining decisions.

Jev reads each text column separately and all available text columns jointly, with fixed
questions filled from task metadata. **Non-text features and true row targets never enter Jev.**
Skip empty inputs and store missing numeric features. Identical complete requests reuse a
cached response; future experiments join different combinations of those saved features.

| Notebook | Purpose |
|---|---|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | Text counts, lengths, missingness, per-column and joint duplicates; complete dataset audit. |
| [02_jev_input_design](notebooks/02_jev_input_design.ipynb) | Deterministic requests, task-specific questions, temporary mock cache and token estimates. |
| [03_feature_creation](notebooks/03_feature_creation.ipynb) | Intended API bodies, numeric feature tables, reuse and cost estimates. No API calls. |

All logic is in `src/`. The inherited synthetic example remains separate. Every notebook
ends with a full printed report containing all displayed tables, previews, plotted values and
captions. Source notebooks have cleared outputs; generated reports remain local and ignored.

## Local environment and notebooks

Open the inner `JEVPFN/` directory containing this README and `pyproject.toml`.
For a fresh checkout, run in PowerShell:

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
.\.venv\Scripts\python.exe -m src.data.prepare
```

The existing local environment is already set up. Its prompt/kernel is **JEVPFN**; the
`.venv` folder is its location. Activation is optional. Launch Jupyter or run everything:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 4
```

Select **Python 3.12 (JEVPFN)** in Jupyter/VS Code. The runner uses four separate processes,
saves logs and preserves reports from other notebooks during partial reruns.
`--only 01_data_exploration` selects one notebook; `--summaries-only` rebuilds saved reports.

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --offline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.jev.review
```

The last command recalculates the duplicate-aware offline budget; it sends nothing.
All dependencies are declared in `pyproject.toml`. Optional `jev`, `models`, `baselines`,
`tabicl`, `tabdpt`, `mitra` and `tabstar` extras are for later implementation. Notebook setup
installs none of those extras. GPU/model environments still require VSC validation; see
[the VSC guide](docs/VSC.md). TabFM is catalogued pending a pinned source revision.

## Files and outputs

```text
config/
  exploration/          dataset catalog and audit
  feature_creation/     local feature preparation
  experiment_0/         VSC debugging
  experiment_1/         main comparison, shared CV proposal and model catalog
  experiment_2/         original-text comparison proposal
  experiment_3/         Jev output-representation proposal
notebooks/              thin notebook clients
src/                    reusable data, Jev, cluster, plotting and utility code
tests/                  deterministic checks
data/raw/               20 numbered folders, files directly inside each folder
data/jev_cache/         future durable responses and feature tables, created when needed
output_JEVPFN/
  Allresults.md         complete reports from every notebook
  Captions.md           all figure captions
  figures/
    exploration/<notebook>/
    feature_creation/<notebook>/
    experiment_0/<notebook>/ ...
  exploration/          reports/, logs/, manifests/, results/
  feature_creation/     same layout
  experiment_0/ ...     same layout, created when used
docs/                   research plan, data sources, VSC guide and maintenance records
scripts/slurm/          prepared preflights; no jobs submitted
tfm-library/            pinned read-only literature submodule
```

Each phase also has `Allresults.md` and `Captions.md`. Notebook figure folders contain PDFs,
`Captions.md` and a figure manifest. No empty future-output folders are created.

On VSC, small outputs use `$VSC_DATA/JEVPFN/output_JEVPFN/`, including the shared `figures/`
folder. Large tables use `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/`.
Raw data, paid caches and weights also live under that project-storage `JEVPFN/` directory.
A project-storage error stops the run rather than filling personal DATA.

## GitHub and provenance

VS Code tracks local Git changes; they appear on GitHub only after committing and pushing.
The repository is already connected to the public GitHub URL above. For future code updates:

```powershell
git status
git add <specific-code-or-config-files>
git commit -m "Describe the change"
git push origin main
```

On VSC, initially clone with `git clone --recurse-submodules https://github.com/andreasgoethals/JEVPFN.git`;
subsequent updates use `git pull --ff-only` and `git submodule update --init --recursive`.
Transfer data and completed feature artifacts separately; Git does not sync them.

The [data-source notes](docs/DATA_SOURCES.md) document official definitions and immutable hashes.
Raw data and paid caches are protected inputs, never cleanup targets. The old empty-folder
cleanup command is disabled after the dataset-loss incident. Credentials, `.env*`, data,
model weights and generated reports are ignored. No credentials are needed for the current notebooks.

Based on [Andreas' research template](https://github.com/andreasgoethals/0.-Template).
Useful source, style, test and Git-history conventions are retained; project-specific phases,
numbered raw folders and shared figures replace the original generic layout.
