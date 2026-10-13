# VSC setup

Current status: no JEVPFN allocation or model run has been submitted. Local Jev feature
creation precedes experiment 0. VSC prediction consumes completed feature tables offline;
Jev API access from compute nodes is unnecessary for this workflow and remains unverified.

## Self-hosted decision extractors

Self-hosting Liquid d1-3B, Strands Decider 2B or Perplexity Decider 27B removes the vendor
API token fee; it does not remove weight licence conditions, Slurm credit charges, setup,
storage or inference time. The six-provider comparison remains disabled. Hosted extraction
can run locally; a separate VSC extraction job for an open checkpoint is a proposed exception
to the local-only workflow, not an allocation that has been launched.

The read-only snapshot `VSC Documentation.txt` was checked 13 October at the existing library
pin `de324efcca55be392feb0c01830b3b6ce364587f`: `wice_hardware.rst` lists 80 GiB A100/H100
GPUs and `mindwell_hardware.rst` lists 192 GiB B200 GPUs. One wICE A100 is the first resource
to profile for the 27B extractor's approximately 49 GiB BF16 weights; full request memory,
attention kernels and throughput are unmeasured. Start at batch size one, measure peak GPU/RAM,
then increase batches from measurements. Use H100 for demonstrated throughput benefit or
Mindwell B200 if 80 GiB proves insufficient; do not require B200 solely because it is newer.
Liquid/Strands can also be profiled on one A100. Keep each incompatible/custom-code model
in its own frozen Python environment and audit the pinned inference code before loading it.

Approved checkpoints, raw inputs and permanent dataset feature files belong on PROJECT;
compact extraction logs/manifests belong on DATA. Use Lustre on wICE and native GPFS on
Mindwell, with the existing project-root override. Node scratch is disposable. These are
batch extraction jobs, not notebooks or public inference servers. Prepare checkpoints on a
connected setup host and run offline inside Slurm, so self-hosted inference needs no external
API connectivity. Actual account access, quota, backup and CUDA compatibility remain untested.

## Environment

Use the CreditPFN Conda bootstrap with a separate Python 3.12 environment named **JEVPFN**.
Dependencies remain managed by pip and `pyproject.toml`; do not share CreditPFN's environment.
Run setup on an approved connected setup/login session, not inside an experiment:

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda create -n JEVPFN python=3.12 pip  # Skip if it already exists.
conda activate JEVPFN
cd "$VSC_DATA/JEVPFN"
python -m pip install -e '.[dev,notebooks]'
python -m pip check
```

After selecting the GPU runtime in experiment 0, optional model extras can be installed.
The existing TabPFN/TabICL extras pin Torch 2.12.1. Select a CUDA wheel compatible with the allocated
GPU and driver before installing extras: see the [official PyTorch wheel table](https://pytorch.org/get-started/previous-versions/).
The wICE A100 candidate is cu126; Mindwell B200 needs a Blackwell-capable build such as cu130.
Neither has been validated for JEVPFN on the cluster. Check `nvidia-smi` in an allocation and
`ldd --version` on the host; relevant Linux wheels require a compatible glibc baseline.

```bash
# Later, after driver compatibility is confirmed; not needed for exploration:
python -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev,notebooks,models,baselines]'
python -m pip check
mkdir -p output_JEVPFN/experiment_0/manifests
python -m pip freeze > output_JEVPFN/experiment_0/manifests/environment.txt
```

`models` supplies TabPFN-3/3.5 and `baselines` supplies CatBoost/scikit-learn for the focused
Jev comparison. The `tabicl`, `tabdpt` and `tabstar` extras are deferred interface options;
do not install them for the current main comparison. Package
metadata was checked on 30-09-2026; this is not a tested joint environment. Validate imports
and tiny task examples on VSC before selecting the final roster. TabDPT also needs compatible
FAISS wheels. Prefer a separate Python environment for heavier/conflicting integrations,
with its own frozen manifest, rather than changing a working notebook environment.

AutoGluon is excluded by the owner. No environment using its MITRA runtime is prepared;
Mitra-v2 remains deferred until an independent supported integration is verified. Its `autogluon/` checkpoint
namespace is a publisher name; the actual runtime dependency was verified in the official
`mitra_finetune.api.MitraFinetune` and runner code. TabM and Google TabFM are excluded.
Installing packages must not trigger model
weights during notebook execution. Prepare approved weights on a connected host before
using offline compute; use explicit checkpoint paths and record their hashes.

Deferred catalog entries would need separate environment acceptance if later restored to scope:

| Model | Documented environment/interface difference |
|---|---|
| LimiX-2 | Python ≥3.12, Torch 2.9.1 in its official source dependencies; CUDA/attention build must match. |
| Causilo | Python 3.10–3.14, Torch ≥2.13; incompatible with the existing 2.12.1 pin in one environment. |
| EXAONE Tabular | Python ≥3.11, Torch ≥2.6,<3 and NumPy ≥2.3.5; numeric NumPy input and custom estimator lifecycle. |
| ConTextTab | Python 3.11 reference requirements; text encoder plus gated weights; substantial GPU memory. |

Primary references and model capabilities are in [Literature Review](LITERATURE_REVIEW.md).
The notebook environment stays Python 3.12 with no model imports. Prepare one frozen environment
per compatible model group, record its interpreter and full dependency manifest, and select that
environment explicitly in each future Slurm job. No new model environment has been installed here.
Notebooks run only locally; their runner automatically chooses parallelism and saves actual
Jupyter outputs into the source files. No worker argument is needed. Metric utilities use the lightweight
`evaluation` extra; fit/prediction timing must include explicit CUDA synchronization.

## Two storage tiers, both under JEVPFN

| Location | Contents |
|---|---|
| `$VSC_DATA/JEVPFN/` | Git checkout and Python environment metadata. |
| `$VSC_DATA/JEVPFN/output_JEVPFN/` | Compact experiment logs/manifests, scalar metric summaries and pointers to large outputs. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/data/` | Raw data, durable paid responses and feature tables. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/` | Large model weights. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/` | Large prediction/metric tables, feature matrices and experiment results. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/predictions/<run_id>/` | Every individual outer-test or inner-OOF prediction in `predictions.parquet`, plus the full immutable evaluation record. |
| Local `output_JEVPFN/` | Notebook figures/reports, root `All Results.md` and `figures/Captions.md`; notebooks never run on VSC. |

`JEVPFN_STAGING_ROOT` is the **allocation parent**, not its JEVPFN child. The path resolver
inserts the project name exactly once. The current wICE default is inherited from CreditPFN;
confirm access/quota before use. Override it for a different allocation, especially native
GPFS on Mindwell. Never use cross-mounted Lustre for sustained Mindwell I/O or GPFS for wICE I/O.

```bash
# Set JEVPFN_STAGING_ROOT to your verified project allocation parent when overriding the default.
python -c 'from src.utils.paths import describe; print(describe())'
```

Large-file writes never fall back to limited personal DATA. Slurm activation directs
TabPFN, Hugging Face and Torch weight caches to project storage. Other future adapters must
use explicit paths from `checkpoints_dir`, not package defaults under HOME. Node scratch is disposable; never keep the only paid cache there.
Compiler/runtime caches use scratch with a PROJECT fallback, never personal DATA.
Confirm allocation backup policy and test an independent backup of paid responses.
Transfer closed/consistent per-dataset SQLite files, including their embedded feature tables. Verify
`python -m src.data.prepare --offline` after copying the numbered raw folders; do not recreate
`v1` wrappers. GitHub synchronizes code, not datasets or generated features.

Local feature artifacts are permanent in `data/jev_cache/<dataset_id>.sqlite3`: raw responses,
all source origins, compressed numeric/row-request tables and embedded immutable manifests.
Copy each closed file to PROJECT's equivalent `JEVPFN/data/jev_cache/` location and verify its
feature checksums after transfer. Source CSV fingerprints and original `csv:N` row IDs must
match before a join. Select each block explicitly by artifact ID; experiment-0 `feature_artifacts`
accepts `dataset_id`/`artifact_id` mappings. DATA holds compact discovery indexes and logs under
`output_JEVPFN/feature_creation/`. Ordinary output cleanup never touches paid dataset files.

The prediction writer rejects explicit DATA destinations on VSC. Notebook stdout/stderr and
rich outputs stay local; Slurm experiment logs use DATA. No per-row
probabilities or raw Jev text belong in operational logs. This routing has local simulated-tier
tests; actual allocation access/quota and interrupted transfer checks belong to experiment 0.

## Experiment 0

Before any allocation, run the local offline preparation in the project environment:

```powershell
.\.venv\Scripts\python.exe -m src.cluster.preflight --local --verify-data --tests
```

This verifies all 18 selected datasets, shared five-fold feasibility and synthetic contracts.
It saves a unique `preflight_<id>.json` and test log under `experiment_0/manifests` and `logs`.
It does not probe-write raw inputs or paid caches. Corrupt feature references are recorded as
failures without losing the rest of the report. Explicit pending acceptance checks in
`config/experiment_0/debug.yaml` cover real provider contracts, context/robustness, budgets and
recovery, complete artifacts, transfer/backup, allocation/storage, checkpoint/runtime identity,
real predictor tasks, CV boundaries, full-fold resources and interrupted output recovery.
These checks require real evidence; a preflight never sets full Experiment 0 acceptance true.

Shared questions are permanent inputs at `data/decision_inputs/questions/<bank_id>/<dataset_id>.json`,
with the complete `<bank_id>.json` beside that directory, on
PROJECT when running on VSC. Transfer the chosen immutable bank alongside closed dataset
feature stores. Raw responses from different providers remain distinct, and every future
provider-specific feature artifact must identify the question and checkpoint used.

Use the existing CreditPFN login/account convention and verify the current Slurm association.
The prepared account/partitions are in `config/experiment_0/debug.yaml` and the Slurm scripts.
The [wICE guide](https://docs.vscentrum.be/leuven/wice_quick_start.html) documents the resources;
the pinned [VSC snapshot](<../tfm-library/repositories/VSC Documentation.txt>) records the source
sections `wice_quick_start.rst`, `mindwell_quick_start.rst` and `kuleuven_storage.rst`.

| Check | Candidate resource |
|---|---|
| CPU environment, hashes, deterministic tests | wICE interactive, 2 cores / 8 GB / 10 minutes. |
| CUDA/import preflight | wICE gpu_a100_debug, one full A100 / 10 minutes. |
| Later model profiling | wICE gpu_a100 first; select RAM/time from measurements. |
| Memory-heavy alternative | Mindwell gpu_b200 with native GPFS and compatible CUDA, if profiling justifies it. |

The wICE interactive GPU slice is not a full A100. Check current availability/account limits;
no allocation is automatically launched. After local features are complete and execution approved:

```bash
cd "$VSC_DATA/JEVPFN"
mkdir -p output_JEVPFN/experiment_0/logs
sinfo --clusters=wice
sbatch scripts/slurm/experiment_0_cpu.slurm
sbatch scripts/slurm/experiment_0_gpu.slurm
```

The prepared preflights check CPU data/tests and CUDA imports/tiny tensors. They do not load
model weights, fit estimators or establish full experiment-0 acceptance. Future checks must
cover all selected engines/tasks/class limits, feature joins, missing values, five-fold split
identity, validation-only threshold selection, fold-local preprocessing, restarts and resource
budgets. Do not call a passing infrastructure check a completed benchmark debug phase.

Optional `python -m src.cluster.preflight --network` inside an allocation tests outbound
DNS/TLS/HTTPS with one credential-free HEAD request. No inference POST is sent. A login-node
success does not establish compute access, and an HTTP error response does not establish
billing/authentication. This check is not needed for the chosen local Jev workflow.
