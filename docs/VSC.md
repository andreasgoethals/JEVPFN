# VSC setup

Current status: no JEVPFN allocation or model run has been submitted. Local Jev feature
creation precedes experiment 0. VSC prediction consumes completed feature tables offline;
Jev API access from compute nodes is unnecessary for this workflow and remains unverified.

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
python -m pip install -e '.[dev,notebooks,models,baselines,tabicl]'
python -m pip check
mkdir -p output_JEVPFN/experiment_0/manifests
python -m pip freeze > output_JEVPFN/experiment_0/manifests/environment.txt
```

`models` supplies TabPFN-3/3.5, `baselines` supplies CatBoost/scikit-learn, and `tabicl` supplies
TabICL v2. Optional `tabdpt` and `tabstar` extras are declared separately. Package
metadata was checked on 30-09-2026; this is not a tested joint environment. Validate imports
and tiny task examples on VSC before selecting the final roster. TabDPT also needs compatible
FAISS wheels. Prefer a separate Python environment for heavier/conflicting integrations,
with its own frozen manifest, rather than changing a working notebook environment.

AutoGluon is excluded by the owner. No environment using its MITRA runtime is prepared;
Mitra-v2 remains deferred until an independent supported integration is verified. Its `autogluon/` checkpoint
namespace is a publisher name; the actual runtime dependency was verified in the official
`mitra_finetune.api.MitraFinetune` and runner code. TabFM still needs a source pin/backend.
Installing packages must not trigger model
weights during notebook execution. Prepare approved weights on a connected host before
using offline compute; use explicit checkpoint paths and record their hashes.

New catalog entries need separate environment acceptance, not a single installation of every extra:

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
The notebook runner selects parallelism automatically, respecting CPU affinity and
`SLURM_CPUS_PER_TASK`. No worker argument is needed. Metric utilities use the lightweight
`evaluation` extra; fit/prediction timing must include explicit CUDA synchronization.

## Two storage tiers, both under JEVPFN

| Location | Contents |
|---|---|
| `$VSC_DATA/JEVPFN/` | Git checkout and Python environment metadata. |
| `$VSC_DATA/JEVPFN/output_JEVPFN/` | `All Results.md`, `Captions.md`, shared `figures/<phase>/<notebook>/`, phase logs/reports/manifests. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/data/` | Raw data, durable paid responses and feature tables. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/` | Large model weights. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/` | Large audit tables, predictions and experiment results. |

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
Confirm allocation backup policy and test an independent backup of paid responses.
Transfer a closed/consistent SQLite database and complete immutable feature folders. Verify
`python -m src.data.prepare --offline` after copying the numbered raw folders; do not recreate
`v1` wrappers. GitHub synchronizes code, not datasets or generated features.

## Experiment 0

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
