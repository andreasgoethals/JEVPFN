# VSC setup and experiment-0 checks

Checked 30-09-2026 against current package metadata and official VSC/PyTorch documentation,
alongside the synced TFM Library at
`81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba` and the existing CreditPFN checkout.
**No JEVPFN allocation has been submitted; compute-node networking/CUDA remain unverified.**

## Sources and inherited conventions

The read-only [VSC documentation snapshot](<../tfm-library/repositories/VSC Documentation.txt>)
contains `source/leuven/wice_quick_start.rst`, `source/leuven/mindwell_quick_start.rst`,
`source/leuven/tier2_hardware/kuleuven_storage.rst` and the SSH/Slurm sections. Refer to these
embedded section names: snapshot line numbers are not stable.

CreditPFN's `scripts/slurm/_activate_env.sh`, experiment Slurm scripts, `docs/VSC.md` and
`docs/AGENTS_MEMORY.md` establish these conventions:

- Slurm credit account **`lp_verbekelab`**. Verify current association before submitting;
  an old `lp_mindwell_pilot` attempt is a recorded failure, not an account to reuse.
- Conda bootstrap under `$VSC_DATA/miniconda3`; separate environment **JEVPFN**.
- Repository `$VSC_DATA/JEVPFN`; wICE project storage under
  `/lustre1/project/stg_00211/JEVPFN` if this allocation is available to the account.
- Explicitly select `$CONDA_PREFIX/bin/python` after activation; an inherited venv can otherwise
  shadow Conda. `_activate_env.sh` checks the selected interpreter and Python 3.12.
- Use `--gpus-per-node`, following CreditPFN's working scheduler convention.

The current [wICE quick start](https://docs.vscentrum.be/leuven/wice_quick_start.html)
confirms the `gpu_a100_debug` full-GPU partition and its one-hour maximum.

No user-specific SSH alias/login was present in the inspected setup. Use the same authenticated
session or SSH login you already use for CreditPFN. The documented KU Leuven login endpoint is
`login.hpc.kuleuven.be`. For example, replacing `YOUR_VSC_LOGIN`:

```powershell
ssh YOUR_VSC_LOGIN@login.hpc.kuleuven.be
```

The environment has no ready authenticated VSC session. No credential discovery or password
export was attempted; the existing interactive login is sufficient for the commands below.

## Appropriate cluster by phase

| Work | Initial recommendation | Reason and qualification |
|---|---|---|
| Local notebook inspection | Local Python 3.12 / JEVPFN | No GPU or cluster credits needed. |
| Exp0 networking, hashes, CPU tests | wICE `interactive`, 2 cores / 8 GB / 10 min | Small diagnostic allocation; docs allow up to 8 cores and 16 hours here. |
| Future large CPU audits | wICE CPU `batch` | Profile RAM first; Jev feature creation is local. |
| Exp0 Torch/TabPFN GPU check | wICE `gpu_a100_debug`, one full A100 | Full GPU; documentation permits up to one hour. Script requests 10 min. |
| Main TabPFN inference later | Start by profiling wICE `gpu_a100` | Choose memory/time from exp0; H100/B200 is an option if measurements justify it. |
| Memory-intensive alternative | Mindwell `gpu_b200` | Use native GPFS storage and a supported CUDA build; do not copy CreditPFN's large training request by default. |

The wICE interactive GPU resource is a 1/7 A100 slice, not a full A100; do not assume it can
hold TabPFN for the largest tasks. Mindwell has a separate RTX 5000 Ada interactive resource.
Scheduler limits and account access can change. Check `sinfo` and the current associations.

## Environment: same pyproject, separate Python environments

Bash below runs on VSC. Reuse CreditPFN's Conda installation, **not its CreditPFN environment**.
The project still uses pip/pyproject for dependencies; no duplicate requirements or Conda package
specification is introduced.

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda create -n JEVPFN python=3.12 pip
conda activate JEVPFN
cd "$VSC_DATA/JEVPFN"
python -m pip install -e '.[dev,notebooks]'
python -m pip check
python -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

If the environment already exists, skip `conda create`. Perform installs on an approved connected
setup node/session. Do not install packages in a running experiment or change a shared environment.

For the later GPU package check, first inspect `nvidia-smi` in an allocated GPU session. The
[official PyTorch wheel table](https://pytorch.org/get-started/previous-versions/) lists the
following candidate matching the Torch 2.12 family already used by CreditPFN:

```bash
# Candidate for wICE A100, only after confirming driver compatibility:
python -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev,notebooks,jev,models]'
python -m pip check
mkdir -p output_JEVPFN/experiment_0/manifests
python -m pip freeze > output_JEVPFN/experiment_0/manifests/environment.txt
```

For Mindwell B200, select a Blackwell-capable build (for example the documented `cu130` build
of the same Torch release) after checking its actual driver. A wheel's availability does not
certify that runtime. Do not install `torchvision`, `torchaudio`, CatBoost or TabSTAR for these
notebooks; none is needed. `models` pins TabPFN 9.0.0, which includes `ModelVersion.V3_5` in
[the official implementation snapshot](<../tfm-library/repositories/TabPFN .txt>). The future
estimator should select that version explicitly, not rely on a moving default.

The `notebooks` extra explicitly includes IPython, ipykernel, JupyterLab, nbclient and nbformat;
`dev` adds pytest and Ruff. No separate cluster requirements file is necessary. `models` pins
Torch 2.12.1; a matching `+cu126` or `+cu130` wheel satisfies that pin without a package upgrade.
The CUDA choice remains a host-specific installation step, not a global notebook dependency.

Local notebook dependencies are installed and tested. The SDK/Torch/TabPFN extras are declared,
but **have not been installed or validated on VSC**. The GPU preflight imports packages and does
one tiny CUDA tensor multiplication; it never constructs an estimator, fits data or loads weights.
Weights/authentication/licence acceptance and the eventual small model smoke are separate steps.

Published package metadata was checked for Python 3.12: TabPFN 9.0.0 and TypeSafe SDK 0.7.2
have platform-independent wheels; Torch 2.12.1 and PyArrow 25.0.1 have Windows and Linux x86-64
wheels. Their Linux wheels use the `manylinux_2_28` baseline: check `ldd --version` on the actual
host rather than assuming every VSC environment meets it. Record the final resolved environment
on that host; the local `pip check` is not a substitute for this verification.

## Storage and transfer

On wICE, `src/utils/paths.py` uses the same two-tier convention as CreditPFN:

- `$VSC_DATA/JEVPFN/output_JEVPFN/<phase>/`: figures, reports, captions, logs and small manifests; code also lives under `$VSC_DATA/JEVPFN`.
- `/lustre1/project/stg_00211/JEVPFN`: raw data, durable Jev cache and weights; large tables go to `output_JEVPFN/<phase>/results/`.
- Project storage failure stops the run; no fallback writes large files to personal DATA.
- Node scratch: disposable runtime/compiler caches only.

The Slurm activator sets `TABPFN_MODEL_CACHE_DIR` to the resolved project
`checkpoints/tabpfn` directory, so future weights persist across jobs. It does not download or
create weights. Compiler/plotting caches can still use node scratch.

The snapshot lists personal DATA backups and a 75 GB quota; scratch may be purged after 30 days
without access. It does **not** verify backup/quota policy for `stg_00211`. Establish a separate
backup/restore plan for paid responses before extraction. Never put the only paid cache on scratch.

Override the storage **parent**, not the JEVPFN subdirectory:

```bash
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
python -c 'from src.utils.paths import describe; print(describe())'
```

For Mindwell, set `JEVPFN_STAGING_ROOT` to your confirmed project allocation on native GPFS.
The VSC documentation requires intensive Mindwell I/O on GPFS and wICE I/O on Lustre;
cross-mounted storage is for transfers, not sustained model I/O. No GPFS allocation path is
invented here.

Copy the numbered raw folders to the resolved `data/raw/` location, placing CSV/metadata directly in each dataset folder and preserving the hash
manifests. Copy a **closed/consistent** response database and complete feature directories when
transferring a cache; do not copy an actively written SQLite file. Verify data with
`python -m src.data.prepare --offline`. The library travels through `git clone --recurse-submodules`
after GitHub publication, or `git submodule update --init --recursive` in an existing clone.

## Compute-node API connectivity: verify before paid work

The owner decided to complete the small tests and full Jev feature creation locally, before
experiment 0. Transfer the saved feature tables to VSC for debugging and prediction. Jev API
connectivity on VSC is unnecessary for this chosen workflow; the check below remains optional.

Jev is a hosted API. Its Python client needs outbound DNS/TLS/HTTPS, not a local GPU. The inspected
VSC documentation does not establish a blanket ban or guarantee for outbound HTTPS from all
partitions. A successful login-node request would not prove compute-node connectivity.

The optional `--network` probe resolves `api.typesafe.ai` and sends **one credential-free HEAD** to the
published endpoint. It follows no redirect, sends no body/key, and never makes an inference POST.
An HTTP 401/403/405 can establish an HTTP response, but does not establish authentication, billing
access, or permission for a long extraction job. Review proxy/site restrictions if applicable.

After the environment, data, account and allocation are approved, run from a VSC login session:

```bash
cd "$VSC_DATA/JEVPFN"
mkdir -p output_JEVPFN/experiment_0/logs
sinfo --clusters=wice
sbatch scripts/slurm/experiment_0_cpu.slurm
# After installing/validating the appropriate GPU packages:
sbatch scripts/slurm/experiment_0_gpu.slurm
```

Use the returned job IDs to inspect `squeue` and logs. Reports are written under
`output_JEVPFN/experiment_0/manifests/preflight_<job_id>.json`; `full_experiment_0_passed` stays false.
The CPU script also verifies all 20 raw datasets offline and runs the deterministic tests.
The GPU script tests the actual CUDA runtime without model weights. These commands are prepared
for the owner; the agent has not submitted them.

If already inside an allocated compute session with the JEVPFN environment active:

```bash
python -m src.cluster.preflight --network
python -m src.cluster.preflight --gpu --network
```

If compute HTTPS is allowed and reachable, a later single CPU extraction job is a reasonable
host for Jev calls. If it is blocked, use an approved connected workstation/service for feature
creation and transfer immutable cached artifacts to VSC. Do not add a tunnel or proxy to evade
site policy. The numbered modelling experiments can consume the cache offline either way.

## What remains before an actual run

Account association/quotas, authenticated cluster access, the installed CUDA/TabPFN combination,
full-data storage and backup,
local Jev credentials/rate limits, pilot acceptance and model weights are still pending. No main
experiment or paid feature extraction Slurm script is supplied yet; the live client/protocol
must be agreed first. The original generic `job.slurm` now points to these specific preflights.
