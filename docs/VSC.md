# VSC operations

Use the Open OnDemand login shell for setup/downloads and Slurm allocations for computation.
Notebook analysis always runs locally. Current run evidence is in [AGENTS_MEMORY](AGENTS_MEMORY.md);
the research contract is in [README](../README.md). These instructions follow the read-only
[VSC guide](<../tfm-library/repositories/VSC Documentation.txt>) at library pin
`de324efcca55be392feb0c01830b3b6ce364587f`.

## Checkout and storage

In [Open OnDemand](https://ondemand.hpc.kuleuven.be/), choose **Clusters → Login Shell Access**.
For a new checkout:

```bash
cd "$VSC_DATA"
git clone --recurse-submodules https://github.com/andreasgoethals/JEVPFN.git JEVPFN
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
```

For later owner-published updates, run `git pull --ff-only` and
`git submodule update --init --recursive`. Wait for jobs using the checkout to finish before
updating its code or environment. Do not reset local changes or clone over an existing folder.

| Tier under JEVPFN | Contents |
|---|---|
| DATA: `$VSC_DATA/JEVPFN/` | Checkout, small CPU environment, compact logs/manifests. |
| PROJECT: `$JEVPFN_STAGING_ROOT/JEVPFN/` | Raw/prepared inputs, permanent dataset caches, weights, CUDA environments and package caches. |
| DATA `output_JEVPFN/<number>. phase/` | Operational logs, compact completion indexes and manifests. |
| PROJECT `output_JEVPFN/<number>. phase/` | Token-count databases, predictions, feature matrices, full metric records and other large results. |
| Node scratch | Disposable compiler/runtime caches; never the only saved result. |

Use native Lustre project storage on wICE and native GPFS on Mindwell. Set the project root
to the allocation's existing parent, not a missing mount or personal DATA directory. Large
writes enforce this routing and never fall back to DATA. A filesystem's free space is not
personal/group quota; account credits are not storage quota. Verify current quota and an
independent backup before full extraction. The pinned guide's storage sections explain that
PROJECT can be single-copy; transferring results to the laptop is part of the recovery plan.

## CPU environment and input transfer

The project uses the CreditPFN Conda bootstrap with a separate Python 3.12 environment.
Skip environment creation when JEVPFN already exists:

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda create -n JEVPFN python=3.12 pip
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export PIP_CACHE_DIR="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/pip_cache"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?Activation must succeed}" -m pip install -e '.[dev,staging,token-audit]'
"$JEVPFN_PYTHON" -m pip check
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
"$JEVPFN_PYTHON" -m src.cluster.token_audit --stage-only
"$JEVPFN_PYTHON" -m src.cluster.token_audit --check-only
```

Stop if any command fails. The activation helper clears inherited venv/Python overrides,
checks the actual Conda prefix and prints the interpreter before installation. Use the explicit
`JEVPFN_PYTHON`; a changed prompt alone is insufficient. Keep other projects' environments separate.
Jinja2 is explicitly required for chat-template rendering. `--stage-only` downloads small
pinned tokenizers/source files only; `--check-only` renders all three task formats offline.
VSC CPU tests exclude only the three local notebook tests; notebooks and their tests run locally.

Git does not contain the data. Copy all raw datasets and the 18 closed prepared-input files
from the laptop to PROJECT. Open OnDemand's visible directories do not determine shell permissions.
In Windows PowerShell, from the local repository:

```powershell
$jevpfnRemote = 'YOUR_VSC_LOGIN@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/data/'
scp -i "$env:USERPROFILE\.ssh\id_rsa_vsc" -r .\data\raw .\data\decision_inputs $jevpfnRemote
```

Use the SSH key registered to the account. Open OnDemand browser authentication and SSH/MFA
are separate. Follow [Windows key setup](https://docs.vscentrum.be/accounts/generating_keys_windows.html)
and [SCP/SFTP](https://docs.vscentrum.be/data/transfer/scp_sftp.html); use
[Globus](https://docs.vscentrum.be/data/transfer.html) for larger/interruption-prone transfers.
Never upload a private key. Stop writers before copying SQLite input files.

## Experiment 0A: CPU and exact tokens

Submit from the checkout after setup creates the log directory:

```bash
sbatch scripts/slurm/experiment_0_cpu.slurm
# After successful CPU checks:
sbatch scripts/slurm/experiment_0_tokens.slurm
squeue --me --clusters=wice
sacct --clusters=wice -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

The CPU job uses wICE `interactive`, two cores, 8 GB and ten minutes. It checks datasets,
CV feasibility and synthetic contracts. Token counting uses `batch`, four cores, 16 GB and
four hours. It counts every complete unique input for each pinned open-weight tokenizer,
including task/question/options and publisher formatting. Counts are committed incrementally
and resume after interruption; completed provider summaries survive another worker's failure.
Do not delete count databases to fix a dependency. The fingerprinted tokenizer modules are
kept stable so the current audit remains reusable. Jev's hosted tokenization remains unverified.

Small summaries/logs use DATA `3. experiment_0/{manifests,logs}/before_extraction/`;
large SQLite count databases use PROJECT `3. experiment_0/results/token_audit/`.
A passing CPU/token check is infrastructure evidence; GPU/provider compatibility is separate.

## Checkpoints and isolated GPU environments

Download weights directly onto PROJECT on the connected setup host, without first storing
them on the laptop. The staging utility pins revisions and writes file hashes; no model is loaded:

```bash
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.jev.checkpoints --provider cloudflare
"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider intern
```

Use the same command with `liquid`, `strands` or `perplexity` when preparing those models.
Strands stages both its adapter and the configured Qwen3.5-2B base model. The inferred base
revision caveat is retained in config. Every compute job verifies hashes and runs offline.
No inference belongs on a login node.

For Clef, use `JEVPFN_PROVIDER=cloudflare` and `JEVPFN_EXTRA=clef`. For Intern, use `intern`
for both. Create each environment only once; packages/wheel caches and large environments
belong on PROJECT:

```bash
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export JEVPFN_PROVIDER=cloudflare
export JEVPFN_EXTRA=clef
export JEVPFN_CONDA_ENV="$JEVPFN_STAGING_ROOT/JEVPFN/environments/$JEVPFN_PROVIDER"
export CONDA_PKGS_DIRS="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/conda_packages"
export PIP_CACHE_DIR="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/pip_cache"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda create --prefix "$JEVPFN_CONDA_ENV" python=3.12 pip
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m pip install 'torch==2.12.1' 'torchvision==0.27.1' --index-url https://download.pytorch.org/whl/cu126
"$JEVPFN_PYTHON" -m pip install -e ".[dev,token-audit,$JEVPFN_EXTRA]"
"$JEVPFN_PYTHON" -m pip check
"$JEVPFN_PYTHON" -m src.cluster.token_audit --check-only --provider "$JEVPFN_PROVIDER"
```

CUDA 12.6 is the A100 candidate; verify driver/glibc compatibility in the allocation. The
[official paired wheels](https://pytorch.org/get-started/previous-versions/) establish package
availability, not JEVPFN compatibility. Clef's publisher tested H200 with Torch 2.11 and
Transformers 5.10.2; this candidate keeps Transformers 5.17 aligned with our token audit.
A failure requires diagnosis rather than silent version substitution.

| Open extractor | Runtime preparation |
|---|---|
| Liquid | Separate environment, compatible CUDA wheel and `.[dev,token-audit,liquid]`; staged custom decision implementation. |
| Strands | Separate Linux/CUDA environment and `.[dev,token-audit,strands]`; build `causal-conv1d --no-build-isolation` with matching CUDA toolkit. Direct `SystemOneEngine` inference; no HTTP server needed. |
| Perplexity | Separate environment; install the pinned checkpoint's `requirements.txt` and project dependencies. Keep its declared Torch version separate from downstream/Clef pins. Add staged `source/src` for `autojev`. |
| Clef | `.[dev,token-audit,clef]`; staged joint head/backbone; retain complete softmax precision. |
| Intern | `.[dev,token-audit,intern]`; retain the pinned publisher's calibration temperature and complete probabilities. |

Custom inference code is reviewed and hash-verified before loading. Complete dependency
versions enter response identity. All model and Hugging Face/Xet caches use PROJECT;
HF offline flags are set before runtime imports.

## Bounded GPU contract tests and larger pilots

After exact counts for the chosen provider are complete, submit its prepared environment:

```bash
sbatch scripts/slurm/experiment_0_decision_contracts.slurm
```

The job uses one full A100, four CPUs, 128 GB host RAM and one hour in `gpu_a100_debug`,
with the configured CreditPFN account. Only one debug GPU job may be queued at a time:
wait for Clef to finish before submitting Intern. The pinned guide confirms that this is a
full GPU, unlike the interactive MIG slice. Verify account/partition access on the actual cluster.

The contract job tests binary, three-class and nine-level regression fixtures, a 30-class
vector, missing/duplicate/Unicode input, overflow rejection, cache reopen/reuse, original-row
alignment, and saving/reloading both feature representations. It then evaluates first/longest
unique inputs from each selected dataset. At most 67 forwards are planned, with a hard ceiling
of 100. Runtime token-count agreement, load/inference time and peak GPU memory are recorded.

Synthetic artifacts use PROJECT `3. experiment_0/results/gpu_contracts/<run>/`; sampled real
responses use permanent PROJECT `data/jev_cache/<dataset_id>.sqlite3` for reuse. DATA holds
compact reports/logs under `3. experiment_0/{manifests,logs}/before_extraction/gpu_contracts/`
and the Slurm log directly under the phase's `logs/`. Failed model loading also leaves a report.

A passing contract report requires review and does not set full Experiment 0 acceptance.
Robustness, throughput and remaining provider checks are tracked in `debug.yaml`. Larger
profiling uses `self_hosted.yaml` with its explicit pilot opt-in and
`experiment_0_decision_pilot.slurm`: first/middle/last rows plus ten longest requests per
dataset, capped at 1,000. Full extraction has a separate switch and a reviewed acceptance
manifest matching runtime settings and token-audit identity.

Start serial inference on one wICE A100. Its 80 GiB can hold the Perplexity checkpoint's
approximately 49 GiB BF16 weights, but working memory must be profiled. Use a regular GPU
partition for longer jobs. Consider H100 for measured throughput benefit or Mindwell B200
(192 GiB, native GPFS and a Blackwell-compatible CUDA build) if measured memory justifies it.
GPU hours, quota and licences remain separate from vendor API token fees.

## Local Jev and Experiment 0B

Jev calls run locally after an approved small pilot. Install `.[jev]`, set `TYPESAFE_API_KEY`
privately, and use the pinned `jev-1.13.0` client. `src.jev.extraction` disables hidden retries,
uses durable spending reservations and stops on ambiguous billing outcomes. Config requires
verified account rates, a billing-token upper bound, context validation and explicit dataset/
campaign budgets. Published rates are references, not automatic account authorization.
Microsoft is outside the selected roster.

After full feature creation and validated transfer, Experiment 0B checks complete selected
artifacts and real predictor fits on all task types. Main extras are `models`, `baselines`,
`tabicl`, `limix`, `kumo`; original-text extras are `tabstar`, `contexttab`. Stage/hash their
checkpoints and any text encoders separately; choose exact assets/runtime pins in
`config/4. experiment_1/models.yaml`. Use isolated environments when dependencies conflict.

The injector/factories are synthetically tested; actual full-fold resources remain pending.
ConTextTab uses one bag and the full training context rather than default row sampling;
TabSTAR's default adaptation reserves internal validation for stopping. Kumo rejects
500-column overflow. Record failures without silently reducing rows/features. Pretraining
and prediction-time input availability still need review before interpreting results.

## Transfer results

Wait for all writers to close. In the CPU environment on VSC:

```bash
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.cluster.transfer export
```

The printed ZIP path is under PROJECT `JEVPFN/output_JEVPFN/transfers/`. It snapshots
per-dataset caches, exact counts, predictions and phase outputs from PROJECT plus compact
logs/manifests from DATA. Weights, environments, credentials, raw data and prepared inputs
already held locally are omitted. Export receipts use DATA Experiment 0 manifests.

Download that ZIP with SCP/SFTP/Globus; do not copy remote databases over populated local
ones. In local PowerShell, from the repository:

```powershell
.\.venv\Scripts\python.exe -m src.cluster.transfer import 'C:\path\to\JEVPFN_bundle.zip'
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Import validates every archived path/hash, backs up existing caches and merges each dataset
transactionally. Identical records are reused; conflicting cache records roll back that
entire dataset. A retry safely reuses completed merges. Non-cache output collisions retain
both versions; compatible token summaries are combined and paths rebased for local notebooks.
Keep the original bundle and backups. Imported spending records are historical evidence,
not a replacement for an active budget or extraction queue.
