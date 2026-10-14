# VSC setup

Current status: the owner's 14 October transcript confirms the checkout, writable PROJECT
directories and staged tokenizers. It also reveals package installation into an inherited
TabPFNCredit venv, requiring the interpreter correction below. No JEVPFN allocation or model
run appears in that transcript. Hosted extraction runs
locally; Liquid, Perplexity and Strands use separate VSC GPU environments after a small pilot.
VSC prediction consumes completed feature tables offline. Hosted API access from compute
nodes is unnecessary for this workflow and remains unverified.

## Self-hosted decision extractors

Self-hosting Liquid d1-3B, Strands Decider 2B or Perplexity Decider 27B removes the vendor
API token fee; it does not remove weight licence conditions, Slurm credit charges, setup,
storage or inference time. The six-provider comparison remains disabled. Hosted extraction
will run locally; separate VSC extraction jobs serve the selected open checkpoints after
runtime validation. No allocation has been launched.

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
API connectivity. PROJECT write access is confirmed; allocation access, quota, backup and
CUDA compatibility remain untested.

## First checkout through Open OnDemand

First publish the current local changes yourself. GitHub cannot supply unpushed code.
In [KU Leuven Open OnDemand](https://ondemand.hpc.kuleuven.be/), choose **Clusters → Login
(Server) Shell Access**. The pinned guide's `compute/portal/ondemand/login-shell.rst`
documents that this is a login shell, not a compute allocation.

For a new checkout (do not run `git clone` over an existing folder):

```bash
cd "$VSC_DATA"
git clone --recurse-submodules https://github.com/andreasgoethals/JEVPFN.git JEVPFN
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
```

The clone sets `origin` automatically. Later update clean checkouts with `git pull --ff-only`
and `git submodule update --init --recursive`. Do not reset local changes on VSC.
The project allocation parent must exist and be accessible; setup refuses a missing mount.

## Environment

Use the CreditPFN Conda bootstrap with a separate Python 3.12 environment named **JEVPFN**.
Dependencies remain managed by pip and `pyproject.toml`; do not share CreditPFN's environment.
Run setup on an approved connected setup/login session, not inside an experiment:

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda create -n JEVPFN python=3.12 pip  # Skip if it already exists.
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export PIP_CACHE_DIR="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/pip_cache"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh && "$JEVPFN_PYTHON" -m pip install -e '.[dev,token-audit,staging]'
"${JEVPFN_PYTHON:?Activation must succeed first}" -m pip check
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
"$JEVPFN_PYTHON" -m src.cluster.token_audit --stage-only
```

The activation helper clears inherited venv/Python overrides and shell aliases/functions,
selects the Conda environment, verifies the actual Python 3.12 prefix, and prints its path
**before installing anything**. `/data` and `/vsc-hard-mounts` aliases are resolved before
comparison. Use the printed interpreter explicitly for installation; a prompt saying
`(JEVPFN)` alone is insufficient. Slurm jobs use the same helper. If a previous install
affected another project's environment, restore that environment separately from its own
recorded requirements; do not try arbitrary downgrades or install JEVPFN into it again.

`setup` creates/probes both JEVPFN trees and the Slurm log directory, writes a compact DATA
manifest, and submits nothing. `--stage-only` downloads about 58 MB of tokenizers/source,
not weights. Transfer the inputs as below before submitting CPU checks.

## Transfer inputs when PROJECT is absent from the file browser

The Open OnDemand Files menu does not determine filesystem permissions. The login shell
can access PROJECT directly. Copy the local raw directories and 18 closed prepared SQLite
files to PROJECT; neither is in Git. Stop question preparation before copying, so the SQLite
files form complete snapshots. Preserve names and keep raw Zomato/Spotify for exploration.

On Windows PowerShell, from the repository directory, with an SSH key registered to the VSC
account (replace `YOUR_VSC_LOGIN` and use the key file actually registered):

```powershell
$jevpfnRemote = 'YOUR_VSC_LOGIN@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/data/'
scp -i "$env:USERPROFILE\.ssh\id_rsa_vsc" -r .\data\raw .\data\decision_inputs $jevpfnRemote
```

An Open OnDemand browser login is separate from SSH-key authentication. Follow the official
[Windows key setup](https://docs.vscentrum.be/accounts/generating_keys_windows.html), register
the **public** `.pub` key through the [VSC account page](https://account.vscentrum.be/), and
complete the required VSC MFA flow. Never upload the private key. The official
[SCP/SFTP guide](https://docs.vscentrum.be/data/transfer/scp_sftp.html) documents recursive
copies; [Globus](https://docs.vscentrum.be/data/transfer.html) is the preferred alternative
for larger or interruption-prone transfers. Use absolute remote paths in PowerShell:
`$VSC_DATA` is a remote variable and must not be expanded by Windows.

## CPU checks before downloading weights

The full token audit runs in a CPU job, not on the login node:

```bash
sbatch scripts/slurm/experiment_0_cpu.slurm
squeue --me --clusters=wice
# After reviewing a successful CPU-preflight log:
sbatch scripts/slurm/experiment_0_tokens.slurm
```

The token job resumes after interruption. It does not remove source rows. Every self-hosted
input includes task definitions, text, question, options and publisher formatting. Jev/Clef/
Microsoft hosted counts remain unverified until their actual server tokenization can be checked.

After selecting the GPU runtime in experiment 0, optional model extras can be installed
into a separate PROJECT environment (see the Liquid environment example below), with
`JEVPFN_CONDA_ENV` set to that absolute prefix. Keep CUDA dependencies out of the CPU environment.
The existing TabPFN/TabICL extras pin Torch 2.12.1. Select a CUDA wheel compatible with the allocated
GPU and driver before installing extras: see the [official PyTorch wheel table](https://pytorch.org/get-started/previous-versions/).
The wICE A100 candidate is cu126; Mindwell B200 needs a Blackwell-capable build such as cu130.
Neither has been validated for JEVPFN on the cluster. Check `nvidia-smi` in an allocation and
`ldd --version` on the host; relevant Linux wheels require a compatible glibc baseline.

```bash
# Later, after driver compatibility is confirmed; not needed for exploration:
python -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev,models,baselines]'
python -m pip check
mkdir -p "output_JEVPFN/3. experiment_0/manifests"
python -m pip freeze > "output_JEVPFN/3. experiment_0/manifests/environment.txt"
```

`models` supplies TabPFN-3.5; `baselines` supplies CatBoost/scikit-learn. The other selected
extras are `tabicl`, `limix`, `kumo`, `tabstar` and `contexttab`. Package/source interfaces were
rechecked on 13-10-2026; this is not a tested joint environment. Install each custom predictor
in its own PROJECT Python 3.12 environment, for example `environments/limix` with
`"$JEVPFN_PYTHON" -m pip install -e '.[evaluation,limix]'`.
Set `JEVPFN_CONDA_ENV` before Slurm activation to select it. Pin task checkpoint paths/hashes
and installed versions in `config/4. experiment_1/models.yaml`; factory validation fails
before loading an unpinned checkpoint. Freeze the complete resolved environment after the pilot.

AutoGluon is excluded by the owner. No environment using its MITRA runtime is prepared;
Mitra-v2 remains deferred until an independent supported integration is verified. Its `autogluon/` checkpoint
namespace is a publisher name; the actual runtime dependency was verified in the official
`mitra_finetune.api.MitraFinetune` and runner code. TabM and Google TabFM are excluded.
Installing packages must not trigger model
weights during notebook execution. Prepare approved weights on a connected host before
using offline compute; use explicit checkpoint paths and record their hashes.

Selected custom predictors require these additional checks:

| Model | Documented environment/interface difference |
|---|---|
| LimiX-2 | Current source requires Python ≥3.12 and Torch ≥2.9.1. Use the V2 no-retrieval classifier/regressor JSON; archive its contents/hash with the run settings. |
| NVIDIA Kumo Tabular | SDM package and local task-specific Kumo weights. Uses complete context, query batches and the publisher's eight-estimator recipe; rejects silent selection above 500 features. This is not KumoRFM. |
| TabSTAR | Stage the task model and language encoder; standard supervised adaptation uses a validation subset inside training. Each fit writes its own PROJECT training directory. |
| ConTextTab | Stage SAP RPT-1 OSS and its sentence encoder. The adapter overrides default 8,192-row context sampling with the full training fold and one bag; memory may exceed available VRAM. Its text encoder truncates individual strings at 512 tokens. |

Primary references and model capabilities are in [Literature Review](LITERATURE_REVIEW.md).
The notebook environment stays Python 3.12 with no model imports. Prepare one frozen environment
per compatible model group, record its interpreter and full dependency manifest, and select that
environment explicitly in each future Slurm job. No new model environment has been installed here.
Notebooks run only locally; their runner automatically chooses parallelism and saves actual
Jupyter outputs into the source files. No worker argument is needed. Metric utilities use the lightweight
`evaluation` extra; fit/prediction timing must include explicit CUDA synchronization.

## Decision-provider connection recipes

Checked 13-10-2026 against publisher documentation/source. These describe later pilot setup;
no live integration has passed acceptance. The exact saved input can be inspected now:

```powershell
.\.venv\Scripts\python.exe -m src.jev.preparation --preview BIN_TEXT_FAKE_JOB_POSTING --row 0
```

The `state` carries task definitions and named text values; `questions` carries instructions
and answer options. `src.jev.alternatives.preview` renders the six corresponding calls.
Its placeholders never contain credentials. Use saved requests instead of hand-writing prompts.

| Provider | Setup and connection | Context / price reference |
|---|---|---|
| [Jev](https://docs.typesafe.ai/models) | Local `pip install -e '.[jev]'`; set `TYPESAFE_API_KEY` privately. Pinned `jev-1.13.0`, `POST https://api.typesafe.ai/v1/systemone`, Bearer authorization. The guarded transport uses `TypeSafeClient.system_one(state, questions)`. | 32k state + question for our one-question calls; $0.042/M input tokens. Confirm account quotas. |
| [Cloudflare Clef-flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/) | Workers AI account, Account ID and token with inference access. POST to `/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash`; Bearer `CLOUDFLARE_AUTH_TOKEN`; body `model`, `state`, `questions`. The [public Apache-2.0 checkpoint](https://huggingface.co/Cloudflare/clef-flash) now provides an alternative self-hosted route; this project's Clef staging/tokenizer/runtime adapter is not yet implemented. | Hosted 24,576 tokens; $0.038/M input tokens. It can silently truncate; reject overlong inputs beforehand. Self-hosting removes vendor token fees, but requires a new exact-token audit and GPU pilot. Hosted/checkpoint parity is not assumed. |
| [Microsoft-Decision-1](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) | Foundry subscription/project and a version-1 deployment. POST to the resource endpoint plus `/providers/microsoft/v1/systemone`; `api-key` header, `model` set to the deployment name. Store the returned underlying model too. | 32,768 is the OpenRouter reference, not yet verified for our Foundry deployment. Published input reference $0.042/M. Hosted weight immutability remains unresolved. |
| [Liquid d1-3B](https://huggingface.co/LiquidAI/d1-3B) | Own Python 3.12/CUDA environment, `pip install -e '.[liquid]'`. Load the staged pinned directory using `AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16).to('cuda')`; call `model.system_one(state, questions)`. Review custom code before loading. | 32,768 tokens. No vendor inference fee; LFM Open License and GPU credits apply. Use its decision interface, not autogenerated chat-serving examples. |
| [Perplexity Decider v1.1-27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | Own Python 3.12/CUDA environment; install the pinned checkpoint's `requirements.txt` and this project. Add `<checkpoint>/source/src` to Python's path; use `autojev.model.DecisionModel(checkpoint, device='cuda')`, `predict([{'state': ..., 'question': ...}])`, then `answer(question, probabilities)`. | 8,192 tokens; preserve supplied attention mode/calibration. About 49 GiB weights plus working memory. No self-hosted token fee; Apache-2.0. Public hosted access was not established. |
| [Strands Decider](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/docs/inference.md) | Own Linux CUDA environment; `pip install -e '.[strands]'`, then `pip install causal-conv1d --no-build-isolation` with a matching CUDA toolkit. Stage both adapter and base checkpoint. `strands-decider serve <checkpoint> --device cuda --port 8000 --strict-window`; POST `state`/`questions` to `http://127.0.0.1:8000/v1/systemone`. | 4,096 tokens. Serialize calls per engine; loopback server has no authentication. No vendor inference fee; Apache-2.0 and GPU credits apply. |

All listed prices are input-only references, to recheck before approval. API output is not
charged under these listed plans. Self-hosted GPU hours are a separate cost. The common
semantics are Noul P(positive), Choice P(each class), or nine Score probabilities; regression
uses their weighted mean on -4..4. Do not use the models to generate explanatory text.

For later checkpoint staging, install `.[staging]` on the connected setup host and use the
staging utility. It pins revisions from `extractors.yaml`, redirects Hugging Face/Xet caches
to PROJECT, and creates the checkpoint hash manifest required by the GPU runner:

```bash
# On the connected login/setup host, after selecting the verified CPU environment.
# These explicit commands download large weights directly to PROJECT, not Windows.
"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider liquid
# Later, when preparing each model's pilot:
"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider strands
"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider perplexity
```

For Perplexity use `--provider perplexity`; install its downloaded requirements before the
project's GPU extras. For Strands use `--provider strands`; the utility stages its configured
adapter **and** `Qwen/Qwen3.5-2B-Base` at the two configured revisions. Its base revision is
publisher-reconstructed provenance, not an original training pin. The base stays in the
Hugging Face cache because the adapter loads that model ID.
Record every staged file checksum and the complete environment. Set both `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` in compute jobs. No inference belongs on a login node.

Use one full A100 at batch size one for initial profiles. The existing 10-minute/32-GB-RAM
GPU preflight only checks imports and tiny tensors; it is not a 27B-model Slurm allocation.
The separate decision-pilot script uses the one-hour A100 debug partition with 128 GB host
RAM. Use the normal GPU partition for longer profiling or full extraction after reviewing
pilot measurements; an import check alone does not establish model feasibility.

`python -m src.jev.context` retains the historical approximate screen. The actual Experiment 0
test is `python -m src.cluster.token_audit`; CPU counts are stored transactionally per input
and checkpoint. It uses the pinned Liquid `prompt.render`/single-question route, Strands
`render_state` plus `render_question` and `_fit(strict_window=True)`, and Perplexity
`decision_messages` plus its saved chat template. No generic chat prompt is substituted.
The renderer sources and tokenizer files are fingerprinted; staged weights must carry the
same tokenizer. Runtime input-token counts must match before an answer is accepted.

`src.jev.self_hosted` implements direct, serialized local inference for all three providers;
Strands uses `SystemOneEngine` without an HTTP server. Output normalization and materialization
have synthetic tests, but live GPU contracts remain pending. Empty inputs never enter the
queue; all source-row links remain in the prepared dataset database. Failure is not a missing
text feature. Each completed response is committed before advancing; a restart reuses it.

For the first Liquid pilot, create an isolated Python 3.12 environment on PROJECT. The
small CPU environment can stay on DATA; CUDA environments and package/wheel caches are
large and belong on PROJECT. For example, on the connected setup host:

```bash
export CONDA_PKGS_DIRS="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/conda_packages"
export PIP_CACHE_DIR="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/pip_cache"
conda create --prefix "$JEVPFN_STAGING_ROOT/JEVPFN/environments/liquid" python=3.12 pip
export JEVPFN_CONDA_ENV="$JEVPFN_STAGING_ROOT/JEVPFN/environments/liquid"
source scripts/activate_conda.sh
```

Install the compatible CUDA Torch wheel and `.[dev,token-audit,liquid]`, then on the connected
setup host run `"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider liquid`. This is an explicit large
download into PROJECT, with file hashes. Set `enabled: true` in
`config/3. experiment_0/self_hosted.yaml` only when ready for the GPU test, then submit:

```bash
export JEVPFN_CONDA_ENV="$JEVPFN_STAGING_ROOT/JEVPFN/environments/liquid"
export JEVPFN_PROVIDER=liquid
sbatch scripts/slurm/experiment_0_decision_pilot.slurm
```

The pilot covers first/middle/last rows and ten longest exact requests per dataset, capped
by a configurable 1,000-request compute ceiling. The script requests one A100, 128 GB host
RAM and one hour; verify account/partition access. Only one A100 debug job may be queued at
a time. Repeat with PROJECT `environments/strands` and `environments/perplexity` and their
pinned runtime dependencies. `JEVPFN_CONDA_ENV` accepts either a Conda name or an absolute prefix.
Perplexity's checkpoint requirements pin Torch 2.14.0 and Transformers 5.17.0; do not install
the downstream TabPFN Torch pin into that environment. Strands requires its separate base
checkpoint; the staging utility puts it in the PROJECT Hugging Face cache and verifies it.
`self_hosted --full --execute` additionally requires the full-extraction switch and a reviewed
acceptance manifest matching the runtime-settings hash and token-audit ID. No command creates
that acceptance automatically. A completed pilot is evidence to review, not full Exp0 acceptance.

## Two storage tiers, both under JEVPFN

| Location | Contents |
|---|---|
| `$VSC_DATA/JEVPFN/` | Git checkout and Python environment metadata. |
| `$VSC_DATA/JEVPFN/output_JEVPFN/` | Compact experiment logs/manifests, scalar metric summaries and pointers to large outputs. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/data/` | Raw data, durable paid responses and feature tables. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/` | Large model weights. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/environments/` | Separate large CUDA/model environments; Conda and pip package caches also use PROJECT. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/` | Large prediction/metric tables, feature matrices and experiment results. |
| `$JEVPFN_STAGING_ROOT/JEVPFN/output_JEVPFN/<phase>/results/predictions/<run_id>/` | Every individual outer-test or inner-OOF prediction in `predictions.parquet`, plus the full immutable evaluation record. |
| Local `output_JEVPFN/` | Notebook figures/reports, root `All Results.md` and `figures/Captions.md`; notebooks never run on VSC. |

`JEVPFN_STAGING_ROOT` normally names the **allocation parent**; its existing JEVPFN child is
also accepted without doubling the folder name. The current wICE default is inherited from CreditPFN;
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
Copy a closed file to PROJECT's equivalent `JEVPFN/data/jev_cache/` location only when no
independently populated destination exists, and verify its feature checksums after transfer.
Local API and VSC GPU runs can populate different providers in the same dataset filename:
ordinary SCP replacement would discard one side. A validated, conflict-rejecting cache import
and backup/restore test is still required before combining those independently created files.
Until that exists, use one authoritative file/writer at a time and do not synchronize concurrent
copies by overwriting. Source CSV fingerprints and original `csv:N` row IDs must
match before a join. Select each block explicitly by artifact ID; experiment-0 `feature_artifacts`
accepts `dataset_id`/`artifact_id` mappings. DATA holds compact discovery indexes and logs under
`output_JEVPFN/2. feature_creation/`. Ordinary output cleanup never touches paid dataset files.

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
It saves a unique `preflight_<id>.json` and test log under `3. experiment_0/manifests` and `logs`.
It does not probe-write raw inputs or paid caches. Corrupt feature references are recorded as
failures without losing the rest of the report. Explicit pending acceptance checks in
`config/3. experiment_0/debug.yaml` cover real provider contracts, context/robustness, budgets and
recovery, complete artifacts, transfer/backup, allocation/storage, checkpoint/runtime identity,
real predictor tasks, CV boundaries, full-fold resources and interrupted output recovery.
These checks require real evidence; a preflight never sets full Experiment 0 acceptance true.

Prepared inputs are the 18 permanent `data/decision_inputs/<dataset_id>.sqlite3` files, on
PROJECT when running on VSC. Each contains versioned questions, exact unique text groups and
all original-row links; there is no separate bank JSON to transfer. Copy closed input files
alongside the closed dataset feature stores. Raw responses from different providers remain
distinct, and every future feature artifact must identify its input set, question and checkpoint.
Hosted inference is planned locally; self-hosted extractors need a small VSC GPU compatibility
pilot before full extraction. The later experiment-0 checks must also validate transfer and
downstream feature joins. Open weights still consume the allocation's GPU resources.

Use the existing CreditPFN login/account convention and verify the current Slurm association.
The prepared account/partitions are in `config/3. experiment_0/debug.yaml` and the Slurm scripts.
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
no allocation is automatically launched. Experiment 0A runs before feature creation; 0B requires
explicit completed artifact selections (`--stage after_extraction`). For infrastructure tests:

```bash
cd "$VSC_DATA/JEVPFN"
mkdir -p "output_JEVPFN/3. experiment_0/logs"
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
