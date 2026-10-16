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
Keep a stage's code/config/input versions fixed while its jobs run. Reviewed stage-enabling
changes are separate owner commits between stages: pilot, full extraction, predictor contracts
and full-fold profiling each have their own switch. The committed defaults stay disabled.
Rerun checks affected by a change; unrelated edits do not invalidate matching token/runtime
evidence or require resetting output and reinstalling inputs.

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
independent backup before full extraction. Check both bytes and file counts with `myquota`.
An `Errno 122` with available bytes can mean the file-count limit is exhausted: stop runtime
installation, identify usage and request a PROJECT inode/file-quota increase through KU Leuven
HPC support. Do not move CUDA environments to DATA or delete checkpoints to hide that limit.
Do not delete package caches while Conda or jobs are using them. The pinned guide's storage sections explain that
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
```

Stop if any command fails. The activation helper clears inherited venv/Python overrides,
checks the actual Conda prefix and prints the interpreter before installation. Use the explicit
`JEVPFN_PYTHON`; a changed prompt alone is insufficient. Keep other projects' environments separate.
Jinja2 is explicitly required for chat-template rendering. `--stage-only` downloads small
pinned tokenizers/source files only; `--check-only` renders all three task formats offline.
VSC CPU tests exclude only the three local notebook tests; notebooks and their tests run locally.

Storage setup lists the raw-data directory without writing into existing inputs. An absent
raw directory is created empty for a first upload. Prepared-input installation, feature caches,
checkpoints and outputs still require a successful write probe. The setup receipt records
each check separately; actual raw-file readability and hashes are verified by the CPU job.
If an earlier setup fails, fix that check and rerun setup; a completed output reset or package
installation does not need repeating. Do not change raw-data permissions merely to permit a probe.

## Fresh output reset before a new Experiment 0A

Use this section only when deliberately starting a completely new output campaign, not when
resuming the current one after a fix. After updating the checkout/CPU dependencies, verify that `git rev-parse HEAD` matches the
local checkout. Stop notebooks, manual writers and JEVPFN jobs. Before upload installation
or job submission, preview and apply a permanent output reset:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.fresh_start --delete --include-archives
"$JEVPFN_PYTHON" -m src.cluster.fresh_start --delete --include-archives --apply --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
```

The first command previews both tiers. The second permanently deletes their `output_JEVPFN`
and earlier `archived_output_JEVPFN` trees, including old logs, count databases, predictions
and output transfer bundles/backups. It creates no archive. Raw datasets, prepared inputs,
permanent features, weights and environments remain in place. Linked/mounted subtrees and
active JEVPFN jobs are refused before deletion. If deletion is interrupted, stop remaining
writers and repeat the same reset before starting tests; there is no rollback.

For local output, use the same commands with `--local` and the local Python interpreter.
Windows read-only output files/directories are handled automatically, with bounded retries
for sharing locks. A persistent failure leaves a partial cleanup: repeat the cleanup after
closing the program using that path, before continuing. Check `$LASTEXITCODE` after Python
commands in Windows PowerShell; a traceback alone does not prevent the next pasted command.
Repack inputs after clearing local output. Fresh token counting starts from zero. Do not reset
again after tests start. Keep each stage fixed during execution; publish reviewed changes
between stages as described above.

For a final clean validation, finish diagnosing infrastructure failures first, then freeze
the reviewed code and reset outputs once on VSC and locally. Rerun CPU/storage checks, exact
token counts and the selected provider contracts; import the resulting validated transfer
before rerunning local notebooks. Do not import old output bundles into that clean campaign.
Permanent dataset caches remain reusable and are not part of output cleanup. Keep diagnostic
logs until their failure is understood; a clean output folder alone is not evidence of success.

## Upload the prepared inputs

Git does not contain data. The existing VSC raw datasets passed their hash checks and do not
need uploading again after a fresh-output reset. For a first installation only, copy `data/raw/`
to PROJECT `JEVPFN/data/` with SCP. The current question update uses a checked bundle of exactly
the 18 selected SQLite input files; it contains no raw data, responses, credentials or weights.
Stop local question-generation/notebook processes first.

First create the destination from the VSC login shell:

```bash
mkdir -p /lustre1/project/stg_00211/JEVPFN/uploads
```

Then in Windows PowerShell:

```powershell
Set-Location 'C:\Users\U0152019\PhD Documents\Projects\5. JEVPFN\JEVPFN'
.\.venv\Scripts\python.exe -m src.cluster.input_transfer pack
scp '.\output_JEVPFN\transfers\decision_inputs_v3.zip' '.\output_JEVPFN\transfers\decision_inputs_v3.zip.sha256' 'vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/uploads/'
```

This uses the same working SSH authentication as the owner's CreditPFN SCP commands; no
assumed private-key filename.

After SCP finishes, from the activated CPU environment in `$VSC_DATA/JEVPFN`:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.input_transfer install /lustre1/project/stg_00211/JEVPFN/uploads/decision_inputs_v3.zip --writers-stopped
```

The installer checks the bundle/hash, every database, active question bank, row-link counts
and file hashes before replacement. Existing question files are backed up under PROJECT
`output_JEVPFN/transfers/input_backups/`; repeat uploads reuse identical files. Byte-exact
copies preserve input fingerprints across local/VSC token reports. It refuses active/recovery
SQLite sidecars and queued/running JEVPFN jobs; also stop manual writers yourself.

## Experiment 0A: CPU and exact tokens

After the reset and input installation, stage the small pinned tokenizers and check all
three task formats. Submit from the checkout after setup creates the log directory:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.token_audit --stage-only
"$JEVPFN_PYTHON" -m src.cluster.token_audit --check-only
sbatch scripts/slurm/experiment_0_cpu.slurm
# After successful CPU checks:
sbatch scripts/slurm/experiment_0_tokens.slurm
squeue --me --clusters=wice
sacct --clusters=wice -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

The CPU job uses wICE `interactive`, two cores, 8 GB and twenty minutes. It checks datasets,
CV feasibility and synthetic contracts. Before the preflight, the CPU job now runs
`python -m src.jev.question_audit --phase experiment_0`: every active input hash and every
row/text-slot link is checked against the pinned raw datasets and current question templates.
Its compact receipt is on DATA under `3. experiment_0/manifests/question_audit.json`.
It performs no inference and does not modify question files. Linguistic/source review and
real-provider interpretation checks are separate from this exhaustive mechanical check.
Token counting uses `batch`, four cores, 13,600 MiB and
four hours. It counts every complete unique input for each pinned open-weight tokenizer,
including task/question/options and publisher formatting. Counts are committed incrementally
and resume after interruption; completed provider summaries survive another worker's failure.
Within a run, do not delete count databases to fix a dependency. The fingerprinted tokenizer modules are
kept stable so unchanged input/counter versions remain reusable. Jev's hosted tokenization
remains unverified. When changing question versions, rerun preparation and upload the 18
closed input files after stopping readers/writers, then rerun the CPU/token jobs. Historical
input sets remain inside those files. The explicit fresh-output reset discards old count
databases; otherwise unchanged counts can resume. Unchanged staged weights need no redownload.

Small summaries use DATA `3. experiment_0/manifests/before_extraction/` and operational
logs use DATA `3. experiment_0/logs/`;
large SQLite count databases use PROJECT `3. experiment_0/results/token_audit/`.
A passing CPU/token check is infrastructure evidence; GPU/provider compatibility is separate.

Also submit `sbatch scripts/slurm/experiment_0_storage.slurm`. This five-minute, two-node
CPU test checks that another node cannot acquire a held PROJECT writer lock, can acquire it
after release, sees committed SQLite rows, and can produce a valid snapshot/restore. It uses
only synthetic files under PROJECT Experiment 0 results; its compact evidence goes to DATA.
A pass verifies the tested mount's behavior, not storage quota or an independent backup.
Repeat it if moving to another filesystem/cluster. These checks do not need model weights.
Worker results use atomic JSON receipts under DATA
`3. experiment_0/manifests/before_extraction/storage_contracts/<run>/`; each must match the
job, lock phase and unique invocation ID. Launcher stdout/stderr are retained separately
under DATA `3. experiment_0/logs/storage_contracts/<run>/`. Console output is not parsed as
the result. After successful worker exit the coordinator allows up to 60 seconds for receipt
visibility across nodes and records the observed delay. It retries only missing-file reads,
never the worker operation; invalid receipts or other I/O errors still fail immediately.
If the test fails, inspect its top-level report and those logs before retrying.

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

For all five GPU tests, the setup helper prepares isolated provider
environments, reuses staged weights and performs an offline tokenizer smoke check. Run before
submitting jobs; it refuses active JEVPFN jobs. Packages/wheel caches/environments use PROJECT:

```bash
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
bash scripts/setup_decision_env.sh cloudflare
bash scripts/setup_decision_env.sh intern
bash scripts/setup_decision_env.sh liquid
bash scripts/setup_decision_env.sh strands
bash scripts/setup_decision_env.sh perplexity
```

CUDA 12.6 is the A100 candidate; verify driver/glibc compatibility in the allocation. The
[official paired wheels](https://pytorch.org/get-started/previous-versions/) establish package
availability, not JEVPFN compatibility. Clef's publisher tested H200 with Torch 2.11 and
Transformers 5.10.2; this candidate keeps Transformers 5.17 aligned with our token audit.
A failure requires diagnosis rather than silent version substitution.

| Open extractor | Runtime preparation |
|---|---|
| Liquid | Separate environment, compatible CUDA wheel and `.[dev,token-audit,liquid]`; staged custom decision implementation. |
| Strands | Separate Linux/CUDA environment and `.[dev,token-audit,strands]`; optional causal-conv1d acceleration requires a matching toolkit and a compute-node build. Direct `SystemOneEngine` inference; no HTTP server needed. |
| Perplexity | Separate environment; install the pinned checkpoint's `requirements.txt` and project dependencies. Keep its declared Torch version separate from downstream/Clef pins. Add staged `source/src` for `autojev`. |
| Clef | `.[dev,token-audit,clef]`; staged joint head/backbone; retain complete softmax precision. |
| Intern | `.[dev,token-audit,intern]`; retain the pinned publisher's calibration temperature and complete probabilities. |

Custom inference code is reviewed and hash-verified before loading. Complete dependency
versions enter response identity. All model and Hugging Face/Xet caches use PROJECT;
HF offline flags are set before runtime imports. Bytecode writes are disabled so importing
publisher code does not alter the fingerprinted checkpoint tree.
The helper uses Torch 2.12.1/CUDA 12.6 for four providers and the publisher's Torch
2.14.0/torchvision 0.29.0 CUDA 12.6 pair for Perplexity, then its fingerprinted requirements.
It records installed package versions beside each PROJECT environment. Strands' optional
causal-conv1d accelerator is not compiled on the login node. Its available kernel path must
pass the GPU contracts and a throughput pilot; a slow fallback is not permission for a full
campaign. If compilation is necessary, use a compute allocation with the matching CUDA
toolkit and repeat contracts after installation. A new package pin changes runtime identity.

E5 is shared by all six main predictors. Its requirements match the existing Clef runtime
(Torch/Transformers); no additional CUDA environment or package installation is needed there.
After resolving PROJECT file quota, stage its pinned weights from the connected CPU environment:

```bash
python -m src.data.text_embeddings --stage
sbatch --wait scripts/slurm/experiment_0_embeddings.slurm
```

The bounded test uses the Clef environment in a separate GPU process. It tests short and long
inputs, missing inputs, exact reuse, unit-norm vectors and restart without re-encoding.
It writes a synthetic fixture store to PROJECT Experiment 0 and its small report/log to DATA.
It does not create benchmark features or enable full extraction. After acceptance, enable
`config/2. feature_creation/text_embeddings.yaml` and run
`python -m src.data.text_embeddings --execute` in a compute allocation using that environment.
The permanent cache is PROJECT `data/text_embeddings/<dataset_id>.sqlite3`; logs use DATA
feature-creation logs. These embeddings are required before the comparison benchmark;
decision-model extraction can proceed independently. PCA is computed inside each CV training
fit, not in the permanent embedding cache.

## Bounded GPU contract tests and larger pilots

After exact counts for the chosen provider are complete, submit its prepared environment:

```bash
sbatch scripts/slurm/experiment_0_decision_contracts.slurm cloudflare
```

After Clef completes successfully, repeat for `intern`, `liquid` and `strands`.
Perplexity uses `sbatch --mem=112000M scripts/slurm/experiment_0_decision_contracts.slurm perplexity`.
Use `sbatch --wait` when chaining jobs so the next command is reached only after success.
The provider, optional environment prefix and optional project root are script arguments.
For example, append `/lustre1/project/stg_00211/JEVPFN/environments/intern /lustre1/project/stg_00211`
to select both paths explicitly. KU Leuven starts batch jobs with a clean environment; exporting
variables in the submitting shell is insufficient. CPU/token/storage scripts select the CPU
environment themselves and accept an optional project-root argument. The prepared job
environment is explicitly propagated to `srun` workers. See the guide’s Environment propagation
section and [Slurm argument/export semantics](https://slurm.schedmd.com/sbatch.html).

The job uses one full A100, four CPUs, 28,000 MiB host RAM and one hour in `gpu_a100_debug`,
with the configured CreditPFN account. Only one debug GPU job may be queued at a time:
wait for Clef to finish before submitting Intern. The pinned guide confirms that this is a
full GPU, unlike the interactive MIG slice. Verify account/partition access on the actual cluster.

The contract job tests binary, three-class and nine-level regression fixtures, a 30-class
vector, missing/duplicate/Unicode input, overflow rejection, cache reopen/reuse, original-row
alignment, and saving/reloading both feature representations. It then evaluates first/longest
unique inputs from each selected dataset. Seven additional known-answer probes check binary
polarity, permuted class ordering and score endpoints/zero. At most 74 forwards are planned, with a hard ceiling
of 100. Runtime token-count agreement, load/inference time, peak GPU memory and the complete
`settings_sha256` required for acceptance are recorded in the report.
A failed known-answer probe requires inspecting raw responses; the model itself may be wrong.

Synthetic artifacts use PROJECT `3. experiment_0/results/gpu_contracts/<run>/`; sampled real
responses use permanent PROJECT `data/jev_cache/<dataset_id>.sqlite3` for reuse. DATA holds
compact reports/logs under `3. experiment_0/{manifests,logs}/before_extraction/gpu_contracts/`
and the Slurm log directly under the phase's `logs/`. Failed model loading also leaves a report.

A passing contract report requires review and does not set full Experiment 0 acceptance.
Robustness, throughput and remaining provider checks are tracked in `debug.yaml`. Larger
profiling uses `self_hosted.yaml` with its explicit pilot opt-in and
`experiment_0_decision_pilot.slurm`: first/middle/last rows plus ten longest requests per
dataset, capped at 1,000. Full extraction has a separate switch and a reviewed acceptance
manifest matching runtime settings and token-audit identity. Submit the larger pilot as
`sbatch scripts/slurm/experiment_0_decision_pilot.slurm cloudflare` (or its selected provider).
Pilot/full summaries also expose `settings_sha256`.
`python -m src.cluster.accept_features --provider cloudflare` previews a matching evidence
set: current question audit, CPU tests, cross-node storage, complete exact token counts,
GPU contracts and the larger pilot. Repeat with each provider. It rejects missing, stale
or ambiguous successful reports. After reviewing those files, measured throughput, actual
quota and the independent backup arrangement, add
`--approve --reviewed-by Andreas --quota-verified --backup-verified --throughput-reviewed`.
This explicitly signs `3. experiment_0/manifests/before_extraction/accepted_<provider>.json`;
it never enables extraction. Only pass those flags after doing the corresponding review.
Set each path in `self_hosted.yaml`'s `accepted_before_extraction_manifests` map and enable
its separate full-extraction switch between stages. Tests alone never sign acceptance.
After acceptance, `scripts/slurm/feature_creation.slurm` runs full extraction on regular
`gpu_a100`; its 24-hour allocation is a resumable segment, not a runtime prediction. Create
`output_JEVPFN/2. feature_creation/logs` before submission, then use
`sbatch scripts/slurm/feature_creation.slurm cloudflare` (or its accepted provider). Provider jobs rotate datasets and
wait for busy files. `--continue-on-error` is an explicit diagnostic mode with durable failed
request records; incomplete datasets cannot be materialized or accepted. Repair the cause
and use `--retry-failed` to retry those inputs. No silent substitution by missing values.

The small-model contract/pilot/full-extraction defaults request 28,000 MiB. The predictor
smoke-test default uses 56,000 MiB. Perplexity 27B uses an explicit initial host-memory profile:
`sbatch --mem=112000M scripts/slurm/experiment_0_decision_contracts.slurm perplexity`.
Use the same override for its pilot/full extraction. These are initial profiles, not measured
minimums; revise after recording MaxRSS and load peaks. Regular A100 jobs must stay within
126,000 MiB per GPU. Memory requests can allocate/bill additional CPU cores: with 7,000 MiB
per core, 56,000/112,000 MiB imply eight/sixteen cores despite four requested compute threads.
The 13,600 MiB token profile fits four IceLake cores at 3,400 MiB/core and exceeds the observed
4.6 GiB token-audit peak. Do not assume all workloads fit the small-model profile.

Start serial inference on one wICE A100. Its 80 GiB can hold the Perplexity checkpoint's
approximately 49 GiB BF16 weights, but working memory must be profiled. Use a regular GPU
partition for longer jobs. Consider H100 for measured throughput benefit or Mindwell B200
(192 GiB, native GPFS and a Blackwell-compatible CUDA build) if measured memory justifies it.
GPU hours, quota and licences remain separate from vendor API token fees. Batch size one
is the current verified code path, not an efficiency claim. Before a multimillion-request
run, measure sustained throughput and total GPU-credit cost. Batched inference would require
provider-specific implementation, probability/token parity and resume tests; it is not enabled.
Hardware speedups and cheaper-GPU feasibility must be measured, not inferred from model size.

## Local Jev and Experiment 0B

Jev calls run locally after an approved small pilot. Install `.[jev]`, set `TYPESAFE_API_KEY`
privately, and use the pinned `jev-1.13.0` client. `src.jev.extraction` disables hidden retries,
uses durable spending reservations and stops on ambiguous billing outcomes. Config requires
verified account rates, a billing-token upper bound, context validation and explicit dataset/
campaign budgets. Published rates are references, not automatic account authorization.
The current docs publish no universal RPS/TPS allowance. Obtain your organization's values
from the console or TypeSafe support. The owner's pasted console showed USD 0 balance, so
funding/access must be settled before the paid pilot; the code does not top up an account.
Microsoft is outside the selected roster.

`python -m src.jev.extraction` previews readiness without credentials/network. Its `--execute`
flag runs the bounded local pilot; `--execute --full` needs a separate full-run switch and
accepted pilot evidence. Configure `live_extraction` in `feature_creation.yaml`, including the
reviewed context envelope, full-campaign dataset allocations, and the pilot request ceiling.
The envelope must justify a maximum complete-body byte size and its billing-token bound;
do not infer either from characters/4. Unknown bounds keep dispatch blocked. HTTP 429 has
bounded exponential backoff; 529/timeouts remain review stops because retryable does not
establish that an earlier attempt was unbilled. Raw responses received by the dispatcher
survive a subsequent feature-decoder failure.

Alternatively, keep the tracked defaults disabled and run
`python -m src.jev.setup_pilot --configure` locally. It prompts for verified price, RPS/TPS,
context/billing bounds and a conservative EUR/USD conversion, then writes the Git-ignored
`config/2. feature_creation/jev.local.yaml`. It enforces the approved EUR 5 pilot, includes
all 18 dataset allocations and never enables full extraction. Unknown provider bounds must
be obtained before this step; a guessed byte/token ratio is not acceptable evidence.
Use `python -m src.jev.hosted --config feature_creation/jev.local --execute` for that pilot.
`python -m src.jev.setup_pilot --check-account` sends only the authenticated model-list GET;
it verifies neither paid inference, balance, rate limits nor access to the exact version.
Keep the key in a terminal environment variable and clear it when finished.

After a decoder repair, dispatch revalidates a saved `invalid_response` without another
API call or charge. For exhausted verified-unbilled rejections or uncertain billing, stop
writers and inspect `python -m src.jev.recovery DATASET_ID`. This prints request keys,
accounting and the exact job-state hash, without text or credentials. Prepare a reviewed
JSON file with `stage: jev_request_recovery`, `accepted: true`, `reviewed_by`, checksummed
`evidence`, `dataset_id`, `request_key`, `expected_job_sha256`, and one action:

- `renew_rejected_attempts`: reopen only a verified-unbilled rejection sequence.
- `reconcile_billing`: record the provider-confirmed `charged_usd` for the unresolved
  attempt and `input_tokens` (integer or null), plus explicit
  `retry_if_response_missing: true/false`. Charges are added, never replaced. A stored
  response becomes available for revalidation; without one, false leaves it stopped.
- `retry_resolved_without_response`: explicitly approve a later retry after billing has
  already been settled and no response was recovered.

Preview with `python -m src.jev.recovery DATASET_ID --review REVIEW.json`; apply with
`--apply --writers-stopped`. Changed evidence/job state is rejected. A repeated approval
is idempotent; before/after states and evidence remain in the permanent dataset store and
survive validated transfers. This command never calls Jev. Never erase reservations merely
because a timeout probably went unbilled; first obtain billing evidence from the provider.

If the pilot requires a different allocation, price or token bound, set
`live_extraction.spending_policy_revision_manifest` to an owner-reviewed JSON file. Use
`accepted: true`, `reviewed_by`, `stage: jev_spending_policy` and checksummed `evidence` as
described below. Its `policies` map has one entry per revised dataset, containing
`previous_policy_sha256` from that dataset's run report and `policy` with the exact new
`budget`, `price` and `token_bound`. Budget is cumulative USD including the pilot; price is
USD per million input tokens. Revisions cannot erase charges or unresolved reservations,
and earlier requests retain their original price/policy identity. Unchanged restarts need
no new revision. A revised context envelope also needs new matching pilot acceptance before
full extraction; spending review alone does not approve context or provider compatibility.

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

Enable only the bounded `predictor_execution_enabled` switch in `experiment_0/debug.yaml`
for `python -m src.cluster.predictor_contracts --model MODEL --execute`. This runs actual
binary/three-class/30-class/regression fixtures, five-fold CV and no-refit restart checks.
`sbatch scripts/slurm/experiment_0_predictor.slurm tabpfn_3_5 /lustre1/project/stg_00211/JEVPFN/environments/tabpfn_3_5`
supplies a regular A100 allocation once that selected predictor environment exists. Run CPU
CatBoost/linear checks in a CPU allocation with the same module command.
For full-size feasibility, separately enable `full_fold_profile_enabled` and run
`python -m src.evaluation.campaign --dataset ID --model MODEL --feature-set ARM --provider PROVIDER --profile-fold --execute`
inside the chosen allocation (omit provider for the non-text baseline). It executes only
outer fold 0, including binary inner validation, with all its training rows and saves to
Experiment 0. Full benchmark execution still requires its own switches and reviewed 0B evidence.

Jev/full-benchmark acceptance files are owner-reviewed JSON: `accepted: true`, `reviewed_by`,
`stage` (`jev_pilot` or `after_extraction`) and a nonempty `evidence` list of `{path, sha256}`
test reports. Jev acceptance also copies `settings_sha256`, `input_sets` and
`context_envelope_sha256` from the pilot report. Open-weight acceptance uses stage
`before_extraction` with `settings_sha256` and `token_audit_id`, plus the same reviewer/evidence
fields. Set its file per provider in `accepted_before_extraction_manifests`. None is generated
as accepted by a test.
Inspect the actual evidence, quota/backup and runtime coverage before signing acceptance.

CV folds have completion manifests with record/prediction checksums. Restarts skip only
complete matching folds; interrupted attempts remain evidence. Experiment 3's statistics,
permutation and retained-view arms use `campaign --include-controls` and the explicit
`--control`/`--view` filters. `python -m src.evaluation.results --save` writes complete-arm
per-dataset metrics and paired differences to PROJECT Experiment 4, with its compact index on
DATA. Negative differences are improvements for loss metrics, not for higher-is-better metrics.
E5 is part of Experiment 1: `--feature-set non_text_plus_frozen_e5` selects the shared baseline;
`non_text_plus_frozen_e5_and_joint`, `non_text_plus_frozen_e5_and_per_column` and
`non_text_plus_frozen_e5_and_both` also require `--provider`. No `--control` is used for E5.
Analysis matches non-text, E5-only and the same extractor without E5 references, plus a
separate native-text reference table and dataset-level summaries. `--infer` additionally
requires an approved `experiment_4/analysis` plan; it does not select headline metrics itself.

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
