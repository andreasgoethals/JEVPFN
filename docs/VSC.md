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

## One clean validation campaign

Finish runtime setup and diagnose failures before deleting output. The latest container
failure and its repair are recorded in AGENTS_MEMORY. Do not reset after every fix.
After runtime tests succeed, the owner publishes the final source; local and VSC HEADs
must match. Stop notebook/manual writers and wait for all JEVPFN jobs to finish.

In the activated VSC CPU environment:

```bash
"${JEVPFN_PYTHON:?}" -m src.utils.output_cleanup --tier both --include-archives
"$JEVPFN_PYTHON" -m src.utils.output_cleanup --tier both --include-archives --apply --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
```

This permanently clears generated output on both tiers, including old token counts,
logs and archives. It preserves raw/prepared data, permanent response/accounting/feature
stores, checkpoints, environments and containers. Cleanup refuses active JEVPFN jobs,
links/junctions/mounts and unsafe roots. Manual writers must also be stopped.

For local cleanup and preparation, use this fail-fast PowerShell block:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    Set-Location 'C:\Users\U0152019\PhD Documents\Projects\5. JEVPFN\JEVPFN'
    function Invoke-Jev {
        & .\.venv\Scripts\python.exe @args
        if ($LASTEXITCODE -ne 0) { throw 'Python failed; remaining steps stopped.' }
    }
    Invoke-Jev -m src.utils.output_cleanup --include-archives
    Invoke-Jev -m src.utils.output_cleanup --include-archives --apply --writers-stopped
    Invoke-Jev -m src.data.prepare
    Invoke-Jev -m src.jev.preparation --rebuild
    Invoke-Jev -m src.jev.question_audit
    Invoke-Jev -m src.utils.run_notebooks
    Invoke-Jev -m src.cluster.freeze --record
    Invoke-Jev -m src.cluster.input_transfer pack
}
```

The rebuild replaces only the current input set transactionally, retaining its identity
when content is unchanged; interrupted rebuilds restore the previous complete set.
No paid response is deleted. Raw preparation verifies/reuses the pinned local downloads.
The subsequent VSC CPU job checks those same raw hashes. Notebook regeneration performs
no model inference. The new token audits start from zero because their output was cleared.

Selective cleanup is also available, for example:

```bash
python -m src.utils.output_cleanup --tier data --phase experiment_0 --kind logs --older-than-days 7
python -m src.utils.output_cleanup --tier project --kind results --match '*storage_contracts*'
```

These only preview; append `--apply --writers-stopped` after checking the listed files.
Phases/categories are repeatable. Deleting selected PDFs prunes their caption entries;
aggregate reports are rebuilt by `run_notebooks --summaries-only` or the next notebook run.
On Windows, read-only attributes and brief sharing locks are handled within validated
output trees. Persistent errors stop deletion; close the writer and repeat before continuing.

A frozen receipt covers Git HEAD, normalized text source, base configuration and local
runtime pins. Compare local/VSC source with `--expected-file` after uploading the receipt.
Notebook outputs are excluded; notebook source is included. Shell files must have LF.
Do not modify the frozen stage. After reviewing it, use
`python -m src.cluster.freeze --finish-stage --reviewed-by Andreas --writers-stopped`
on both machines before explicit next-stage configuration changes. Record a new freeze
for that stage without deleting accepted test results.

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
scp '.\output_JEVPFN\transfers\decision_inputs_v3.zip' '.\output_JEVPFN\transfers\decision_inputs_v3.zip.sha256' '.\output_JEVPFN\3. experiment_0\manifests\frozen_revision.json' 'vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/uploads/'
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; do not continue.' }
```

This uses the same working SSH authentication as the owner's CreditPFN SCP commands; no
assumed private-key filename.

After SCP finishes, from the activated CPU environment in `$VSC_DATA/JEVPFN`:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.input_transfer install /lustre1/project/stg_00211/JEVPFN/uploads/decision_inputs_v3.zip --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.freeze --record --expected-file /lustre1/project/stg_00211/JEVPFN/uploads/frozen_revision.json
```

The installer checks the bundle/hash, every database, active question bank, row-link counts
and file hashes before replacement. Existing question files are backed up under PROJECT
`output_JEVPFN/transfers/input_backups/`; repeat uploads reuse identical files. Byte-exact
copies preserve input fingerprints across local/VSC token reports. It refuses active/recovery
SQLite sidecars and queued/running JEVPFN jobs; also stop manual writers yourself.

## Experiment 0A_local and 0A_vsc

Before the final freeze, stage tokenizers/checkpoints/containers/E5 and configure the local
Jev dataset pilot as described below. Then clean, regenerate, transfer and freeze once.

Local PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local --execute --dataset-pilot
if ($LASTEXITCODE -ne 0) { throw '0A_local stopped; inspect its report.' }
```

The first command previews. Execution checks dependencies, every prepared question/row
link, local synthetic contracts (including notebook tests), cached/live Jev task contracts
and the configured paid dataset pilot. It refuses missing/unverified private pilot settings
before starting. Omitting `--dataset-pilot` is useful for infrastructure checks but leaves
0A incomplete. Saved smoke responses are reused, not billed again; dataset pilots provide
additional real-input evidence. Local receipts/logs use `3. experiment_0/{manifests,logs}/0A_local/`.

VSC login terminal, after input installation and matching frozen receipts:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.experiment_zero --site vsc
"$JEVPFN_PYTHON" -m src.cluster.experiment_zero --site vsc --execute --watch
```

The runner submits **one job at a time**: CPU → two-node storage → exact tokens → E5 →
contracts and larger pilot for Clef, Intern, Liquid, Strands and Perplexity. The debug GPU
partition permits only one queued job, so no batch of pending GPU jobs is submitted.
`--watch` polls accounting once per minute, performs no inference on the login node, and
submits the next job only after COMPLETED/0:0. A failed/uncertain submission stops for
review. If OnDemand disconnects, the current Slurm job continues: reconnect, activate the
CPU environment and repeat the same command. It resumes from recorded IDs, not a new run.
Without `--watch`, one invocation checks/advances one step. The receipt records submitted,
waiting, completed or failed; submission alone is never a pass. Receipts/logs are under
DATA `3. experiment_0/{manifests,logs}/0A_vsc/`. Each job also retains its normal Slurm log.

CPU tests audit every input and row link, dataset hashes, splits and synthetic contracts;
only local-notebook execution tests are omitted on VSC. Token jobs count each complete
unique request with every actual open-weight tokenizer, including options/provider format.
Jev tokenization is not inferred from another model's counts. The two-node test validates
PROJECT writer exclusion, release/reacquisition, commit visibility and snapshot/restore.
Atomic DATA receipts tolerate bounded cross-node visibility delay without repeating workers.
These checks do not verify quota or independent backups.

Do not delete logs during this campaign or import older output bundles into it. A later
transfer imports both PROJECT and DATA evidence for local notebook inspection. Finish 0A
only after reviewing the reports, measured throughput, quota and backup arrangements.

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

Use compact runtimes below on this file-limited allocation. The alternative
`bash scripts/setup_decision_env.sh PROVIDER` installs an unpacked environment on PROJECT;
do not create five such environments when inode quota is nearly exhausted. Keep each
provider isolated. Perplexity uses its pinned publisher requirements; Strands stages its
adapter plus base model. The exact packages and CUDA candidates are in the setup scripts
and pyproject, with resolved versions recorded in each runtime manifest.

GPU compatibility must be demonstrated in the allocation. Custom inference code is
hash-verified; compute runs set offline flags and disable bytecode writes before imports.
All model/Hugging Face/Xet caches use PROJECT. Optional Strands kernel acceleration requires
a matching CUDA toolkit and a compute-node build; a slow fallback must be profiled before
acceptance. Changing packages changes runtime identity and requires new contract evidence.

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

## Compact decision runtimes

The selected site module is `hpc-container-wrapper/0.4.2`. Use the login shell to inspect it;
actual packing/building runs through the CPU allocation below. `wrap` packs an existing
working environment; `new` installs a fresh runtime entirely in scratch before packing it.
Images and wrappers end in PROJECT `JEVPFN/containers/<provider>/`. Weights stay outside the
image in PROJECT checkpoints. The build copies the site's wrapper configuration and redirects
its temporary/work directories and package caches to scratch. No source environment is deleted.
For `wrap`, the private scratch copy of wrapper 0.4.2 replaces only the verified site
copy template and frontend alias. It copies contents/links/timestamps without importing
native ACLs/ownership. The site installation and original environment remain untouched;
unknown templates fail closed. See AGENTS_MEMORY for observed failures. Build work/logs
remain available for diagnosis; do not reset output while setup is failing.

The build checks `apptainer`, both wrapper entry points and `mksquashfs` before hashing
checkpoints or copying an environment. If the packer is absent from PATH, it reads
`LIBEXECDIR` from `apptainer buildcfg` and checks the bundled `apptainer/bin/mksquashfs`.
This follows [Apptainer's documented bundle layout](https://apptainer.org/docs/user/1.4/build_a_container.html).
It does not guess a site module, install software or edit shared configuration. A missing
or broken packer stops immediately. The same read-only check can run on the login node:

```bash
bash scripts/check_container_tools.sh
```

After publishing/synchronizing the source and passing that check, start with Clef:

```bash
sbatch scripts/slurm/experiment_0_container_build.slurm cloudflare wrap
# After its job ID is COMPLETED with exit 0:0 and its log confirms sealing:
# First validate saved token evidence in the activated CPU environment (see below).
sbatch scripts/slurm/experiment_0_decision_contracts.slurm cloudflare
```

A sealed receipt hashes image bytes and wrapper/link contents, records package versions and
checks that a source-only sentinel is invisible inside the packed image. The launcher selects
the sealed image automatically, verifies it before execution and exposes GPUs with Apptainer.
Clef/Intern package versions must equal their originals. New runtimes have their own recorded
identity. The GPU report includes the image identity, so native and container evidence cannot
be confused. Clef's build/seal and CUDA tensor execution have now passed; model inference
and other providers' containers still need successful contract evidence.

Only after matching GPU evidence passes, preview and retire the redundant native source:

```bash
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.cluster.containers retire cloudflare
"$JEVPFN_PYTHON" -m src.cluster.containers retire cloudflare --apply --writers-stopped
```

This requires idle JEVPFN jobs, checksummed matching GPU settings and passed semantic probes.
It removes only PROJECT `environments/cloudflare`, retaining an empty image mount point.
It preserves the container, every checkpoint, inputs/features, outputs and other projects.
Retiring Clef frees room before packing Intern. Repeat `wrap`, GPU test and guarded retirement
for `intern`. For `liquid`, `strands`, `perplexity`, use build mode `new`; the partial failed
Liquid installation is not treated as a working source. Test each new image before proceeding.
Perplexity GPU contracts/pilots use `--mem=112000M`. Builds prefer shared Lustre scratch on
wICE, so failed-build work survives the allocation. An empty leftover destination can be
retried; nonempty or linked destinations are refused for manual review. Keep the Slurm log
until the failure is understood; do not reset output while runtime setup is still failing.
The container-build script resolves `default_config/config.yaml` from the installed wrapper;
KU Leuven does not necessarily export `CW_GLOBAL_YAML` from the module itself.

## Bounded GPU contract tests and larger pilots

After exact counts for the chosen provider are complete, submit its prepared environment:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.gpu_contracts --provider cloudflare --check-inputs
sbatch scripts/slurm/experiment_0_decision_contracts.slurm cloudflare
```

Run the first command in the activated CPU environment on the login host. It reads the
saved token report, validates its tokenizer identity, input hashes, dataset coverage and
PROJECT count database; it loads no weights and executes no inference. If it fails, do not
submit the GPU job. Missing DATA `before_extraction/token_audit.json` or PROJECT count files
require `sbatch scripts/slurm/experiment_0_tokens.slurm` first, followed by COMPLETED 0:0
and the read-only check again. A matching surviving count database is reused; otherwise
the CPU job recounts. Do not rebuild a sealed container, regenerate unchanged questions or
disable context checks to repair a missing report. A deliberate full-output reset removes
this evidence, so exact token counting must precede GPU contracts in the subsequent campaign.

After Clef completes successfully, repeat for `intern`, `liquid` and `strands`.
Perplexity uses `sbatch --mem=112000M scripts/slurm/experiment_0_decision_contracts.slurm perplexity`.
The 0A coordinator handles sequential submission. For individual setup jobs, submit without
`--wait` and inspect the recorded ID with `sacct` before the next command; a disconnected
login terminal is not proof that the submitted job was cancelled.
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

Small-model jobs start at 28,000 MiB host RAM; Perplexity uses 112,000 MiB. These are
initial profiles, not measured minima. Regular A100 jobs must stay within 126,000 MiB per
GPU; memory requests can bill extra CPU cores. Start on one wICE A100 80 GiB. Consider
H100 or Mindwell B200/native GPFS only if measured throughput/memory justifies migration.
Batch size one is the current implementation; a million-request campaign needs measured
GPU-credit/runtime estimates. Batching would require provider-specific parity/recovery tests.

## Local Jev and Experiment 0B

First verify the pinned account/task interface locally with:

```powershell
.\.venv\Scripts\python.exe -m src.jev.smoke --execute
```

The owner's key is stored in Git-ignored `secrets/typesafe.local.json` on the laptop and
loaded automatically by account checks, smoke tests and dataset extraction. Do not upload
this file. An explicit `TYPESAFE_API_KEY` environment variable takes precedence; with neither
source, the CLI asks through a hidden prompt. Local format errors never display the key.
A constructor failure with zero recorded attempts can resume with corrected input; an
attempted request without a saved response stops for review. Only a specifically recorded
authentication HTTP 401 can be retried once using `smoke --execute --retry-auth-rejected`
after correcting credentials. Its old diagnostic and conservative billing reservation remain
in the receipt. This option cannot bypass an ambiguous timeout or other unknown outcome.

It sends only three fixed synthetic probes sequentially and
keeps complete responses and the spending ledger in permanent
`data/jev_cache/_control/jev_smoke/`, with rebuildable copies in Experiment 0 output.
Legacy records are migrated before cleanup; clearing logs never resets paid-call history. It checks exact Jev 1.13 identity, binary,
multiclass, nine-level score mapping and usage. The reviewed policy in
`config/3. experiment_0/jev_smoke.yaml` reserves at most USD 0.008064 for three attempts using
the published 64k per-request context and USD 0.042/M price, with no long-context header.
SDK/application retries are disabled. Existing verified responses are reused; a missing
response after an attempt stops for billing review. HTTP errors stop, including rate limits.
This slow interface check does not claim to know account RPS/TPS or enable dataset extraction.
It passed live again on 19 October after credential setup; its three inputs consumed 1,051
reported tokens at computed input cost USD 0.000044142. Repeating the successful command
reuses those saved responses without new requests. The dataset pilot remains separate.

The console's user menu → Settings → Organization displays the organization ID. Billing
shows credits and auto-recharge; neither that page nor Home/Organization/Usage exposed
RPS/TPS during the observed check. Request those values from the existing TypeSafe contact
or `support@typesafe.ai`; do not substitute proposed traffic settings for account limits.
Keep the organization ID and credentials out of public configuration/docs.

Jev calls run locally after an approved small pilot. Install `.[jev]`, use the local credential
file or a private environment override, and use the pinned `jev-1.13.0` client. `src.jev.extraction` disables hidden retries,
uses durable spending reservations and stops on ambiguous billing outcomes. Config requires
verified account rates, a billing-token upper bound, context validation and explicit dataset/
campaign budgets. Published rates are references, not automatic account authorization.
The current docs publish no universal RPS/TPS allowance. Obtain your organization's values
from the console or TypeSafe support. The owner confirmed USD 5 funding on 16 October;
check the remaining balance before the dataset pilot. The code does not top up an account.
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
The local private file remains outside Git and output cleanup. Clear any temporary environment
override when finished so it cannot unexpectedly override that file.

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
