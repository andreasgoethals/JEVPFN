# VSC operations

Use this guide for execution; use README for the experiment design and
AGENTS_MEMORY for dated observations. Historical bounded B200 checks passed for all eleven
open-weight decision runtimes, three encoders and TabSTAR. Final acceptance still requires
one complete frozen campaign; historical setup passes are not current acceptance.
The current frozen campaign is already running. Preserve its queued inputs, source and
output; do not repeat a reset or upload while it has consumers. See AGENTS_MEMORY for
the exact receipt/job IDs. New local Jev validation is a separate revision.

## Shared manual access

The canonical template and this project's copy matched byte-for-byte on 30 October 2026.
The owner renews their shared Windows-agent certificate manually:

```powershell
step ssh login --context VSC
```

Approve KU Leuven Authenticator when prompted. Agents never initiate authentication,
install a per-project agent, copy private keys or schedule renewal reminders. Ask for
this command only if access actually requires renewal. Use native Windows OpenSSH with
host-key checking and `-F none` to avoid optional automatic-login hooks.

Follow TEMPLATE.md's bounded LF-script-on-stdin connection, without a trailing remote
command. On 29 October the remote stdout marker and deliberate exit 7 arrived, but the
unredirected stderr marker did not. Repeating the probe with `exec 2>&1` at the start of
the remote script preserved both markers and exit 7. Use that explicit combined stream
for this gateway, check completion/status markers, and do not treat empty exit-0 output
as success. This is a verified gateway workaround, not an authentication failure.

Read relevant files, logs, quotas and scheduler state directly after that probe. Access
alone authorizes no submissions, cancellations, migration or deletion. Do not pull into
a checkout consumed by active or queued jobs; keep recovery state until it is published.

One temporary 30-minute local monitor follows the current campaign. Its ID is
`jevpfn-0a-progress`, recorded with owner/source/jobs/completion criteria in compact local
output. Keep the host/app available. It stays quiet without meaningful change and never
authenticates, submits or cleans. Pause on expired credentials; remove it once tracked work
is terminal and evidence assessed. Do not create a duplicate monitor.

## Storage and hardware

| Role | Location |
|---|---|
| Actual Git checkout including `.git`, specs, small logs/manifests | `$VSC_DATA/JEVPFN` |
| CPU controller and its immutable dependency image | `PROJECT/JEVPFN/containers/control-v1` |
| Authoritative raw data, prepared questions, weights, containers, features, predictions and large output | `/lustre1/project/stg_00211/JEVPFN` |
| Active B200 working set, isolated by runtime | `$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN` |
| Local notebooks and downloaded/merged results | Local repository |

Set `JEVPFN_PROJECT_ROOT` (or shared `VSC_PROJECT_ROOT`) to the allocated shared root,
currently `/lustre1/project/stg_00211`; the resolver adds `JEVPFN`. An explicit legacy
`JEVPFN_STAGING_ROOT` overrides it for a native working set. Launchers pass the resolved root in arguments and `--export=ALL,JEVPFN_DURABLE_ROOT=...`,
including release jobs. Direct Slurm calls must likewise pass their root argument or explicit
export option; arbitrary login-shell exports alone are not sufficient. No large-output fallback
to DATA is allowed. Moving to another allocation is a reviewed migration, not an alias edit.

Cluster dependencies use immutable containers on PROJECT, with temporary build files
on scratch. The CPU image was recreated from recorded package versions and activated
after job 62469013 passed CPU/data, nested-launch and two-node storage tests. The old
DATA prefix was retired after native B200 test 11753506 passed; its guarded retirement
preserved shared Miniconda and all other environments. GPFS test 11753505 also passed
cross-node exclusion, release/reacquisition and SQLite snapshot/restore.
Model containers follow the same policy. Retain old images only while a recorded run needs them.

Mindwell B200 is the default GPU route. It uses native GPFS scratch; **no GPFS
project allocation is required**. wICE CPU jobs stage/download/build/count on
Lustre, then copy and verify the required working set before B200 starts. Heavy
model/data reads and writes on B200 use GPFS. DATA retains compact logs/manifests.
The H100 profile remains for explicit compatibility use; no default job uses it.
A100, including debug and interactive slices, is excluded.

Committed decision/embedding/evaluation SQLite data are periodically published to
Lustre every 600 seconds, and remaining output is published when the child command
exits, including ordinary failures. Database publication merges compatible records;
it does not overwrite a complete durable cache with a partial copy. Conflicts stop
publication. Repeated mirroring retains each distinct accounting-history version once;
changed charges or job states remain traceable. Before a predictor runs, staging restores
durable evaluation SQLite databases (predictions and completed folds) for all existing
experiment phases. Existing scratch results are unioned transactionally, not replaced;
already completed folds are reused. Decision-runtime staging omits these predictor stores.
Scratch is not a backup: VSC removes unused GPFS scratch files after
30 days, and a killed node can lose work since the last publication. Preserve an
independent downloaded copy of irreplaceable features. Disk free space is not quota;
check `myquota` for both bytes and file counts before staging all large checkpoints.

Containers are compact images rather than tens of thousands of unpacked files.
B200 gets separate `containers/blackwell-cu130/<runtime>/` images, leaving old cu126
images intact. Builds preserve publisher package pins, verify that the exact cu130
wheels exist, and validate the sealed runtime. B200 startup checks CUDA 13.0,
native `sm_100`, BF16 matrix operations and the actual B200 device before model
loading. Real provider/encoder contracts are still needed; these basic checks do
not establish model correctness. Do not substitute newer packages to bypass a failed pin.

The exact selected Torch/torchvision releases are present in the official
[cu130 Torch index](https://download.pytorch.org/whl/cu130/torch/) and
[cu130 torchvision index](https://download.pytorch.org/whl/cu130/torchvision/),
checked 24 October 2026. The cu128 index does not contain these exact releases;
therefore the B200 image uses CUDA 13.0 without changing publisher package pins.
[NVIDIA's compatibility table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver)
requires driver branch 580 or newer for CUDA 13.x. The live B200 check verifies the
reported driver and then executes real kernels; wheel availability is not proof
that the current VSC driver is compatible.

The current B200 setup uses four CPUs/28,000 MiB for small providers and fourteen
CPUs/112,000 MiB for the 27B providers. These fit the reviewed B200 resource limits
and are starting profiles, not measured optima. Independent providers run concurrently;
CPU preparation runs before GPU allocations. Account limits and scheduler priority
still determine start times. More credits do not guarantee an earlier start.

Infrastructure evidence: pinned VSC Library
`3bcdf55e576d10161d6189b40ac39380bbba9eca`, upstream
`1a0cf04a5aab107021d98c563351f912a3caf062`, reviewed 23 October 2026.
See its `guides/SLURM.md`, `STORAGE.md`, `SOFTWARE.md` and `CLUSTERS.md`.
Storage/Slurm guidance for this merge-recovery update was rechecked on 27 October
at VSC Library `1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87`, upstream
`7c04105490d610fee13125c8c70fa4bddd854a88`. The earlier pin records the hardware review.

## CPU control container

CPU preflight uses wICE batch, three cores and its verified 3400 MiB/core default.
The former interactive route was blocked by the user's shared interactive CPU QoS;
no other project's allocation was cancelled to make room.

`experiment_0/control_runtime.yaml` records the exact Python/package versions read
from the former DATA control environment. `control_build.slurm` recreates these
inside `PROJECT/JEVPFN/containers/control-v1`, using only temporary Lustre scratch
for unpacked installation/cache files. It seals image/wrapper hashes and verifies
package versions, checkout visibility and dependency compatibility.

Before cutover, `control_test.slurm` selects that image explicitly. It tests Slurm
accounting and scheduling dry-run commands, launching a model image from the CPU
controller, two-node locking/snapshot behavior, data checks and the synthetic CPU
suite. Host Slurm/Apptainer clients, plugins/config and Munge library are bound into
the CPU image; installed Python dependencies stay in its SquashFS. Nested launches
clear the parent image's Conda stack and automatic bind list.

Newly built, original model wrappers are probed through an overlapping host Slurm
step in the existing allocation; their immutable wrappers are not changed to work
inside the CPU image. The integration test covers both this build-probe route and
the relocated, monitored runtime route. Rebuildable compiler/plot caches use a
marked per-job native scratch leaf (node-local when available), removed by the
job's exit hook. A cache-cleanup failure cannot hide the original job failure;
weights, response queues and recovery workspaces are outside this disposable leaf.

`python -m src.cluster.control_runtime activate --test-job-id JOBID` requires all
three successful reports from the same job and an idle checkout. Its persistent
cutover receipt embeds the reports, so a later deliberate output reset preserves
the image's installation evidence. Normal activation then verifies the selected
image and fails on corruption. B200 staging includes a checksum-verified native
copy of this small CPU image alongside its model image, within the same quota budget.
For future migrations, retain the old environment until cutover validation is complete.
The current migration and retirement have already passed. After
native staging, `control_gpu_test.slurm` verifies both the native CPU image and a
monitored model-container CUDA kernel on B200. Only a matching pass permits
`python -m src.cluster.control_runtime retire-legacy --test-job-id JOBID`; this
removes the exact DATA/conda_envs/JEVPFN prefix and records evidence outside output.

Question ZIP installation has `input_install.slurm`; it validates on a CPU node,
ignores only its own job in the quiescence check and refuses all other JEVPFN consumers.

## Bound and release GPFS working copies

On 29 October, JEVPFN retained 416 GiB across twelve working directories, alongside
84 GiB from CreditICL. This exhausted the owner's 500 GiB scratch allocation. The
quota covers all projects; neither total filesystem free space nor spare Lustre quota
provides additional personal GPFS capacity. Never delete another project's files.

Setup and frozen 0A permit at most two resident runtime working sets. A stage takes
a slot when submitted and holds it until all consumers are terminal and its release
job succeeds. Release still runs after a failed consumer, publishes committed results
first, then removes only copied raw/question inputs, checkpoints, container images and
CPU token evidence. Fitted PCA artifacts are retained. It preserves scratch response queues,
embeddings, predictions and diagnostic output as well as every durable Lustre asset
and DATA log/manifest. A publication conflict preserves the input copies and blocks
further staging. Cloudflare's release waits for all three encoder tests as well.

The configurable application budget is 200 GiB, with a 500 GiB personal allocation
and 20 GiB headroom in experiment_0/scheduling.yaml. Staging also reserves temporary
copy space and checks the other projects' current usage. These are conservative
pre-copy checks, not a scheduler quota reservation; other jobs can grow afterwards.
Predictor workspaces no longer copy unrelated decision token databases. Mutable
result growth remains part of the budget and is never silently deleted to fit it.

After all JEVPFN jobs and watchers stop, pull the reviewed code and run these CPU
jobs from `$VSC_DATA/JEVPFN`. The first previews exact paths and removable bytes;
the second publishes results and applies that scoped release:

```bash
sbatch --wait scripts/slurm/experiment_0_scratch_release.slurm all
sbatch --wait scripts/slurm/experiment_0_scratch_release.slurm all --apply
myquota
```

Both jobs write `output_JEVPFN/3. experiment_0/logs/jevpfn-scratch-release_<jobid>.log`.
Use a runtime name instead of `all` for a single working set. The helper checks
workspace identity, authoritative source files, queued/running consumers and
an exclusive workspace lease. Unknown scheduler state refuses deletion. It never
cleans the scratch root or other projects. Transformers-generated .py/.pyc import-cache
files under checkpoints/huggingface/modules need no durable counterpart; checkpoint
source files and weights outside that narrow exception remain protected. Do not submit new consumers during manual
release. Existing logs are retained; this is storage recovery, not a campaign reset.

A released runtime must be staged again before any GPU work. Setup/frozen 0A handle
that automatically. For manual predictor or production batches, run the runtime's
scratch stage first and its release only after all that runtime's consumers finish;
keep no more than two such runtime batches resident. Durable checkpoints and sealed
containers do not need downloading or rebuilding. Full extraction stays disabled
until acceptance; this storage change does not grant acceptance.

## Setup before a future frozen campaign

The current frozen campaign is already running; keep its source and prepared inputs
unchanged. Once idle, an authorized unfrozen setup uses
`bash scripts/test_all.sh --execute --watch`. The graph stages pinned models, builds
compact PROJECT images, partitions exact-token audits and runs independent CPU/GPU
contracts and pilots concurrently. Resume saved job IDs; never blindly restart an
uncertain scheduler submission. Current jobs and migration evidence are in AGENTS_MEMORY.

## Local Jev setup

The Git-ignored `secrets/typesafe.local.json` is loaded automatically. Never upload
it or put the key in commands/reports. An explicit `TYPESAFE_API_KEY` override takes
precedence; remove stale overrides. In local PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.jev.setup_pilot --check-account
.\.venv\Scripts\python.exe -m src.jev.smoke --execute
.\.venv\Scripts\python.exe -m src.jev.smoke --robustness --execute
```

The account check only sends an authenticated model-list GET. Smoke tests check
pinned `jev-1.14.0` binary, multiclass and nine-level outputs; robustness uses a
separate bounded diagnostic. Existing paid responses/ledgers are reused, including
after output cleanup. `--revalidate-only` reprocesses saved responses without API
calls; never erase uncertain attempts to obtain a retry.

The owner confirmed 80 RPS and 100,000 input TPS on 21 October; recheck the account's
[rate-limits page](https://console.typesafe.ai/rate-limits) before full extraction.
Synthetic probes run at least five seconds apart. The current reviewed dataset pilot uses
one worker at one RPS, within the owner-confirmed limits; setup_pilot initially defaults to
0.2 RPS until an explicit rate review. The documented
32k state/question and 64k complete-request limits are provider-enforced; a 64k
billing reservation is a worst-case spending bound, not a measured token count.
The HTTP body ceiling does not prove token fit. Overflow stops without truncation.

If no private pilot config exists, run `python -m src.jev.setup_pilot --configure`.
It requests the recorded USD 0.042/M price, a conservative EUR/USD conversion and
account evidence, then enforces the approved EUR 5 pilot budget. It refuses to
silently replace an existing config. Review the existing
`config/2. feature_creation/jev.local.yaml` instead of recreating it.
The pilot command is:

```powershell
.\.venv\Scripts\python.exe -m src.jev.hosted --config feature_creation/jev.local --execute
```

Durable reservations, bounded 429 handling and unresolved-timeout stops prevent
uncontrolled repeat billing. A decoder repair revalidates saved responses.
`python -m src.jev.recovery DATASET_ID` inspects stopped requests without calls;
reviewed `--review FILE --apply --writers-stopped` transitions preserve charges
and history. Spending-policy revisions must preserve cumulative charges/reservations.
Neither a larger balance nor successful smoke tests enables full extraction.

After Jev 1.14 is revalidated, the additional first-100-source-rows validation is:

```powershell
.\.venv\Scripts\python.exe -m src.jev.prefix_pilot
.\.venv\Scripts\python.exe -m src.jev.prefix_pilot --execute
if ($LASTEXITCODE -ne 0) { throw 'Prefix pilot stopped; inspect its receipt before resuming.' }
```

It covers 1,800 rows and 7,643 distinct joint/per-column requests at one RPS, with a USD 1
cumulative dataset-charge/reservation cap and the existing private budget caps. The rate
floor is about 127 minutes; latency adds time. Empty text produces missing numeric values
without calls. Numeric artifacts are marked validation-only with a different data identity,
excluded from benchmark inventory and rejected by full-data assembly. Exact 1.14 responses
can be reused later; 1.13 responses/ledgers are preserved separately. This does not enable
full Jev extraction. Current model/price/context references were checked on 30 October:
[models](https://docs.typesafe.ai/models), [request errors](https://docs.typesafe.ai/api/errors).

## Future authorized reset, question preparation and upload

The reset/upload already preceded the current frozen campaign. Routine housekeeping
preserves retained results, submissions, backups and recovery state. Use the following
only for a separately authorized fresh campaign **after** B200 tests, local Jev pilot and CPU-control
container migration described above, with all
JEVPFN jobs and manual/notebook writers stopped. Preserve their evidence until the
cause of every failure is understood. Synchronize the same reviewed source first.
For the owner's interim clean debugging rerun, use the cleanup commands below after
jobs stop, then `bash scripts/test_all.sh --execute --watch`. Keep the existing
audited question files and do not freeze yet. The full regeneration/upload/freeze
sequence below is for the final campaign after setup works.

VSC CPU environment, with the durable root and `VSC_SCRATCH_GPFS1` available:

```bash
"${JEVPFN_PYTHON:?}" -m src.utils.output_cleanup --tier both --include-archives
"$JEVPFN_PYTHON" -m src.utils.output_cleanup --tier both --include-archives --apply --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
mkdir -p /lustre1/project/stg_00211/JEVPFN/uploads
```

This deletes output on DATA, durable PROJECT and GPFS working sets (including
known runtimes whose staging failed before its completion marker),
including old token counts/logs/reports. It preserves raw/prepared data, permanent
features/accounting, weights, containers, environments and secrets. There is no
archive. Selective previews support `--phase`, `--kind`, `--match`,
`--older-than-days` and `--tier`. Do not manually delete root directories.

Local fail-fast PowerShell:

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
    scp '.\output_JEVPFN\transfers\decision_inputs_v3.zip' '.\output_JEVPFN\transfers\decision_inputs_v3.zip.sha256' '.\output_JEVPFN\3. experiment_0\manifests\frozen_revision.json' 'vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/uploads/'
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed; do not continue.' }
}
```

The upload contains exactly the 18 selected input SQLite files plus metadata, not
credentials or model weights. Raw data survive the reset; the VSC CPU job rechecks
their hashes. Install from the activated VSC CPU environment:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.input_transfer install /lustre1/project/stg_00211/JEVPFN/uploads/decision_inputs_v3.zip --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.freeze --record --expected-file /lustre1/project/stg_00211/JEVPFN/uploads/frozen_revision.json
```

Installation validates every member/hash/row link before atomic replacement; prior
input versions remain recoverable. Freeze compares local/VSC source including
notebook code and configuration, excluding rendered notebook output. Runtime pins
are site-specific and recorded separately. Do not change a frozen campaign.

## Fresh Experiment 0A

Local PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local --execute --dataset-pilot
if ($LASTEXITCODE -ne 0) { throw '0A_local stopped; inspect its report.' }
```

VSC CPU environment:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.experiment_zero --site vsc
"$JEVPFN_PYTHON" -m src.cluster.experiment_zero --site vsc --execute --watch
```

0A reruns CPU/question/data checks, two-node storage contracts and exact token
counts, stages fresh native working sets, then runs all eleven decision runtimes and
three encoders on B200. Each decision pilot follows its own contract. Local 0A
checks Jev interfaces/robustness and the configured real-dataset pilot. Its bounded
interface probes use a separate permanent ledger for the frozen source digest, so
a new campaign actually tests the live API and an interrupted round reuses its
saved responses. Earlier ledgers and dataset-pilot caches remain preserved. Omitting
`--dataset-pilot` leaves local acceptance incomplete. Receipts are in DATA/local
`3. experiment_0/{manifests,logs}/0A_vsc` or `0A_local`; normal Slurm logs remain.
The VSC watcher resumes recorded IDs after a disconnect and collects independent failures
using the same resource pools as setup. It never accepts a partially successful campaign.

Review outputs, throughput, probability/order sensitivity, quota and independent
backup before accepting. Provider acceptance previews use:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.accept_features --provider intern
```

For an explicit completed campaign, add `--campaign PATH_TO_0A_RECEIPT`. Only after
review add `--approve --reviewed-by Andreas --quota-verified --backup-verified
--throughput-reviewed --robustness-reviewed`. These flags are attestations, not
checks the helper can perform for you. Accepted files are hash-bound to source,
inputs, runtime, token audit and reports. Tests do not sign acceptance automatically.

## Feature creation, 0B and benchmark


After successful frozen 0A review, finish that validation stage explicitly. Enable
the reviewed full-extraction switches and exact provider acceptance paths, enable
the three frozen encoders, and freeze the extraction configuration.
`python -m src.cluster.feature_campaign` previews the graph; `--execute --watch`
resumes its recorded Slurm jobs. It contains 198 decision/dataset GPU allocations,
three full-dataset encoder allocations and eleven stage/release pairs. Each
decision runtime permits two concurrent dataset jobs; at most two working sets
remain resident. These execution partitions preserve every selected row/input.
No Jev API call or downstream predictor is launched. Release publishes committed
results even after a failed consumer and preserves mutable recovery stores.

The final successful job graph still requires complete-artifact validation; it is
not Experiment 0B acceptance. Each dataset's full probabilities/responses, numeric
features, measured timings and hardware/job identities remain available.


Between reviewed stages, with no writers, use
`python -m src.cluster.freeze --finish-stage --reviewed-by Andreas --writers-stopped`
on both machines before changing the next stage's configuration. Record a new
freeze afterward. Do not delete accepted evidence when moving to the next stage.

Full extraction remains disabled. Enable it only after 0A acceptance:

- Open weights: populate `accepted_before_extraction_manifests` per provider and
  enable `full_feature_creation_enabled` in `config/3. experiment_0/self_hosted.yaml`.
- Jev: the private config requires a reviewed `jev_pilot` acceptance with checksummed
  reports, `settings_sha256`, `input_sets`, `context_envelope_sha256`, and a separately
  reviewed cumulative spending/throughput policy. Full dispatch uses
  `python -m src.jev.hosted --config feature_creation/jev.local --execute --full` locally.
- Encoders: enable `config/2. feature_creation/text_embeddings.yaml` only after all
  three real encoder contracts pass. Pins/recipes stay in `embedding_models.yaml`.

Create the phase log directory, restage affected idle runtimes after changed inputs,
and submit accepted workloads. These examples use existing staged B200 workspaces:

```bash
mkdir -p "output_JEVPFN/2. feature_creation/logs"
sbatch scripts/slurm/feature_creation.slurm intern
sbatch --cpus-per-task=14 --mem=112000M scripts/slurm/feature_creation.slurm perplexity
sbatch scripts/slurm/embedding_creation.slurm e5_small_v2
```

Repeat only for accepted providers/encoders. Perplexity, SII, Kev and Decision3 use
the larger resource profile. A 24-hour allocation is a resumable segment, not a
completion estimate. Exact duplicates reuse saved outputs; empty text stays missing.
Do not run duplicate campaigns for the same provider. Review publication receipts
before downloading or marking a dataset complete.

After all features exist, download/import the VSC bundle locally using the final
section below. The reverse direction uses closed SQLite snapshots of the local
dataset stores, uploaded into a separate PROJECT `uploads/jev_features/` directory;
never SCP over populated `data/jev_cache/` files. In the idle VSC CPU environment,
merge each uploaded snapshot transactionally:

```bash
for source in /lustre1/project/stg_00211/JEVPFN/uploads/jev_features/*.sqlite3; do
    test -f "$source" || { echo 'No uploaded Jev cache snapshots found.' >&2; exit 1; }
    "${JEVPFN_PYTHON:?}" -m src.jev.merge "$source" "/lustre1/project/stg_00211/JEVPFN/data/jev_cache/${source##*/}"
done
```

The merge validates stored identities and preserves other providers; imported
spending queues remain archival evidence. Restage predictor working sets afterward.
Before 0B, fill `feature_artifacts` in `config/3. experiment_0/debug.yaml` and the benchmark
`feature_selectors` with immutable artifact selections; finalize predictor settings and
encoder/PCA recipes too. 0B validates row IDs, source hashes, complete provider/encoder coverage,
transfer/resume and all downstream preprocessing boundaries. Run its CPU validation:

```bash
sbatch scripts/slurm/experiment_0_after.slurm
```

Stage the exact predictor assets listed in `config/4. experiment_1/models.yaml` and
build isolated runtimes before GPU tests. AG-MM has its own dependency environment;
do not combine its older Transformers requirement with the extractor runtime.
`predictor_execution_enabled` permits the explicitly authorized bounded fixtures;
each invocation still needs `--execute`, staged assets and a compute allocation.
Full-fold profiling and benchmark switches remain disabled. Example after
its B200 image/checkpoints/working set exist:

```bash
sbatch scripts/slurm/experiment_0_predictor.slurm tabpfn_3_5 tabpfn_3_5
```

TabSTAR selects v1.1 through package 1.2.2, not the original model interface. Its
classification/regression snapshots and E5 dependency are pinned in the model catalog.
The CPU stager writes checksummed offline Hugging Face snapshots and aliases, with one
predictor manifest; it neither loads a model nor modifies tracked configuration.
TabSTAR's v1.1 bounded contracts passed during setup. This does not replace 0B
full-feature/full-fold validation. No model training runs on login nodes.

CatBoost/linear use `experiment_0_predictor_cpu.slurm MODEL ENV_PREFIX` on wICE.
All nine predictors need real task/feature contracts, including binary, multiclass,
30-class, regression, PCA scopes/widths, native text and restarts. Synthetic embedding
fixtures test the downstream interface; they do not replace actual encoder checks.
Also enable/review the separate `full_fold_profile_enabled` test before benchmarking:
all rows must fit without silently sampling/capping data or columns.

Prepare fold-specific PCA on CPU before GPU predictors. Example (repeat selected
18 datasets and three encoders):

```bash
mkdir -p "output_JEVPFN/5. experiment_2/logs"
sbatch scripts/slurm/prepare_pca.slurm REG_TEXT_ROTTEN_TOMATOES e5_small_v2
```

Only training partitions fit PCA; inner and outer projections are distinct and
cached. Restage the relevant predictor after CPU artifacts complete. After reviewed
0B acceptance, set only `experiment_0_acceptance` and the experiment's `enabled` switch
in its configuration. The acceptance file has
`stage: after_extraction`, `accepted: true`, `reviewed_by`, checksummed evidence and
the `evaluation_identity` recorded by the 0B preflight/predictor reports. This binds
the tested implementation, model settings/checkpoints, protocol, selectors and encoder/PCA
recipes. Changing an execution-enable switch or the acceptance-file path does not change
that identity; changing the tested pipeline requires a new reviewed 0B acceptance.
`python -m src.evaluation.acceptance` prints the current identity without signing anything.
Then write an array plan:

```bash
python -m src.cluster.benchmark_jobs --phase experiment_2 --model tabpfn_3_5 --concurrency 8 --write
```

This prints the exact submission command; it does not submit. GPU models default to
B200, CPU models remain wICE. Each task owns one dataset/model/representation and
resumes its five folds through checksummed records. Predictions, probabilities,
inner-OOF predictions, timing and all recorded metrics persist in per-dataset stores.
Experiments 1/2/3 retain the design in README;
GPU routing does not change the statistical protocol.

## Inspect utilization and transfer results

GPU telemetry uses the native host interpreter around the selected model image.
It records only allocated GPUs, at ten-second intervals by default: utilization,
VRAM, power and sampling failures. Timing separates loading, inference/cache reuse,
fit and prediction. Inspect actual measurements before changing batching/resources;
no job is claimed to achieve 100% GPU utilization merely because it requests a GPU.

After all writers stop, recover/publish any failed scratch job first, then export
from the VSC CPU environment with the durable Lustre root:

```bash
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
"${JEVPFN_PYTHON:?}" -m src.cluster.transfer export
```

Download the printed ZIP path, then merge it locally. Substitute the exact returned
filename in this PowerShell example:

```powershell
scp 'vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/output_JEVPFN/transfers/EXACT_BUNDLE.zip' "$env:USERPROFILE\Downloads\"
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
.\.venv\Scripts\python.exe -m src.cluster.transfer import "$env:USERPROFILE\Downloads\EXACT_BUNDLE.zip"
if ($LASTEXITCODE -ne 0) { throw 'Import failed.' }
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Exports include durable data/results and DATA logs/manifests, not credentials,
weights or environments. Import verifies paths/checksums and transactionally merges
compatible databases; conflicting records stop. Do not overwrite local caches with
an unvalidated folder copy. All notebooks run locally and save executed outputs,
paper-sized PDF figures, `figures/Captions.md` and the complete `All Results.md`.
For a smaller independent response/feature backup, use `export --omit-token-counts`.
It retains cache databases, predictions and compact evidence while omitting only
rebuildable detailed token databases. The default export still includes merged counts.
