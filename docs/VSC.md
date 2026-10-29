# VSC operations

Use this guide for execution; use README for the experiment design and
AGENTS_MEMORY for dated observations. **Ten decision runtimes and three encoders have
bounded B200 passes; Decision3 and the new TabSTAR runtime still need inference tests.**
Final acceptance requires the complete frozen campaign and review.

## Storage and hardware

| Role | Location |
|---|---|
| Checkout, CPU environment, small logs/manifests | `$VSC_DATA/JEVPFN` and `$VSC_DATA/conda_envs/JEVPFN` |
| Authoritative raw data, prepared questions, weights, containers, features, predictions and large output | `/lustre1/project/stg_00211/JEVPFN` |
| Active B200 working set, isolated by runtime | `$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN` |
| Local notebooks and downloaded/merged results | Local repository |

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

## Stop only this project's H100 jobs and update

Do not update source used by active jobs. The cancellation utility previews first,
then checks live ownership, H100 partition and a resolved WorkDir/Command inside
this checkout. A matching job name alone is insufficient. CPU jobs, B200 jobs and
other projects are excluded. From the activated CPU environment:

```bash
cd "$VSC_DATA/JEVPFN"
python -m src.cluster.cancel_h100
python -m src.cluster.cancel_h100 --apply
squeue --clusters=wice,mindwell --me
```

If this utility has not reached VSC yet, upload just the helper from local PowerShell:

```powershell
Set-Location 'C:\Users\U0152019\PhD Documents\Projects\5. JEVPFN\JEVPFN'
scp .\src\cluster\cancel_h100.py 'vsc38338@login.hpc.kuleuven.be:/data/leuven/383/vsc38338/JEVPFN_cancel_h100.py'
if ($LASTEXITCODE -ne 0) { throw 'Cancellation helper upload failed.' }
```

Then use the existing CPU interpreter in the VSC terminal:

```bash
cd "$VSC_DATA/JEVPFN"
export PYTHONPATH="$VSC_DATA/JEVPFN${PYTHONPATH:+:$PYTHONPATH}"
"$VSC_DATA/conda_envs/JEVPFN/bin/python" "$VSC_DATA/JEVPFN_cancel_h100.py"
"$VSC_DATA/conda_envs/JEVPFN/bin/python" "$VSC_DATA/JEVPFN_cancel_h100.py" --apply
squeue --clusters=wice,mindwell --me
```

This does not change the active checkout. Never cancel by username.
Wait for the remaining JEVPFN CPU jobs and cancellation to finish, then update:

The tested local implementation, including new files, must first be committed and pushed. A successful
`git pull` cannot retrieve uncommitted local work. PowerShell blocks run on Windows;
Bash blocks run in the VSC terminal. Pasting `Set-Location` or `$LASTEXITCODE` into Bash
does not upload or validate anything.

```bash
cd "$VSC_DATA/JEVPFN"
git pull --ff-only
git submodule update --init --recursive
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.cluster.setup --cluster wice
myquota
```

Use the existing Python 3.12 CPU environment. Only for a new environment, use the
project's existing Conda installation and install `.[dev,staging,token-audit]`, then
run `python -m pip check`. Activation checks the actual interpreter; a changed
shell prompt alone is insufficient. Do not install all GPU extras together or
modify another project's environment.

## Finish setup before the clean campaign

One command prepares and tests all eleven open-weight extractors and three encoders. Preview first:

```bash
bash scripts/test_all.sh
bash scripts/test_all.sh --execute --watch
```

The preview must select Mindwell B200. The script activates the existing CPU environment,
selects the durable root and starts the resumable coordinator. It never cleans output,
regenerates questions or makes paid calls. The prepared questions remain the audited local files.

The graph performs CPU/data checks, immutable decision-model and encoder checkpoint
staging, missing Blackwell container builds, exact token counts, GPFS staging,
B200 contracts, bounded dataset pilots and all three encoder tests. Both filesystems
must pass two-node locking and SQLite snapshot tests: durable Lustre on wICE and
native GPFS scratch on Mindwell's `batch_graniterapids` CPU partition. Every GPU
job waits for those checks. It does not enable full extraction or certify acceptance.
Liquid d1-2B is `liquid_lfm3`; it keeps its own weights, image and token database.
Its CPU partitions wait for its Transformers 5.19 container and run inside that image;
other models retain their existing tokenizer environments. The same question v3 files
are reused: no grammar or class-vocabulary change accompanies this addition.

The complete setup contains 215 jobs with no reusable containers, or 204 when all eleven
sealed images exist. Containers are verified in their own CPU staging jobs, not hashed
serially on the login node. Each runtime has its own two-node GPFS storage check; a
failed model cannot block every other model through a shared staging prerequisite.
Configuration permits 64 queued/running jobs: at most 40 token partitions, six other CPU
checks/merges, 12 GPU tests, two downloads, two builds and one bulk transfer/release. A separate two-workspace residency limit covers staging through the last GPU consumer
and release, including queue waits; it prevents all providers retaining large GPFS copies.
CPU token/download/build branches still proceed independently. Token audits cannot fill the GPU/build pools.
Slurm still decides placement and start times; this does not promise simultaneous starts.
Cross-cluster dependencies are checked through accounting, not assumed to work
through Slurm federation. `--watch` polls once per minute and submits ready steps.
If OnDemand disconnects, already submitted jobs continue; reconnect, activate the
CPU environment and repeat the same command. Without `--watch`, each invocation
reconciles/submits currently ready work. Never launch another independent graph.

The receipt is DATA
`output_JEVPFN/3. experiment_0/manifests/container_setup/open_models_mindwell_b200.json`.
It records exact job IDs and clusters. Inspect a recorded job with:

```bash
sacct --clusters=wice,mindwell -j JOB_ID --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
squeue --clusters=mindwell --me -o "%.18i %.16P %.10T %.10l %.35R"
```

Use the reported integer in place of `JOB_ID`. Successful submission or an empty
queue is not a test result. A failed job blocks only its descendants. Other branches
continue; the same receipt's `summary` lists failed job IDs, commands/exit codes and
blocked tests. The watcher exits nonzero after all runnable tests finish if any failed.
Missing accounting stays pending. Failed/timed-out `sacct` queries also remain pending:
setup and frozen 0A watchers retain job IDs, save the bounded error details, defer new
submissions and retry on their next minute poll. At most one failed query per cluster
is made in a pass; the other cluster can still have its recorded jobs reconciled.
Recovery resumes the same jobs and retains the last error for diagnosis. It never
converts an unavailable status into success. Ambiguous/interrupted `sbatch` submissions
still stop the coordinator to prevent duplicate jobs. Keep its receipt; never erase it
to guess retries. An older launcher already marked `stopped_requires_review` needs the
idle/restart procedure below after accounting recovers; do not manually edit its receipt.
Submission errors now display their saved stdout/stderr; frozen 0A also retains these
diagnostics. Failed storage workers print bounded log tails into the parent job log.
Controller DNS/socket errors are infrastructure failures, not model or credit-balance
failures. Submitted jobs can continue while the coordinator is stopped. A timed-out
sbatch may have been accepted: inspect squeue and sacct on that cluster before any
retry. Do not clear a starting receipt or update source while existing jobs use it.

If accounting confirms that a timed-out submission was accepted and is now terminal,
record its ID before restarting. The helper queries [Slurm accounting](https://slurm.schedmd.com/sacct.html)
and matches the user, account, checkout and full submission command. It refuses active,
missing or mismatched records; applying it also requires all JEVPFN jobs to be idle.
It preserves diagnostics and the stopped status and never submits a replacement or
claims test acceptance. For the confirmed 26 October GPFS job:

```bash
"$JEVPFN_PYTHON" -m src.cluster.setup_recovery --step liquid_gpfs_storage --job-id 11739854
"$JEVPFN_PYTHON" -m src.cluster.setup_recovery --step liquid_gpfs_storage --job-id 11739854 --apply
```

The first command previews; the second updates only the stopped setup receipt.
Then use the restart command below. If accounting cannot identify the submission,
keep the receipt unchanged and investigate; this helper provides no force override.

Within a provider, independent GPU diagnostic groups and real-dataset checks collect
failures before returning a failing job. A poisoned CUDA context, memory exhaustion or
request ceiling aborts that worker. CPU library, question and synthetic test commands
also all run even if an earlier command fails. Neither behavior turns a failure into a pass.

After fixing a stopped setup or changing its source, wait until all JEVPFN jobs
finish, then explicitly restart the checks:

```bash
"$JEVPFN_PYTHON" -m src.cluster.setup_extractors --all --restart
"$JEVPFN_PYTHON" -m src.cluster.setup_extractors --all --restart --execute --watch
```

Restart checks both clusters for active JEVPFN jobs and requires accounting for
every recorded job. An ambiguous submission is refused. It preserves earlier
attempts inside the same receipt, verifies and reuses sealed Blackwell images,
and retains weights, token batches, caches and logs. Cheap checks run again;
the previous failure is never relabelled as a pass. After a disconnect, resume
with `--all --execute --watch` without `--restart`.

The 28 October download completed 174/176 steps. All ten token merges passed; the
earlier five merge timeouts are resolved. Nine open-weight contracts/pilots passed.
Decision3 failed before inference because its publisher runtime requires an explicit
CUDA index. The adapter now uses the current logical device (respecting Slurm's visible
device mapping). This fix does not change questions, token identities or images.

After publishing/pulling with JEVPFN jobs idle, the targeted setup command is:
Export the roots and activate the CPU environment as above, then:

```bash
"$JEVPFN_PYTHON" -u -m src.cluster.setup_extractors \
  --provider decision3 --provider liquid_lfm3 --restart --execute --watch
```

This creates a targeted diagnostic attempt reusing the sealed image and completed
token counts, and retains the complete previous attempt inside the same receipt.
It repeats CPU/storage prerequisites and verifies cached inputs before submitting the
GPFS/B200 descendants. After a disconnect, use `--provider decision3 --provider liquid_lfm3 --execute --watch`
and omit `--restart`; the `--all` wrapper would select a
different provider list. The final clean 0A campaign still tests all providers/encoders.

The 29 October Decision3 preparation completed in 7m39s and all ten partitions passed.
Merge 62463241 subsequently completed 0:0 in 2h47m55s. Its verified counts are reusable.
The first cleanup preview stopped on a generated Transformers import cache; the narrow
check is corrected, and preview/apply must be retried. No cleanup was applied. Liquid d1-2B's B200 contract and
18-dataset pilot passed. TabSTAR's stage 62463463 and container 62463466 succeeded;
scratch stage 62463487 failed on the personal GPFS byte quota. Its storage/GPU tests
were not submitted. Recover space using the section below after active JEVPFN jobs finish.
A changed setup graph requires an idle `--restart`; it retains prior receipts and reuses
valid assets/counts. After a disconnect on the same source/graph, omit `--restart`.

The 26 October Liquid/Strands GPFS staging failure was a link-checksum normalization
mismatch: Tykky writes image paths containing `bin//python`, while the seal used the
equivalent `bin/python`. Sealing and relocation now share that normalization. Existing
sealed images and copied trees can be reused; do not rebuild them or bypass verification.
Actual changes to links, images or wrappers still fail, with mismatched filenames reported.

The `--all` graph includes E5-small-v2, E5-large-v2 and Qwen3-Embedding-0.6B. Their
CPU checkpoint-staging job finishes before the Clef runtime's GPFS working set is
copied; their B200 tests then use that sealed runtime. There is no separate manual
encoder download step. A provider-only setup without Clef deliberately omits encoders.
Checkpoint downloads are allowed in connected CPU Slurm allocations; a Slurm job
ID alone is not a GPU allocation. GPU allocations and frozen-campaign encoder
downloads are refused before any network work.
Mindwell activation resolves and verifies the prepared native GPFS workspace before
inference. No GPU job downloads weights or copies a missing working set. Contract
tests and pilots check actual
probabilities, class/score order, exact token parity, batching, missing/duplicate
inputs, context rejection, cache/row mappings and prompt sensitivity. Semantic
failures must be reviewed rather than hidden by relaxing tests. Larger pilots are
capped at 1,000 requests/provider; their timings guide throughput estimates.

Each provider's audit uses ten independent jobs, each counting a disjoint ~10% of every
dataset's unique inputs. A short recovery job first rolls back any hot SQLite journal
left by cancellation and checks the existing database before parallel readers start.
Each partition writes one PROJECT SQLite file; no parallel worker writes
the final database. Existing matching monolithic counts are reused, including committed
batches from cancelled jobs. A separate CPU merge checks identities, file hashes, exact input
coverage and prepared-token contents, and rolls back all merge inserts on any conflict.
The final counts retain their original schema/identity for GPU consumers. Restart partitions
stay on PROJECT for recovery; they are excluded from GPFS copies and downloaded result bundles.
The serial preparation step recovers interrupted SQLite transactions and checks the compact
identity/count tables and prepared-token key coverage. It no longer scans all token blobs
with a whole-database quick_check before the parallel work. Existing prepared caches get
a compact lookup index if needed. Elapsed-stage messages identify recovery, table checks,
each dataset and key coverage. This step is not final acceptance: the merge still verifies
every input, partition checksum, compressed token checksum and count before GPU use.
The five added renderers use eight CPUs/27,200 MiB per partition and a 12-hour limit; the
original five and Liquid d1-2B use two CPUs/6,800 MiB and four hours. Progress logs report throughput at least
once per minute between batches. Shorter partitions improve parallelism, but queueing, text
length and I/O prevent a guaranteed tenfold speedup. Frozen 0A contains 192 VSC jobs: eleven recovery checks, 110 token
partitions, eleven merges, CPU checks, one Lustre check, eleven scratch stages, eleven native GPFS checks,
eleven GPU contracts, eleven bounded pilots, three encoder tests and eleven workspace releases. Setup also downloads/builds.

To migrate an active monolithic audit, first stop its watching coordinator with Ctrl+C.
Cancel only its explicitly verified token job IDs, leave container builds running, and wait
for those builds to finish before pulling new source. Do not clean output during migration.
Then run `bash scripts/test_all.sh --restart --execute --watch`; it preserves assets and
committed token batches. After a disconnect, omit `--restart` to resume the new graph.

Mindwell's observed BSD `flock` did not exclude a second node. Linux locks and workspace
leases now use POSIX `lockf`, with thread serialization to respect process-owned lock semantics.
The two-node exclusion/release/snapshot test remains mandatory on both filesystems. Never mix
old and new lock implementations in simultaneous writers. Checkpoint staging no longer treats
publisher `.lock` assets as transient files; real hash mismatches still stop and list filenames.

After a kill or failed publication, preserve scratch and recover committed results
in the CPU environment before restaging that runtime:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.scratch_stage publish --runtime intern
sbatch scripts/slurm/experiment_0_scratch_stage.slurm intern
```

Replace `intern` with the affected runtime. Publication never deletes the working
set. Restaging refuses active runtime leases. A semantic cache conflict requires
inspection; do not replace populated databases manually.

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
pinned `jev-1.13.0` binary, multiclass and nine-level outputs; robustness uses a
separate bounded diagnostic. Existing paid responses/ledgers are reused, including
after output cleanup. `--revalidate-only` reprocesses saved responses without API
calls; never erase uncertain attempts to obtain a retry.

The owner confirmed 80 RPS and 100,000 input TPS on 21 October; recheck the account's
[rate-limits page](https://console.typesafe.ai/rate-limits) before full extraction.
The pilot deliberately runs one worker, at least five seconds apart. The documented
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

## One clean reset, question preparation and upload

Do this **after** the B200 setup/encoder tests and local Jev pilot work, with all
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
counts, stages fresh native working sets, then runs all ten decision runtimes and
three encoders on B200. Each decision pilot follows its own contract. Local 0A
checks Jev interfaces/robustness and the configured real-dataset pilot. Omitting
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
The 29 October download confirms its weights and versioned container are ready.
The GPFS copy failed on quota; after scoped scratch recovery, resume the final three
steps below. The first two commands are needed only for a new installation or a
failed/missing authoritative checkpoint/container. Staging retries transport failures
at most three times; valid pinned downloads are reused and integrity errors stop.
Do not mix TabSTAR's Transformers 4.57.6 environment with the decision-tokenizer environment.
Its new `tabstar-v1p1` container path preserves old images. Once JEVPFN jobs are idle,
and before a frozen stage, run in the VSC Bash terminal:

```bash
(
set -euo pipefail
cd "$VSC_DATA/JEVPFN"
mkdir -p "output_JEVPFN/3. experiment_0/logs"
sbatch --wait scripts/slurm/stage_tabstar.slurm
sbatch --wait scripts/slurm/experiment_0_container_build.slurm tabstar new /lustre1/project/stg_00211 blackwell-cu130
sbatch --wait scripts/slurm/experiment_0_scratch_stage.slurm tabstar
sbatch --wait scripts/slurm/experiment_0_storage_gpfs.slurm tabstar
sbatch --wait scripts/slurm/experiment_0_predictor.slurm tabstar tabstar
sbatch --wait scripts/slurm/experiment_0_scratch_release.slurm tabstar --apply
)
```

These are prerequisite-ordered jobs for this one runtime; the extractor/encoder graph
can progress independently through `test_all`. A source update requires idle JEVPFN jobs;
do not pull into an active checkout. Bounded TabSTAR fits can be tested before full feature
creation, but they are 0B predictor-interface evidence, not complete 0B acceptance. The
full-feature and full-fold checks must run again after extraction. No weights or model
training are run on login nodes. Required adaptation retains the publisher's fixed
LoRA/three-epoch averaging recipe; temporary epoch checkpoints are not retained as results.

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
Experiment 1/2/3 and optional exploratory follow-ups retain the design in README;
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
