# VSC operations

The [README](../README.md) defines the benchmark. The [operational record](AGENTS_MEMORY.md)
holds dated job status, receipts and recovery history. For the active campaign, the next
stage begins with [feature verification and 0B](#46-experiment-0b) after extraction and
publication finish. Keep the DATA checkout frozen while it has consumers.

## 1. Access

The owner renews the shared Windows-agent certificate manually, only when access fails:

```powershell
step ssh login --context VSC
```

The owner approves KU Leuven Authenticator. Use native Windows OpenSSH with strict host-key
checking and the bounded LF stdin probe in [TEMPLATE](TEMPLATE.md). The gateway drops
unredirected stderr, so start the script with `exec 2>&1` and require both output markers
and a successful exit. Git Bash's `ssh` cannot use the Windows agent.

Inspection and workload authorization are separate. When the checkout is idle, fast-forward
`main`, verify the commit and rerun affected checks before freezing the next stage.
Authentication and monitoring boundaries are in [AGENTS](../AGENTS.md).

## 2. Storage and hardware

| Role | Location |
|---|---|
| Git checkout with `.git`, specs, compact logs/manifests | `$VSC_DATA/JEVPFN` (`/data/leuven/383/vsc38338/JEVPFN`) |
| Authoritative data, questions, weights, containers, features, predictions, large results | `/gpfs1/project/prj_00013/JEVPFN` (PROJECT, GPFS) |
| CPU controller image | `<allocation>/JEVPFN/containers/control-v1`, selected by `control.active.json` |
| Active B200 working set, one per runtime | `$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN` |
| Notebooks and merged results | Local repository |

- `scripts/storage_environment.sh` reads the ignored `.vsc-storage.local.json` or an
  explicit `JEVPFN_DURABLE_ROOT`/`JEVPFN_PROJECT_ROOT`. Set the allocation parent. The
  resolver appends `JEVPFN`. `JEVPFN_STAGING_ROOT` points to a separate working root.
- Inference uses Mindwell B200. CPU staging, builds, token counts and PCA use
  `batch_graniterapids` for this GPFS deployment. An explicitly selected wICE route needs
  native Lustre staging or node-local transfer I/O. B200 uses 32,000 MiB for four CPUs
  and 112,000 MiB for fourteen. CPU jobs use the partition default per core. A100 is excluded.
- Jobs read immutable assets directly from GPFS PROJECT and isolate mutable stores on
  native scratch. They publish compatible SQLite unions every 600 seconds and on exit
  under per-destination locks. Periodic publication can defer a busy store. Final publication
  waits. Dataset jobs publish their own store, and resumed work reuses committed records.
- The [scheduling configuration](<../config/3. experiment_0/scheduling.yaml>) allows two
  resident workspaces, a 200 GiB working budget and 20 GiB headroom in a 500 GiB scratch
  allocation. Check both bytes and files with `myquota`. Retain an independent feature
  copy because unused scratch files can be purged and failed jobs can lose unpublished work.

The infrastructure references are the pinned [VSC Library](../vsc-library/README.md)
guides. Check the current pin with `git submodule status vsc-library`. The completed
Lustre-to-GPFS migration and its quiescent copy protocol are in the
[operational record](AGENTS_MEMORY.md#gpfs-project-migration-and-resume-34-november).

## 3. Runtimes and containers

All cluster dependencies are sealed, compact PROJECT images; build files use temporary
scratch. Keep old images only while a recorded run or rollback needs them.

- **CPU controller.** `config/3. experiment_0/control_runtime.yaml` records exact package
  versions. `control_build.slurm` builds `control-v1`; `control_test.slurm` tests Slurm
  accounting, nested model launches, two-node locking/snapshots, data checks and the CPU
  suite. `python -m src.cluster.control_runtime activate --test-job-id JOBID` activates it
  from three passing reports of one job on an idle checkout. After native staging,
  `control_gpu_test.slurm` verifies the image and a CUDA kernel on B200; only then
  `control_runtime retire-legacy` may remove the old DATA prefix (already done). CPU
  preflight follows the selected durable filesystem: Mindwell batch_graniterapids for
  the GPFS deployment, with the partition memory-per-core default.
- **B200 images.** Separate `containers/blackwell-cu130/<runtime>/` images keep publisher
  pins with the exact cu130 Torch/torchvision wheels (the cu128 index lacks them; CUDA 13
  needs driver branch >= 580). Startup checks CUDA 13.0, native `sm_100`, BF16 and the B200
  device. These checks do not establish model correctness; never substitute newer packages
  to bypass a failed pin.
- **Model-image probes** inside a CPU-controller job use an overlapping host Slurm step;
  rebuildable compiler caches use a per-job scratch leaf removed on exit.
- **Predictor runtimes.** Review the checkpoint and runtime pins in
  [models.yaml](<../config/4. experiment_1/models.yaml>). Before freezing 0B,
  `python -m src.cluster.predictor_assets --execute` stages the pinned TabSTAR snapshots on
  CPU. TabSTAR v1.1 uses `tabstar-v1p1`. AG-MM uses a separate image.
- Question ZIPs install with `input_install.slurm`, which refuses other JEVPFN consumers.

### CPU environment parity

For manual VSC commands below, activate the controller from the DATA checkout.

```bash
cd "$VSC_DATA/JEVPFN"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
```

The Windows notebook venv and sealed Linux CPU image share exact versions for the
packages listed in `config/3. experiment_0/control_runtime.yaml`. The `control`
extra in `pyproject.toml` pins the scientific subset. To reproduce the full shared
package set in an existing local Python 3.12 venv, generate a temporary requirements
file from the lock and install it into that venv.

```powershell
$jevControlLock = Join-Path $env:TEMP 'jevpfn-control-requirements.txt'
.\.venv\Scripts\python.exe -c "from src.utils.config import load_yaml; print('\n'.join(f'{k}=={v}' for k,v in load_yaml('experiment_0/control_runtime')['packages'].items()))" | Set-Content -Encoding ascii $jevControlLock
.\.venv\Scripts\python.exe -m pip install -r $jevControlLock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.cluster.preflight --runtime-parity
```

The verifier reports package parity separately from interpreter parity. Windows
currently uses CPython 3.12.10 and Windows wheels. The VSC CPU image uses Linux
CPython 3.12.15. Notebook and hosted API extras are local. The sealed, model-specific
Linux/CUDA images provide the benchmark runtimes. An active VSC campaign keeps
its frozen checkout and sealed images through completion.


## 4. Pipeline

### 4.1 Reset and upload (future authorized campaign only)

For a separately authorized new campaign, reconcile existing jobs, stop JEVPFN writers,
and verify the reviewed source on both machines. Preview with
`python -m src.utils.output_cleanup --include-archives` on each machine. Add `--tier both`
on VSC. Review the listed targets before running the destructive reset below.

```bash
"${JEVPFN_PYTHON:?}" -m src.utils.output_cleanup --tier both --include-archives --apply --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster mindwell
mkdir -p /gpfs1/project/prj_00013/JEVPFN/uploads
```

This permanently deletes output on DATA, PROJECT and GPFS working sets, and keeps data,
features, accounting, weights, containers, environments and secrets. Locally:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    Set-Location 'C:\Users\U0152019\PhD Documents\Projects\5. JEVPFN\JEVPFN'
    function Invoke-Jev {
        & .\.venv\Scripts\python.exe @args
        if ($LASTEXITCODE -ne 0) { throw 'Python failed; remaining steps stopped.' }
    }
    Invoke-Jev -m src.utils.output_cleanup --include-archives --apply --writers-stopped
    Invoke-Jev -m src.data.prepare
    Invoke-Jev -m src.jev.preparation --rebuild
    Invoke-Jev -m src.jev.question_audit
    Invoke-Jev -m src.utils.run_notebooks
    Invoke-Jev -m src.cluster.freeze --record
    Invoke-Jev -m src.cluster.input_transfer pack
    scp '.\output_JEVPFN\transfers\decision_inputs_v3.zip' '.\output_JEVPFN\transfers\decision_inputs_v3.zip.sha256' '.\output_JEVPFN\3. experiment_0\manifests\frozen_revision.json' 'vsc38338@login.hpc.kuleuven.be:/gpfs1/project/prj_00013/JEVPFN/uploads/'
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed; do not continue.' }
}
```

The upload holds the 18 question stores and metadata, never credentials or weights.
Install the question bank on Mindwell CPU and verify the matching freeze after the job
passes. Installation checks bundle members, hashes and row links before replacement.

```bash
mkdir -p "output_JEVPFN/3. experiment_0/logs"
sbatch --wait scripts/slurm/input_install.slurm &&
  "$JEVPFN_PYTHON" -m src.cluster.freeze --record --expected-file /gpfs1/project/prj_00013/JEVPFN/uploads/frozen_revision.json
```

An idle, unfrozen setup for a later campaign runs `bash scripts/test_all.sh --execute --watch`:
it stages pinned models, builds images, partitions exact-token audits and runs CPU/GPU
contracts and pilots concurrently. Resume saved job IDs; never blindly resubmit an
uncertain submission.

### 4.2 Experiment 0A

```powershell
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local
.\.venv\Scripts\python.exe -m src.cluster.experiment_zero --site local --execute --dataset-pilot
if ($LASTEXITCODE -ne 0) { throw '0A_local stopped; inspect its report.' }
```

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.experiment_zero --site vsc
"$JEVPFN_PYTHON" -m src.cluster.experiment_zero --site vsc --execute --watch
```

Local 0A checks dependencies, the full question mapping, synthetic contracts, Jev
interfaces, a bounded wording/repetition/instruction diagnostic (15 calls) and, with
`--dataset-pilot`, the real-dataset pilot plus 13 real-text variants; without it local
acceptance is incomplete. Diagnostic rounds use a ledger per frozen source; interrupted
rounds reuse saved responses. VSC 0A reruns CPU/question/data checks, two-node storage
contracts and exact token counts (ten CPU partitions, recovery check and coverage-checked
merge per provider), stages working sets and runs all eleven decision contracts/pilots and
three encoder contracts on B200. The watcher resumes recorded job IDs, blocks only the
descendants of a failure and never accepts a partial campaign. Receipts go to
`3. experiment_0/{manifests,logs}/0A_vsc` or `0A_local`.

### 4.3 Acceptance

Review outputs, throughput, probability/order sensitivity, quota and independent backup:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.accept_features --provider intern
```

Add `--campaign PATH_TO_0A_RECEIPT` to select an exact campaign. Only after review add
`--approve --reviewed-by Andreas --quota-verified --backup-verified --throughput-reviewed
--robustness-reviewed`; these flags are attestations, not checks. Acceptances are
hash-bound to source, inputs, runtime, token audit and reports; tests never sign them.
Between reviewed stages, with no writers, run on both machines:

```bash
python -m src.cluster.freeze --finish-stage --reviewed-by Andreas --writers-stopped
```

then reconfigure and record a new freeze. Accepted evidence is never deleted at a transition.

### 4.4 Open-weight and encoder extraction

For a fresh, authorized campaign, configure `accepted_before_extraction_manifests` and
`full_feature_creation_enabled` in `self_hosted.yaml` and enable `text_embeddings.yaml`.
Run from the frozen checkout with the controller active.

```bash
mkdir -p "output_JEVPFN/2. feature_creation/logs"
python -m src.cluster.feature_campaign
python -u -m src.cluster.feature_campaign --execute --watch
```

**Active campaign.** The reviewed `src.cluster.feature_admission` controller owns the
expanded 274-step graph. It schedules 198 decision dataset jobs and 54 encoder dataset
jobs, plus staging and release. The output roster is 396 decision blocks and 54 embedding
stores. Ready datasets queue after native staging. Workspace quotas and the bulk-transfer
limit still apply. See the [recorded handover](AGENTS_MEMORY.md#parallel-admission-2-november).

**Monitor and recover.** Reconcile the manifest, Slurm accounting, and publication receipts.
A step marked `starting` without a job ID needs accounting review before retry. Named
submission-limit refusals remain pending. Other policy errors or timeouts need review.
The 2 November snapshot measured 500 submitted and 16 running slots per user/account
on Mindwell. Arrays use one submission slot per pending task. Recheck site limits as needed.

For reviewed recovery, wait for all known jobs to become terminal, archive the freeze,
test the idle checkout, and record the next freeze. `feature_recovery_test.slurm` checks
saved features and numerical source without replacing 0A evidence. Choose the recovery
entry point for the graph recorded in the manifest.

| Graph | Recovery command |
|---|---|
| Original unpartitioned | `python -u -m src.cluster.feature_campaign --execute --watch --recover --reviewed-by NAME` |
| Expanded parallel | `python -u -m src.cluster.feature_admission --recover-from OLD_EXECUTION_ID --previous-freeze ARCHIVED_COMPLETED_REVISION --expected-source NEW_SOURCE --reviewed-by NAME --execute --watch` |

Recovery preserves completed IDs and numerical artifacts and refuses changes to scientific
configuration or runtime pins. Keep the scratch recovery state until publication is verified.

### 4.5 Jev (local, owner-run)

The local CLI loads the ignored `secrets/typesafe.local.json`. `TYPESAFE_API_KEY` takes
precedence. Keep credentials local. The following paid validation commands are for
an authorized Jev validation stage.

```powershell
.\.venv\Scripts\python.exe -m src.jev.setup_pilot --check-account
.\.venv\Scripts\python.exe -m src.jev.smoke --execute
.\.venv\Scripts\python.exe -m src.jev.smoke --robustness --execute
.\.venv\Scripts\python.exe -m src.jev.hosted --config feature_creation/jev.local --execute
```

- The account check is one authenticated model-list GET. Smoke tests check pinned
  `jev-1.14.0` binary, multiclass and nine-level outputs. Saved paid responses are reused;
  `--revalidate-only` reprocesses them without calls. Never erase uncertain attempts.
- Limits: owner-confirmed 80 RPS and 100,000 input TPS (21 October; recheck the
  [rate-limits page](https://console.typesafe.ai/rate-limits) before full extraction).
  Provider-enforced 32k state/question and 64k request bounds; the 64k billing
  reservation is a spending bound, not a token count. Overflow stops without truncation.
- `python -m src.jev.setup_pilot --configure` creates the private config (USD 0.042/M,
  conservative EUR/USD, EUR 5 pilot budget) and never replaces an existing one; review
  `config/2. feature_creation/jev.local.yaml` instead. The reviewed pilot runs one worker
  at 1 RPS (setup default 0.2 RPS).
- Durable reservations, bounded 429 handling and timeout stops prevent repeat billing.
  `python -m src.jev.recovery DATASET_ID` inspects stopped requests without calls;
  `--review FILE --apply --writers-stopped` transitions preserve charges and history.

The first-100-row validation passed on all 18 datasets. Its 7,643 requests produced
validation-only artifacts. See the [Jev accounting record](AGENTS_MEMORY.md#jev-accounting).
`python -m src.jev.prefix_pilot` previews a repeat.

Full Jev needs a reviewed `jev_pilot` acceptance in the private config (checksummed
reports, `settings_sha256`, `input_sets`, `context_envelope_sha256`) and a reviewed
cumulative spending/throughput policy, then
`python -m src.jev.hosted --config feature_creation/jev.local --execute --full`.
References checked 30 October: [models](https://docs.typesafe.ai/models),
[request errors](https://docs.typesafe.ai/api/errors).

To add local Jev responses to VSC, upload closed SQLite snapshots to PROJECT
`uploads/jev_features/` (never over `data/jev_cache/`) and merge them in an idle CPU job:

```bash
for source in /gpfs1/project/prj_00013/JEVPFN/uploads/jev_features/*.sqlite3; do
    test -f "$source" || { echo 'No uploaded Jev cache snapshots found.' >&2; exit 1; }
    "${JEVPFN_PYTHON:?}" -m src.jev.merge "$source" "/gpfs1/project/prj_00013/JEVPFN/data/jev_cache/${source##*/}"
done
```

The merge validates identities and preserves other providers. Restage predictor working sets afterwards.

### 4.6 Experiment 0B

After extraction and publication finish, reconcile terminal jobs and verify the durable
feature stores. Import the independent local copy using [the transfer procedure](#6-utilization-and-transfer).
Include complete owner-run Jev features when qualifying the full roster.

1. **Bind inputs.** Close the reviewed extraction stage using [the freeze procedure](#43-acceptance).
   Set each benchmark `feature_selectors` entry to the provider, model,
   version, question bank, question variant and `settings_sha256`. An optional
   `feature_artifacts` list in `debug.yaml` must match the selected artifact IDs exactly.
2. **Stage runtimes and prepare PCA.** Stage the reviewed predictor images and checkpoints.
   Set the 0B fixture and profile switches (`predictor_execution_enabled` and
   `full_fold_profile_enabled`), then record the 0B freeze. Submit one CPU preparation per
   dataset and encoder (18 × 3). Each job writes joint PCA64 and separate PCA32 for the
   exact outer and binary inner training partitions.

   ```bash
   mkdir -p "output_JEVPFN/5. experiment_2/logs"
   sbatch scripts/slurm/prepare_pca.slurm REG_TEXT_ROTTEN_TOMATOES e5_small_v2
   ```

   **Prepare TAR before this preflight too.** Follow [the TAR setup](TAR.md#vsc-preparation-after-current-extraction-finishes): build its separate sealed image, prepare five adapters and both PCA recipes for each dataset. TAR uses native GPFS PROJECT directly for immutable bundles; Trainer temporary checkpoints use per-job scratch.

3. **Run preflight and predictor contracts.** After PCA and TAR are published, submit the CPU
   preflight and a contract job for each of the nine predictors. For each GPU runtime,
   stage its working set first. This example uses TabPFN. CPU predictors use
   `experiment_0_predictor_cpu.slurm MODEL ENV_PREFIX`.

   ```bash
   mkdir -p "output_JEVPFN/3. experiment_0/logs"
   sbatch scripts/slurm/experiment_0_after.slurm
   sbatch --wait scripts/slurm/experiment_0_scratch_stage.slurm tabpfn_3_5 &&
     sbatch scripts/slurm/experiment_0_predictor.slurm tabpfn_3_5 tabpfn_3_5
   ```

   Preflight checks the dataset, feature and PCA identities and checksums, the CPU package
   lock and the offline tests. It reports exact cross-fold feature-row overlap and rejects
   exact target copies and class recodings. The predictor fixtures exercise binary,
   multiclass including 30 classes, and regression tasks. They check missing and categorical
   inputs, both PCA recipes at both encoder widths, class order and resume.
4. **Profile full folds.** Keep each GPU working set through its contracts, profiles and
   verified publication. Release it before staging beyond the workspace limit. List the required arms with
   `python -m src.evaluation.qualification`. Generate the matching arrays for each phase
   and predictor. The command prints a reviewed submission command.

   ```bash
   python -m src.cluster.benchmark_jobs --phase experiment_2 --model tabpfn_3_5 --profile-fold --write
   ```

   The generated profile plan covers the tall and wide extremes of every phase, including
   both frozen and TAR PCA recipes. Use its current count rather than an old fixed total.
   Each fits the first outer fold at full size, retains binary inner-OOF predictions
   except for TAR’s fixed-threshold workflow, and
   verifies predictor width, row IDs and targets. Review resource use and publication.
5. **Accept the stage.** Add the current preflight, nine contract reports and all required
   profiles to the checksummed evidence list. Verify the owner's reviewed record with
   `python -m src.evaluation.qualification --acceptance FILE` in a CPU allocation against
   durable PROJECT. It reads the published predictions. Benchmark jobs require the same
   dataset, folds, features and sealed predictor runtime that passed 0B.

### 4.7 Benchmark

After reviewed 0B acceptance (`stage: after_extraction`, `accepted: true`, `reviewed_by`,
checksummed evidence and the `evaluation_identity` from the 0B reports), close the 0B freeze.
Set `experiment_0_acceptance` and the experiment's `enabled` switch, then freeze the
benchmark stage. Changes to the tested pipeline require fresh acceptance.
`python -m src.evaluation.acceptance` prints the current identity. Write an array plan
to print its submission command.

```bash
python -m src.cluster.benchmark_jobs --phase experiment_2 --model tabpfn_3_5 --concurrency 8 --write
```

Each task owns one dataset × model × representation and resumes its five folds from
checksummed records. GPU models default to B200; CPU models follow the native route of the allocated
durable root (Mindwell for this GPFS deployment).

## 5. Working sets and storage maintenance

A stage holds a working-set slot from submission until its consumers are terminal and its
release succeeds. In the GPFS shared-asset deployment, immutable inputs/images are read
directly from PROJECT; release publishes committed mutable stores, then retires only
verified redundant working copies. PROJECT retains response history, features, embeddings,
PCA and predictions; DATA retains compact operational evidence. A publication conflict,
unknown recovery snapshot or consumer keeps the workspace and blocks further staging.
Legacy mirrored workspaces retain their mutable queues until an explicit reviewed
migration retirement. Manual release, after
all JEVPFN jobs and watchers stop (preview, then apply):

```bash
sbatch --wait scripts/slurm/experiment_0_scratch_release.slurm all
sbatch --wait scripts/slurm/experiment_0_scratch_release.slurm all --apply
myquota
```

Logs: `output_JEVPFN/3. experiment_0/logs/jevpfn-scratch-release_<jobid>.log`. Use a runtime
name instead of `all` for one working set. The helper checks identity, sources, consumers
and an exclusive lease, refuses on unknown scheduler state and never cleans the scratch
root. A released runtime must be restaged before GPU work; durable checkpoints and images
need no rebuild.

Retention exceptions, each retired only by a reviewed, scoped cleanup with no consumers:

- Token recovery partitions (~39 GiB) stay on PROJECT through the first verified
  production run; they are excluded from GPU staging.
- Six legacy cu126 image trees (~24 GiB) are retained on the new PROJECT for historical
  reproducibility; they are not selected for B200. Their individually reviewed seals
  and sizes are in `storage_migration/retention_review.json`. No old-host copy is needed
  once relocation acceptance and independent backup pass.
- Build scratch is temporary. The build script does not retire successful build leaves
  itself; clean only verified owned leaves after the owner reviews the plan.

## 6. Utilization and transfer

GPU telemetry samples the allocated GPUs every ten seconds (utilization, VRAM, power).
Timings separate loading, inference, cache reuse, fit and prediction; inspect them before
changing batch sizes or resources.

After all writers stop and failed scratch jobs are published, export on VSC and import
locally (substitute the printed bundle name):

```bash
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
"${JEVPFN_PYTHON:?}" -m src.cluster.transfer export
```

```powershell
scp 'vsc38338@login.hpc.kuleuven.be:/gpfs1/project/prj_00013/JEVPFN/output_JEVPFN/transfers/EXACT_BUNDLE.zip' "$env:USERPROFILE\Downloads\"
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
.\.venv\Scripts\python.exe -m src.cluster.transfer import "$env:USERPROFILE\Downloads\EXACT_BUNDLE.zip"
if ($LASTEXITCODE -ne 0) { throw 'Import failed.' }
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Exports contain durable data, results and DATA logs/manifests, never credentials, weights
or environments; `export --omit-token-counts` omits only rebuildable token databases.
Import verifies checksums and merges databases transactionally, stopping on conflicts.
VSC freeze receipts land under `manifests/imported_revisions/`, never as the active local
freeze. Keep the VSC copy as the second copy.
