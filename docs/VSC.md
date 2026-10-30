# VSC operations

Commands and site rules in execution order. The design is in [README](../README.md);
dated run evidence, job IDs and failures are in [AGENTS_MEMORY](AGENTS_MEMORY.md).

Current stage: 0A is accepted and the open-weight/encoder extraction campaign runs from
the frozen DATA checkout at `954c93f`. Do not reset successful output, and do not pull
into that checkout while any JEVPFN job is queued or running.

## 1. Access

The owner renews the shared Windows-agent certificate manually, only when access fails:

```powershell
step ssh login --context VSC
```

They approve KU Leuven Authenticator. Agents never initiate login, install another agent,
copy keys or schedule reminders. Connect with native Windows OpenSSH
(`C:\Windows\System32\OpenSSH\ssh.exe -F none -T`, strict host-key checking) and send a
bounded LF Bash script on stdin, as in TEMPLATE.md. This gateway drops unredirected remote
stderr, so the script starts with `exec 2>&1` and prints start/end/status markers; empty
exit-0 output proves nothing. Git Bash's own `ssh` cannot reach the Windows agent.

Access authorizes inspection of this project's files, logs, quotas and jobs only, not
submissions, cancellations, migration or deletion. All checkouts use `main`; a frozen VSC
checkout may lag GitHub while it has consumers. When idle, fast-forward it, verify the
commit and rerun relevant checks before the next freeze; never force it.

Authorized long campaigns get one temporary 30-minute monitor with a recorded ID, owner,
jobs, source and completion criteria. It stays quiet without changes, pauses on expired
credentials and is removed once all tracked work is terminal and assessed.

## 2. Storage and hardware

| Role | Location |
|---|---|
| Git checkout with `.git`, specs, compact logs/manifests | `$VSC_DATA/JEVPFN` (`/data/leuven/383/vsc38338/JEVPFN`) |
| Authoritative data, questions, weights, containers, features, predictions, large results | `/lustre1/project/stg_00211/JEVPFN` (PROJECT, Lustre) |
| CPU controller image | `PROJECT/JEVPFN/containers/control-v1`, selected by `control.active.json` |
| Active B200 working set, one per runtime | `$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN` |
| Notebooks and merged results | Local repository |

- Set `JEVPFN_PROJECT_ROOT` (or `VSC_PROJECT_ROOT`) to `/lustre1/project/stg_00211`; the
  resolver appends `JEVPFN`. `JEVPFN_STAGING_ROOT` explicitly overrides it for a working
  set. Launchers pass the root as an argument and `--export=ALL,JEVPFN_DURABLE_ROOT=...`;
  direct `sbatch` calls must do the same. Large output never falls back to DATA.
- GPU work runs on **Mindwell B200**; wICE CPU jobs stage, download, build and count on
  Lustre, then copy and verify a native GPFS working set (no GPFS project allocation is
  needed). B200 jobs never stream bulk data from cross-mounted Lustre. A100 (including
  debug/MIG) is excluded; the H100 profile is only for explicit compatibility use.
- Resource profiles: 4 CPUs/28,000 MiB for small providers, 14 CPUs/112,000 MiB for
  Perplexity, SII, Kev and Decision3. A 48-hour allocation is a resumable segment.
- Working sets publish committed SQLite data to Lustre every 600 s and on exit, including
  failures. Publication merges compatible records, never replaces a complete durable
  cache with a partial one, and stops on conflicts. Predictor staging restores and unions
  durable evaluation stores, reusing completed folds.
- Scratch is not a backup: unused GPFS files are purged after 30 days and a killed node
  loses work since the last publication. Keep an independent downloaded copy of features.
- At most two runtime working sets are resident, within a 200 GiB application budget,
  500 GiB personal allocation and 20 GiB headroom (`config/3. experiment_0/scheduling.yaml`).
  Staging checks other projects' usage; check `myquota` for bytes and files. Free
  filesystem space is not quota. Never delete another project's files.

Infrastructure references: VSC Library `3bcdf55e576d10161d6189b40ac39380bbba9eca`
(hardware review, 23 October) and `1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87` (storage and
Slurm recheck, 27 October); see its `guides/SLURM.md`, `STORAGE.md`, `SOFTWARE.md`, `CLUSTERS.md`.

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
  preflight uses wICE batch with three cores at the 3,400 MiB/core default.
- **B200 images.** Separate `containers/blackwell-cu130/<runtime>/` images keep publisher
  pins with the exact cu130 Torch/torchvision wheels (the cu128 index lacks them; CUDA 13
  needs driver branch ≥ 580). Startup checks CUDA 13.0, native `sm_100`, BF16 and the B200
  device. These checks do not establish model correctness; never substitute newer packages
  to bypass a failed pin.
- **Model-image probes** inside a CPU-controller job use an overlapping host Slurm step;
  rebuildable compiler caches use a per-job scratch leaf removed on exit.
- **Predictor runtimes.** Stage the assets in `config/4. experiment_1/models.yaml`
  (`python -m src.cluster.predictor_assets --execute` writes checksummed offline Hugging
  Face snapshots). AG-MM has its own environment; TabSTAR v1.1 has the `tabstar-v1p1` image.
- Question ZIPs install with `input_install.slurm`, which refuses other JEVPFN consumers.

## 4. Pipeline

### 4.1 Reset and upload (future authorized campaign only)

The completed 0A must not be restarted. A new campaign needs separate authorization, all
JEVPFN jobs and writers stopped, failures understood and the same reviewed source on both
machines. On VSC (`JEVPFN_PYTHON` is the controller interpreter):

```bash
"${JEVPFN_PYTHON:?}" -m src.utils.output_cleanup --tier both --include-archives
"$JEVPFN_PYTHON" -m src.utils.output_cleanup --tier both --include-archives --apply --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.setup --cluster wice
mkdir -p /lustre1/project/stg_00211/JEVPFN/uploads
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

The upload holds the 18 question stores and metadata, never credentials or weights.
Install and freeze on VSC; installation validates every member, hash and row link before
an atomic replacement, and freeze compares source and configuration (not notebook output):

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.input_transfer install /lustre1/project/stg_00211/JEVPFN/uploads/decision_inputs_v3.zip --writers-stopped
"$JEVPFN_PYTHON" -m src.cluster.freeze --record --expected-file /lustre1/project/stg_00211/JEVPFN/uploads/frozen_revision.json
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

Enabled switches: `accepted_before_extraction_manifests` and `full_feature_creation_enabled`
in `config/3. experiment_0/self_hosted.yaml`, and `config/2. feature_creation/text_embeddings.yaml`
(recipes in `embedding_models.yaml`). From the frozen checkout:

```bash
mkdir -p "output_JEVPFN/2. feature_creation/logs"
python -m src.cluster.feature_campaign
python -u -m src.cluster.feature_campaign --execute --watch
```

The graph has 223 steps: 198 decision × dataset B200 jobs, three full encoder jobs and
eleven stage/release pairs, for 396 decision blocks and 54 embedding stores. Each runtime
runs at most two dataset jobs and at most two working sets are resident. It launches no
Jev call or predictor. Exact duplicates reuse saved outputs. The launcher resumes
recorded job IDs; a step left `starting` without a job ID is ambiguous and must be
checked in Slurm accounting before any resubmission. Never run two campaigns for one
provider. Review publication receipts before downloading or marking a dataset complete;
completed extraction still needs artifact validation and is not 0B.

### 4.5 Jev (local, owner-run)

`secrets/typesafe.local.json` (Git-ignored) is loaded automatically; never upload it or
print the key. An explicit `TYPESAFE_API_KEY` overrides it; remove stale overrides.

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

The first-100-source-rows validation (1,800 rows, 7,643 requests, USD 1 cumulative cap,
validation-only artifacts excluded from benchmark inventory) is complete:

```powershell
.\.venv\Scripts\python.exe -m src.jev.prefix_pilot
.\.venv\Scripts\python.exe -m src.jev.prefix_pilot --execute
if ($LASTEXITCODE -ne 0) { throw 'Prefix pilot stopped; inspect its receipt before resuming.' }
```

Full Jev needs a reviewed `jev_pilot` acceptance in the private config (checksummed
reports, `settings_sha256`, `input_sets`, `context_envelope_sha256`) and a reviewed
cumulative spending/throughput policy, then
`python -m src.jev.hosted --config feature_creation/jev.local --execute --full`.
References checked 30 October: [models](https://docs.typesafe.ai/models),
[request errors](https://docs.typesafe.ai/api/errors).

To add local Jev responses to VSC, upload closed SQLite snapshots to PROJECT
`uploads/jev_features/` (never over `data/jev_cache/`) and merge them in an idle CPU job:

```bash
for source in /lustre1/project/stg_00211/JEVPFN/uploads/jev_features/*.sqlite3; do
    test -f "$source" || { echo 'No uploaded Jev cache snapshots found.' >&2; exit 1; }
    "${JEVPFN_PYTHON:?}" -m src.jev.merge "$source" "/lustre1/project/stg_00211/JEVPFN/data/jev_cache/${source##*/}"
done
```

The merge validates identities and preserves other providers. Restage predictor working sets afterwards.

### 4.6 Experiment 0B

Fill `feature_artifacts` in `config/3. experiment_0/debug.yaml` and the benchmark
`feature_selectors` with immutable selections; finalize predictor settings and PCA
recipes. 0B validates row IDs, source hashes, provider/encoder coverage, transfer/resume
and preprocessing boundaries:

```bash
sbatch scripts/slurm/experiment_0_after.slurm
sbatch scripts/slurm/experiment_0_predictor.slurm tabpfn_3_5 tabpfn_3_5
```

GPU predictor contracts need their staged image, checkpoints and working set; CatBoost and
linear use `experiment_0_predictor_cpu.slurm MODEL ENV_PREFIX` on wICE. All nine
predictors need real binary, multiclass, 30-class, regression, PCA-width, native-text and
restart contracts, and the separate `full_fold_profile_enabled` test must show that all
rows fit without sampling. `predictor_execution_enabled` permits only bounded fixtures.
No training runs on login nodes.

### 4.7 PCA and benchmark

Prepare fold-specific PCA on CPU (repeat for 18 datasets × three encoders), then restage
the predictor:

```bash
mkdir -p "output_JEVPFN/5. experiment_2/logs"
sbatch scripts/slurm/prepare_pca.slurm REG_TEXT_ROTTEN_TOMATOES e5_small_v2
```

After reviewed 0B acceptance (`stage: after_extraction`, `accepted: true`, `reviewed_by`,
checksummed evidence and the `evaluation_identity` from the 0B reports), set only
`experiment_0_acceptance` and the experiment's `enabled` switch. Changing the tested
pipeline requires a new 0B acceptance; `python -m src.evaluation.acceptance` prints the
current identity. Write an array plan (it prints, but does not submit, the command):

```bash
python -m src.cluster.benchmark_jobs --phase experiment_2 --model tabpfn_3_5 --concurrency 8 --write
```

Each task owns one dataset × model × representation and resumes its five folds from
checksummed records. GPU models default to B200, CPU models to wICE.

## 5. Working sets and storage maintenance

A stage holds a working-set slot from submission until its consumers are terminal and its
release succeeds. Release runs after failed consumers too: it publishes committed results,
then removes only copied inputs, checkpoints, images and token evidence. It keeps response
queues, embeddings, PCA, predictions, diagnostics, durable assets and DATA logs; a
publication conflict keeps the inputs and blocks further staging. Manual release, after
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
- Legacy GPU image manifests are part of the frozen runtime identity; keep them until an
  idle stage boundary confirms no rollback need.
- Build scratch is temporary. The build script does not retire successful build leaves
  itself; clean only verified owned leaves after the owner reviews the plan.

## 6. Utilization and transfer

GPU telemetry samples the allocated GPUs every ten seconds (utilization, VRAM, power).
Timings separate loading, inference, cache reuse, fit and prediction; inspect them before
changing batch sizes or resources.

After all writers stop and failed scratch jobs are published, export on VSC and import
locally (substitute the printed bundle name):

```bash
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
"${JEVPFN_PYTHON:?}" -m src.cluster.transfer export
```

```powershell
scp 'vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/JEVPFN/output_JEVPFN/transfers/EXACT_BUNDLE.zip' "$env:USERPROFILE\Downloads\"
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
