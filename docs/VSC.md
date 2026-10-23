# VSC operations

Shared infrastructure reference: [VSC Library](../vsc-library/README.md). The parent Git commit pins its revision; inspect it with `git submodule status -- vsc-library`.
Its [official repository snapshot](<../vsc-library/VSC Repository.txt>),
[guides](../vsc-library/guides/) and [Slurm templates](<../vsc-library/slurm templates/README.md>)
are read-only. Keep this project's actual paths, environment, resource choices,
submission/resume steps and run evidence in this file or the project's own run logs.
Check [upstream review flags](../vsc-library/UPSTREAM_CHANGES.md) and the
[live Leuven status](https://status.vscentrum.be/tier2_leuven.html) before applying a changed policy.

Use the Open OnDemand login shell for setup/downloads and Slurm allocations for computation.
Notebook analysis always runs locally. Current run evidence is in [AGENTS_MEMORY](AGENTS_MEMORY.md);
the research contract is in [README](../README.md). These instructions follow the read-only
[VSC guide](<../vsc-library/VSC Repository.txt>). Record the checked-out library revision
with `git -C vsc-library rev-parse HEAD`; operational evidence is date-specific.

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
`python -m src.utils.check_libraries` verifies that all three library gitlinks are present
in HEAD and their checkouts match. A directory and `.gitmodules` entry alone are insufficient.
The CPU job and local 0A run this read-only check before proceeding.
Rerun checks affected by a change; unrelated edits do not invalidate matching token/runtime
evidence or require resetting output and reinstalling inputs.

| Tier under JEVPFN | Contents |
|---|---|
| DATA: `$VSC_DATA/JEVPFN/` | Checkout, small CPU environment, compact logs/manifests. |
| PROJECT: `$JEVPFN_STAGING_ROOT/JEVPFN/` | Raw/prepared inputs, permanent dataset caches, weights, CUDA environments and package caches. |
| DATA `output_JEVPFN/<number>. phase/` | Operational logs, compact completion indexes and manifests. |
| PROJECT `output_JEVPFN/<number>. phase/` | Token-count databases, predictions, feature matrices, full metric records and other large results. |
| Node scratch | Disposable compiler/runtime caches; never the only saved result. |

Predictions/full metrics/fold completion now share one PROJECT
`<phase>/results/evaluation/<dataset>.sqlite3`. Each prediction table is stored as a
checksummed Parquet payload inside it. DATA holds one small dataset locator and shared
logs. Import merges immutable records transactionally, rejecting conflicting identities;
legacy individual files remain readable. The bounded fold-lock pool has at most 4,096
entries per phase. This avoids the previous six-figure file count for the planned benchmark.

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

## GPU routing and the native-storage prerequisite

For subsequent GPU testing and extraction, prefer Mindwell `gpu_b200` once its storage
and runtime checks pass. The owner reports markedly shorter B200 queues than H100 queues;
that is an operational observation, not a guaranteed future start time or measured speedup.
B200 supplies 192 GiB VRAM versus H100's 80 GiB, allowing larger batches where the native
model implementation supports them. Use one GPU per independent model/dataset task first;
do not request several GPUs for a serial inference loop. Keep CPU token preparation,
downloads, PCA preparation and image builds outside GPU allocations, and use the existing
telemetry/timing records to choose batches after probability-parity checks.

The 23 October owner diagnostic established `VSC_PROJECT_GPFS1=/gpfs1/project` and
`VSC_SCRATCH_GPFS1=/gpfs1/scratch/383/vsc38338`. **The former is the general root, not an
allocation.** `/gpfs1/project/stg_00211` does not exist. `myquota` lists only the Lustre
project allocation, plus personal GPFS scratch. Therefore the B200 profile's allocation
remains unset; the current setup graph is still wICE-only. Do not simply override its
GPU partition to Mindwell while retaining Lustre paths or cross-cluster dependencies.

The durable solution is to request a native GPFS project allocation through KU Leuven
HPC support (Open OnDemand Help), stating the existing Lustre allocation, B200 use,
required checkpoint/active-result capacity and file quota. Alternatively, a separately
reviewed workflow can stage a bounded working set on GPFS scratch, keep authoritative
assets on Lustre, and checkpoint/transfer durable results back. Scratch has a 30-day
cleanup policy and is not a backup or an interchangeable permanent PROJECT allocation.
That two-root scratch/mirror workflow is **not** implemented by the current project-root
override. Avoid copying whole environments or all other projects into scratch.

Containers currently install CUDA 12.6 PyTorch wheels. Do not assume these work on B200:
Blackwell needs a compatible CUDA build and actual application kernels, not merely an
updated host driver. The official PyTorch Blackwell release documents CUDA 12.8 builds.
Stage/build an isolated compatible runtime at the native destination, record its hash,
then run the exact model contracts and bounded pilot there before selecting B200 for
production. Preserve the already tested H100 images; CUDA/runtime changes alter cache
identities. No running H100 job or active checkout is changed by this recommendation.

Use these read-only commands to compare current scheduling conditions:

```bash
sinfo --clusters=mindwell -p gpu_b200 -o "%P %a %l %D %t %G"
squeue --clusters=mindwell --me -o "%.18i %.12P %.10T %.10l %.35R"
squeue --clusters=wice --me -p gpu_h100 -o "%.18i %.12P %.10T %.10l %.35R"
```

References: VSC Library `guides/{CLUSTERS,STORAGE,SOFTWARE,SLURM}.md`, pinned
`3bcdf55e576d10161d6189b40ac39380bbba9eca`, upstream `1a0cf04a5aab107021d98c563351f912a3caf062`;
[Mindwell](https://docs.vscentrum.be/leuven/mindwell_quick_start.html),
[native filesystems](https://docs.vscentrum.be/leuven/tier2_hardware/parallel_filesystems.html),
[PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

## CPU environment and input transfer

### Set up the eleven-extractor roster

The selected roster now includes Jev locally and ten open-weight VSC extractors.
The additions are `h2o`, `sii`, `kev`, `decision3` and `nimble`; they have no prior
VSC inference evidence. Existing weights and sealed containers are reusable. Do not
reinstall environments merely because project source changed.

After the owner publishes the tested code, update an idle checkout. With the existing
CPU environment activated and `JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211`:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.setup --cluster wice
myquota
"$JEVPFN_PYTHON" -m src.cluster.setup_extractors --all
"$JEVPFN_PYTHON" -m src.cluster.setup_extractors --all --execute
```

The first setup-extractors command previews. Execution submits a persistent Slurm
dependency graph, so disconnecting Open OnDemand does not lose the remaining steps.
CPU jobs download immutable weights/source, build only missing containers in scratch,
and count every prepared input. At most two stage chains and two build chains run
concurrently. Existing verified sealed images are reused. Each H100 contract job waits
for CPU/storage checks, its image and its exact counts; its bounded real-dataset pilot
waits for that contract. This command does not enable full extraction or award acceptance.
Use repeatable `--provider NAME` for a selected subset, or omit options for the five additions.
All three embedding encoders are separately covered by the final 0A campaign.

Submission IDs/dependencies are in DATA
`output_JEVPFN/3. experiment_0/manifests/container_setup/open_models.json`.
Repeat the same command after a terminal disconnect to resume submission without
duplicating jobs. A failed prerequisite leaves dependent jobs blocked; inspect their
logs before cancelling/replanning. Ambiguous submissions stop for scheduler review,
rather than guessing whether a duplicate is safe. Changes of source/selection require
reviewing the old receipt and waiting for its jobs, not silent replacement.

GPU contracts request 30 minutes on wICE H100; the dataset pilot requests one hour.
The 27B runtimes request eight CPU cores and 112,000 MiB host RAM; smaller models
request four cores and 28,000 MiB. These are initial feasibility allocations, not measured
optima. CPU token IDs for the five additions are compressed in each provider's PROJECT
token-count SQLite database and reused by inference. The checkpoint hash index is
also a single PROJECT SQLite file; Linux reuses hashes only while file metadata matches.
Windows always rehashes because its creation time is not a reliable change indicator.
Runtime code/calibration/option descriptions remain part of response-cache identities.

H2O requires six publisher probability anchors during loading, recorded separately
from its bounded diagnostic request budget. They reject an incorrect attention/readout
implementation. The new runtimes initially use serial inference until measured batch
parity and memory tests justify a change. Do not infer 100% utilization from requesting
a GPU. Host telemetry records actual allocation GPU load/memory, power and errors.
No A100 route is allowed. Mindwell requires an actual native GPFS allocation and a
Blackwell-compatible tested runtime before using its configured B200 profile.

To inspect only jobs from this setup graph after submission:

```bash
jevpfn_ids=$("${JEVPFN_PYTHON:?}" -c 'import json; from src.utils import paths; p=paths.manifests_dir("experiment_0")/"container_setup/open_models.json"; print(",".join(s["job_id"] for s in json.loads(p.read_text())["steps"] if s.get("job_id")))')
sacct --clusters=wice -j "$jevpfn_ids" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
squeue --clusters=wice --me -o "%.18i %.16P %.10T %.10l %.35R"
```

No raw/prepared/paid feature data is removed by setup. Check group quota before adding
several 27B checkpoints: containerization reduces file counts, not weight byte size.

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

The runner submits independent ready jobs concurrently: CPU, storage and token checks
start independently; encoders follow CPU validation; decision contracts follow CPU/storage/
token validation; each decision pilot also requires its own contract. Scheduling uses regular
H100 by default, with explicit routes in `experiment_0/scheduling.yaml`. All A100 routes,
including debug and interactive GPU slices, are excluded by owner decision.
`--watch` polls accounting once per minute, performs no inference on the login node, and
submits newly ready jobs only after their prerequisites reach COMPLETED/0:0. A failed/uncertain submission stops for
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

The three frozen encoders share the Clef Torch/Transformers container; their weights and
cache namespaces remain separate. Stage weights from the connected CPU setup environment:

```bash
for encoder in e5_small_v2 e5_large_v2 qwen3_embedding_0_6b; do
    "${JEVPFN_PYTHON:?}" -m src.data.text_embeddings --encoder "$encoder" --stage || break
done
```

After Clef's container works, test each encoder in a GPU allocation:

```bash
for encoder in e5_small_v2 e5_large_v2 qwen3_embedding_0_6b; do
    sbatch --clusters=wice scripts/slurm/experiment_0_embeddings.slurm cloudflare /lustre1/project/stg_00211 "$encoder" || break
done
```

These bounded tests cover multiple chunks, missing text, reuse, unit norms and resume.
They also verify prepared IDs/masks against the publisher tokenizer. Chunk preparation uses
the backend special-token processor, not the removed Transformers 5 `prepare_for_model`
method. Qwen adds end-of-text ID 151643 even though its configured EOS ID is 151645;
compare against the actual publisher tokenization, never a guessed EOS assertion.
They save fixture stores to PROJECT Experiment 0 and compact reports/logs to DATA.
No benchmark features are created. After acceptance, enable the shared `text_embeddings.yaml`
execution switch and run `python -m src.data.text_embeddings --encoder ENCODER --execute`
inside a compute allocation using this container, or submit
`sbatch scripts/slurm/embedding_creation.slurm ENCODER` (optional later arguments:
container/environment, PROJECT root, dataset ID). This job uses the regular H100 queue
and resumes completed embedding batches. Small E5 retains the existing
`data/text_embeddings/<dataset>.sqlite3` path; other encoders use an additional encoder
subfolder. All three caches reside on PROJECT; logs use DATA feature-creation logs.
PCA is fitted later on CPU for each exact CV training partition, never in these permanent embedding stores.

## Compact decision runtimes

First inspect only this project's runtimes:

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.containers inventory
myquota
```

The inventory counts entries, not shared quota or other projects' use. Never remove another
project's environment. Package caches and unpacked build trees use scratch; only sealed
images/wrappers belong in PROJECT containers. New builds test index connectivity on the
build node before installation; a network/index failure stops early and requires inspection.

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

`activate_runtime.sh` verifies an image using the native CPU interpreter, then removes
the full inherited Conda activation stack before executing its Python wrapper. Clearing
only CONDA_PREFIX while retaining CONDA_SHLVL breaks newly created images at startup.
Slurm, CUDA, PROJECT paths and the verified container identity remain available. This
launcher correction does not require rebuilding a sealed image or downloading weights.

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
After the new Liquid image passes, its obsolete partial native installation can also be
removed safely with `containers retire liquid --allow-replacement` (preview), then the same
command with `--apply --writers-stopped`. This requires a verified image and matching GPU
evidence. The ordinary `retire` command cannot remove an unrelated or untested installation.

All eight predictor names can also be packed in `wrap` mode once their independently reviewed
environments exist. For example, use the same build job with `tabpfn_3_5 wrap`, then run
`experiment_0_predictor.slurm tabpfn_3_5 /lustre1/project/stg_00211/JEVPFN/containers/tabpfn_3_5`.
Retirement requires its exact image to pass predictor contracts. Automatic new predictor
installations are deliberately unavailable until their package/checkpoint configurations are
reviewed in 0B. E5 already reuses Clef's container; the small CPU environment stays on DATA.
Perplexity GPU contracts/pilots use `--mem=112000M`. Builds prefer shared Lustre scratch on
wICE, so failed-build work survives the allocation. An empty leftover destination can be
retried; nonempty or linked destinations are refused for manual review. Keep the Slurm log
until the failure is understood; do not reset output while runtime setup is still failing.
The container-build script resolves `default_config/config.yaml` from the installed wrapper;
KU Leuven does not necessarily export `CW_GLOBAL_YAML` from the module itself.

New builds use `container_conda_pre_install.sh` to override inherited Conda `envs_dirs`
with the image's own `miniforge/envs`, and patch a private copy of the checked 0.4.2
Conda template to create/activate by absolute image prefix. The envs-dir override alone
was insufficient in Liquid job 62378847; the pre-pip assertion correctly stopped it. The post-install script verifies `sys.prefix`
before pip runs. Without this, the 21 October Liquid build installed into personal
DATA `conda_envs/env1` and produced an image without `bin/python`. That image was not
sealed and must not be used. The fix checks exact template anchors and refuses an unknown site template;
it does not change working Clef/Intern containers or the user's global Conda config.
Liquid build 62383996 subsequently sealed successfully on 21 October; do not rebuild
that sealed image. The following commands are recovery instructions only for a future
unsealed build failure, with no active JEVPFN job (they preserve weights and other runtimes):

```bash
"${JEVPFN_PYTHON:?}" -m src.cluster.containers discard-unsealed liquid
"$JEVPFN_PYTHON" -m src.cluster.containers discard-unsealed liquid --apply --writers-stopped
sbatch --wait scripts/slurm/experiment_0_container_build.slurm liquid new
sbatch --wait scripts/slurm/experiment_0_decision_contracts.slurm liquid
```

The cleanup refuses sealed containers and active JEVPFN allocations. The misplaced
`$VSC_DATA/conda_envs/env1` is outside the project environment folder and is deliberately
not removed by this helper. Inspect its Conda history/size before separately deleting
it; never remove `$VSC_DATA/conda_envs/JEVPFN`, the working CPU environment.
The following narrowly scoped removal refuses active JEVPFN jobs, symlinks and a Conda
history without the failed Liquid build path. If the history check fails, inspect the
folder instead of weakening the guard. It does not require the failed image to work:

```bash
(
set -euo pipefail
cd "$VSC_DATA/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -c 'from pathlib import Path; import os; from src.cluster.maintenance import require_idle_vsc,reject_links; require_idle_vsc(); root=Path(os.environ["VSC_DATA"])/"conda_envs"; p=reject_links(root/"env1"); assert p.resolve()==root.resolve()/"env1"; h=(p/"conda-meta/history").read_text(); assert "jevpfn-container-liquid-" in h, "History does not identify the failed Liquid build; inspect before deleting"; print("Verified misplaced Liquid environment:",p)'
conda env remove --prefix "$VSC_DATA/conda_envs/env1" --yes
myquota
)
```

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

Submit only one full token audit at a time. The coordinator takes a PROJECT lock before
loading tokenizers; each count database also has a writer lock, and DATA report merges
are serialized. A duplicate run stops promptly without deleting or replacing counts.
The OS releases these locks when the process exits; do not remove lock files manually.
If older overlapping jobs failed with `UNIQUE constraint failed: counts.dataset_id,
counts.input_no`, wait for both to exit, synchronize the fixed code and submit one audit.
It reuses matching committed counts and checks completeness before republishing reports.
Preserve the databases, prepared inputs and sealed containers during this recovery.

After Clef completes successfully, repeat for `intern`, `liquid` and `strands`.
Perplexity uses `sbatch --mem=112000M scripts/slurm/experiment_0_decision_contracts.slurm perplexity`.
The 0A coordinator handles dependency-aware concurrent submission. For individual setup jobs, submit without
`--wait` and inspect the recorded ID with `sacct` before the next command; a disconnected
login terminal is not proof that the submitted job was cancelled.
The provider, optional environment prefix and optional project root are script arguments.
For example, append `/lustre1/project/stg_00211/JEVPFN/environments/intern /lustre1/project/stg_00211`
to select both paths explicitly. KU Leuven starts batch jobs with a clean environment; exporting
variables in the submitting shell is insufficient. CPU/token/storage scripts select the CPU
environment themselves and accept an optional project-root argument. The prepared job
environment is explicitly propagated to `srun` workers. See the guide’s Environment propagation
section and [Slurm argument/export semantics](https://slurm.schedmd.com/sbatch.html).

The job uses one full H100, four CPUs, 28,000 MiB host RAM and one hour in `gpu_h100`,
with account `lp_verbekelab`. Independent provider jobs can run concurrently, subject to
site/account limits. Verify account/partition access and each runtime on the actual cluster.

The contract job tests binary, three-class and nine-level regression fixtures, a 30-class
vector, missing/duplicate/Unicode input, overflow rejection, cache reopen/reuse, original-row
alignment, and saving/reloading both feature representations. It then evaluates first/longest
unique inputs from each selected dataset. Seven additional known-answer probes check binary
polarity, permuted class ordering and score endpoints/zero. With robustness and batch-parity
checks and 13 real-text sensitivity probes, at most 106 input evaluations are planned, with a hard ceiling
of 110. Runtime token-count agreement, load/inference time, peak GPU memory and the complete
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
`--approve --reviewed-by Andreas --quota-verified --backup-verified --throughput-reviewed --robustness-reviewed`.
When several successful runs exist, pass `--campaign` with the exact completed `0A_vsc`
receipt. CPU/storage/GPU/pilot evidence must match current source/configuration and runtime
fingerprints, and the selected campaign's job IDs. Token evidence is reusable only when its
input hashes and exact tokenizer identity match. Do not delete evidence to resolve ambiguity.
This explicitly signs `3. experiment_0/manifests/before_extraction/accepted_<provider>.json`;
it never enables extraction. Only pass those flags after doing the corresponding review.
Set each path in `self_hosted.yaml`'s `accepted_before_extraction_manifests` map and enable
its separate full-extraction switch between stages. Tests alone never sign acceptance.
After acceptance, `scripts/slurm/feature_creation.slurm` runs full extraction on regular
`gpu_h100`; its 24-hour allocation is a resumable segment, not a runtime prediction. Create
`output_JEVPFN/2. feature_creation/logs` before submission, then use
`sbatch scripts/slurm/feature_creation.slurm cloudflare` (or its accepted provider). Provider jobs rotate datasets and
wait for busy files. `--continue-on-error` is an explicit diagnostic mode with durable failed
request records; incomplete datasets cannot be materialized or accepted. Repair the cause
and use `--retry-failed` to retry those inputs. No silent substitution by missing values.

Small-model jobs start at 28,000 MiB host RAM; Perplexity uses 112,000 MiB. These are
initial profiles, not measured minima. The pinned H100 profile permits at most 187,200 MiB
host RAM and 16 CPU cores per GPU; memory requests can bill extra CPU cores. H100 has
80 GiB device memory. A100 is excluded. Mindwell B200 remains disabled until its native
GPFS allocation is configured and the runtime/checkpoint passes a bounded GPU test there.
Clef and Perplexity use batches of two; 0A checks their probabilities against serial execution
with an absolute tolerance of 0.001. Liquid, Strands and Intern remain serial until a native
batch interface is independently verified. All retain exact context checks, row ordering and
cache reuse. In explicit `--continue-on-error` mode, a failed batch is retried once per
input serially so its valid neighbour can complete. Storage failures stop immediately.
This is open-weight diagnostic recovery, not a paid API retry. Runtime settings include
batch size. Larger batches require a new reviewed profile and acceptance.
0A additionally measures repetition, paraphrase, irrelevant-context and embedded-instruction
sensitivity on synthetic inputs. These diagnostics are separate from frozen dataset questions.
Inspect probability shifts before approving; do not automatically relax a failing tolerance.
For every provider, estimate full GPU hours from the measured pilot rather than assuming
that a short successful contract test establishes million-request throughput.

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

`python -m src.jev.smoke --robustness --execute` runs the separate 15-call diagnostic
for repeated inputs, paraphrases, irrelevant context and embedded instructions. Its
permanent ledger is `data/jev_cache/_control/jev_robustness/`; normal reruns reuse saved
responses. The initial attempt allowance is at most USD 0.04032 under the configured
64k billing bound and USD 0.042/M price. Review the measured probability shifts; this
diagnostic does not change the frozen dataset questions or establish account rate limits.
The local 0A coordinator includes this step automatically.

If a saved response fails decoding, `python -m src.jev.smoke --robustness --revalidate-only`
reprocesses existing responses without credentials or network access. It cannot fill missing
responses or erase unknown attempts. Jev 1.13 has returned score 7.99 alongside displayed
probabilities implying 8.00. The decoder now accepts only discrepancies mathematically
consistent with two-decimal rounding of both fields; open-weight responses remain strict.
Raw fields and the discrepancy are retained, and the feature remains the expectation
computed from the returned probabilities. TypeSafe does not guarantee this rounding rule.

After configuring the private context envelope, run
`python -m src.jev.smoke --dataset-robustness --config feature_creation/jev.local --execute`
for 13 real-text diagnostic requests (initial maximum USD 0.034944). It selects the first
prepared input of the first dataset per task, in sorted dataset order, without accessing
row targets. It compares fixed paraphrases, irrelevant context and injected instructions;
multiclass also reverses options and realigns probabilities. Regression's ordered rubric
is not reversed. Local 0A includes this when `--dataset-pilot` is selected. Report total
variation and top-option agreement; this small sample is not comprehensive robustness evidence.

The console's user menu → Settings → Organization displays the organization ID. Billing
shows credits and auto-recharge. Account RPS/TPS are at
https://console.typesafe.ai/rate-limits. On 21 October the owner confirmed 80 RPS and
100,000 input TPS; these are recorded in `throughput_review`, not inferred from public
defaults. Recheck before full extraction; increases go through `sales@typesafe.ai`.
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
The default setup uses `provider_enforced` context validation: public Jev 1.13 documentation
specifies 32k state-plus-question tokens, 64k per complete request, and HTTP 400 with
`max_tokens_exceeded` for context overflow. The local guard checks the separate 32 MiB
HTTP body ceiling; it does not pretend that byte size proves token fit. No long-context
header is sent. Reserve 64k billed tokens per attempt conservatively, retain actual usage,
and stop on overflow without truncation or an automatic retry. The older explicitly
reviewed byte-envelope mode remains available for existing private configurations. HTTP 429 has
bounded exponential backoff; 529/timeouts remain review stops because retryable does not
establish that an earlier attempt was unbilled. Raw responses received by the dispatcher
survive a subsequent feature-decoder failure.

Alternatively, keep the tracked defaults disabled and run
`python -m src.jev.setup_pilot --configure` locally. It accepts `0.042`, `$0.042` or
`USD 0.042`, reprompts invalid numbers and supplies the recorded RPS/TPS defaults. It
asks for a conservative EUR/USD conversion and evidence, then writes the Git-ignored
`config/2. feature_creation/jev.local.yaml`. It enforces the approved EUR 5 pilot, includes
all 18 dataset allocations and never enables full extraction. Context/billing policy comes
from https://docs.typesafe.ai/models and https://docs.typesafe.ai/api/errors (checked
21 October), without an invented tokenizer or byte/token ratio. The pilot uses one worker
and at least five seconds between calls, below the account ceilings. Configuration performs
no inference and never overwrites an existing private spending policy.
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

The account's 80 RPS / 100,000 TPS are ceilings, not the current dispatcher speed.
The pilot uses one worker and a five-second minimum interval. Even when its RPS setting
is raised, the current RateGate conservatively reserves 64,000 tokens for every attempt,
so TPS alone enforces at least 0.64 seconds between starts. Full extraction needs a
separate reviewed throughput configuration and spending-policy revision; do not present
the account's theoretical six-hour request floor as this implementation's runtime.

After full feature creation and validated transfer, Experiment 0B checks complete selected
artifacts and real predictor fits on all task types. Main extras are `models`, `baselines`,
`tabicl`, `limix`, `kumo`; original-text extras are `tabstar`, `contexttab`, `ag-mm`. Stage/hash their
checkpoints and any text encoders separately; choose exact assets/runtime pins in
`config/4. experiment_1/models.yaml`. Use isolated environments when dependencies conflict.
The after-extraction preflight also validates all 54 complete encoder/dataset stores against current
raw rows/column hashes and the frozen encoder recipe, without expanding dense feature
matrices. Missing/corrupt/wrong-version embeddings fail 0B rather than being skipped.

AG-MM is a standard selected reference, with a fixed medium-quality recipe and no HPO
or model ensemble. Install `.[dev,ag-mm]` in its own reviewed environment, not with
`token-audit` or `embeddings` (AG-MM 1.6.1 requires Transformers <5.15). Stage
`google/electra-base-discriminator` at an immutable reviewed revision, record every
encoder file hash in `models.ag_mm.runtime.assets` and set its local `text_checkpoint_path`.
The adapter refuses unreviewed files and model loading is offline. Intermediate training
checkpoints use a temporary fit directory; individual predictions and metrics still use
PROJECT. It can be wrapped/verified with provider name `ag_mm`, like other predictors.
Its real 0B tests and full-fold resource profile are outstanding.

The injector/factories are synthetically tested; actual full-fold resources remain pending.
ConTextTab uses one bag and the full training context rather than default row sampling;
TabSTAR's default adaptation reserves internal validation for stopping. Kumo rejects
500-column overflow. Record failures without silently reducing rows/features. Pretraining
and prediction-time input availability still need review before interpreting results.

Enable only the bounded `predictor_execution_enabled` switch in `experiment_0/debug.yaml`
for `python -m src.cluster.predictor_contracts --model MODEL --execute`. This runs actual
binary/three-class/30-class/regression fixtures, five-fold CV and no-refit restart checks.
Each main predictor tests non-text, per-column PCA32, joint PCA256/128/64/32/16/8, 1,024-dimensional vectors and
PCA32 with both decision-feature blocks.
The vectors are synthetic downstream-interface fixtures; actual encoder inference is tested in
0A and saved dataset vectors in 0B. TabSTAR/ConTextTab/AG-MM test original text alone and with joint/per-column/both decision features.
`sbatch scripts/slurm/experiment_0_predictor.slurm tabpfn_3_5 /lustre1/project/stg_00211/JEVPFN/environments/tabpfn_3_5`
supplies a regular H100 allocation once that selected predictor environment exists. Run CPU
CatBoost/linear checks with `scripts/slurm/experiment_0_predictor_cpu.slurm MODEL ENV_PREFIX`.
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

CV folds have completion records in the dataset database with record/prediction checksums. Restarts skip only
complete matching folds; interrupted attempts remain evidence. Experiment 5's statistics,
permutation and retained-view arms use `campaign --include-controls` and the explicit
`--control`/`--view` filters. `python -m src.evaluation.results --save` writes complete-arm
per-dataset metrics and paired differences to PROJECT Experiment 5, with its compact index on
DATA. Negative differences are improvements for loss metrics, not for higher-is-better metrics.
Experiment 1 selects decision arms with `--phase experiment_1`. Experiment 2 additionally
requires `--encoder`, `--pca-components 8|16|32|64|128|256` and `--pca-scope per_column|joint`;
per-column scope uses only 32; joint scope uses all six dimensions. The internal feature-set identifiers retain
`frozen_e5` for cache compatibility, but the encoder argument selects E5-small, E5-large
or Qwen. Example read-only inspection:

```bash
python -m src.evaluation.campaign --phase experiment_2 --encoder e5_large_v2 --pca-components 32 --pca-scope per_column --model catboost
python -m src.evaluation.campaign --phase experiment_3 --model tabstar
```

Primary comparisons are fixed before outcomes. Experiment 3 uses original text only.
Later winner-selected combinations live in exploratory Experiment 5; select one decision
extractor/mode and one encoder/PCA recipe with checksummed evidence and rationale. Analysis
pairs these additions with the matching Experiment 2 or 3 baseline and labels them
exploratory; confirmatory inference rejects them. Optional
`--infer` needs an approved `experiment_5/analysis` plan; no headline metric is selected
automatically. Experiment 4 analyzes explicit real diagnostic reports using
`python -m src.evaluation.prompt_ablation --report PATH --save`; it makes no new calls.

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

## Parallel execution and resource measurement

Reviewed VSC Library pin: `3bcdf55e576d10161d6189b40ac39380bbba9eca`, upstream
`1a0cf04a5aab107021d98c563351f912a3caf062` (22 October; read on 23 October). The reviewed Leuven
Slurm, hardware and parallel-filesystem guides require explicit cluster selection,
native filesystems and actual application compatibility checks. They do not promise
current queue availability or unlimited concurrent jobs.

`experiment_zero --site vsc --execute --watch` now starts CPU, storage and token checks
independently. Encoders follow CPU validation; provider contracts follow CPU/storage/tokens;
each provider pilot follows its contract. At most eight jobs are active by default.
Submission receipts survive disconnects. A failed job stops new submissions without
cancelling other jobs automatically. No repeated submission is used to improve priority.

Routes are configured in `config/3. experiment_0/scheduling.yaml`. The default is regular
wICE H100. A100 profiles are absent, and submission/runtime guards reject A100 overrides.
This policy also covers debug partitions and interactive GPU slices. Mindwell B200 requires the actual **GPFS**
group allocation in its profile, inputs/checkpoints/runtime staged there, and real B200
compatibility evidence. Its root deliberately remains unset. Never stream heavy I/O from
Lustre on Mindwell. Container wrappers can contain absolute prefixes: build/verify their
GPFS copy; copying the wrapper alone does not establish relocatability. Use verified cache
transfer/merge after separate-cluster extraction. Do not run overlapping writers against
one dataset store; provider contracts now defer busy stores as extraction already did.

Check current queue limits and availability before increasing concurrency:

```bash
sacctmgr show qos debug,interactive,long,normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30
sinfo --clusters=wice,mindwell
squeue --clusters=all --me
```

For a pending job with reason `Priority`, inspect the priority factors and estimated
start rather than treating queue age or the priority number as a queue position:

```bash
sprio --clusters=wice -j 62384605,62384606,62384607,62384608,62384610 -l
squeue --clusters=wice --start -j 62384605,62384606,62384607,62384608,62384610
sinfo --clusters=wice -p gpu_h100 -o "%P %a %l %D %t"
squeue --clusters=wice --me -p gpu_h100 -o "%.18i %.10T %.10l %.35R"
```

These IDs are the owner's 22 October diagnostic jobs; use actual pending IDs in future
runs. Leuven documents fair share and queue age as priority factors; the displayed
`Priority` reason alone cannot identify which factor dominates. A long wait is not a
Python failure. [VSC scheduling explanation](https://docs.vscentrum.be/compute/jobs/why_doesn_t_my_job_start.html).

The owner excludes A100 even when its debug queue is idle. Cancel only this project's
jobs before changing its checkout, then submit H100 tests from the updated source.
Do not cancel by username alone: that would also cancel the owner's other projects.
All project scripts use job names beginning `jevpfn`; inspect job names and IDs first.
H100 waiting times remain scheduler-dependent, and compatibility must be established for
each runtime. [wICE partitions](https://docs.vscentrum.be/leuven/wice_quick_start.html).

Realistic short walltimes, measured memory and several independent single-GPU jobs make
more schedulable work than one oversized allocation. Priority remains under Slurm/site
policy. More credits do not give priority. P100/V100 are not enabled for these modern
runtimes without compatible precision/kernels/VRAM evidence. Multi-GPU speed comes from
independent array tasks; no unimplemented distributed execution is implied.

GPU inference scripts run the telemetry parent in the native host interpreter
(`JEVPFN_MONITOR_PYTHON`) around the selected workload interpreter (`JEVPFN_PYTHON`).
This keeps driver monitoring outside the model image and avoids a nested container just
to start the workload. The host nvidia-smi path is captured during activation; only the
Slurm-allocated GPU IDs are queried. No fallback queries another user's GPU.
Scripts record a compressed telemetry stream in the phase's DATA
`logs/telemetry/` and one summary in `manifests/telemetry/`. The configurable default
sampling interval is ten seconds. It records allocated-device utilization, VRAM and
power alongside job/cluster/partition/CPU identity. Missing sampling is explicit, never
reported as 100% utilization. Failures retain the sampler executable, allocation IDs,
return code and bounded stderr/stdout; a successful model command does not imply successful
telemetry. Inspect `sampling_status` and `gpu_samples` in the summary. Slurm keeps command stdout/stderr; application reports keep
load/inference/fit/predict timing. GPU use can fall during I/O, synchronization, kernel
launches and small batches. Increase measured batch sizes only after parity/memory tests.

`src.utils.timing.TimingJournal` retains extraction measurements in one PROJECT SQLite
per run under `<phase>/results/timings/`, and a compact DATA summary under
`<phase>/manifests/timings/`. Each observation identifies its dataset and input hashes or
batch members, item count, elapsed wall seconds, failure type and CUDA synchronization.
Jev API latency includes SDK/network/provider processing; rate-gate waiting is separate.
Open-weight/embedding batch latency includes input preparation, forward processing and
output conversion; it is not a GPU-kernel-only benchmark. Loading is recorded once per
loaded runtime, and cache reuse is untimed rather than falsely reported as zero inference
latency. Batch averages are not individual-request latency. Do not sum overlapping
concurrent API durations and call them campaign walltime. Detailed measurements permit
later latency quantiles without rerunning inference.

Timing records are committed in groups of 64 to avoid a filesystem commit per call;
dataset/run summaries and failures flush pending records. A killed process can lose at
most 63 buffered measurements. Paid raw responses/accounting are persisted before timing
writes, so timing failure cannot force a new paid call. Timing journals are output, not
feature-cache identity; normal result export/import includes their consistent SQLite snapshots.
Predictor timing remains in the permanent per-dataset evaluation store, with CPU and
synchronized GPU wall timings for initialization, inner/final fit and prediction.
Library work deferred until `fit`, including preprocessing or checkpoint loading, is
included in fit time. No unsupported per-stage precision is implied.

After frozen embeddings are complete, prepare PCA on CPU **before** predictor GPU jobs:

```bash
mkdir -p "output_JEVPFN/5. experiment_2/logs"
sbatch scripts/slurm/prepare_pca.slurm REG_TEXT_ROTTEN_TOMATOES e5_small_v2
```

Repeat for the 18 selected datasets and three encoders (54 independently resumable CPU
jobs). Each job prepares all seven settings and the correct outer/inner training
partitions. Persistent numeric parameters go to PROJECT in one SQLite per dataset/encoder.
Their schema/version includes the scikit-learn version; prepare with the same version
used by the predictor runtime. Non-text preprocessing remains fold-local inside its
predictor adapter. It is not legitimate to fit it or PCA once on the entire dataset.

After 0B acceptance and explicit experiment enablement, this command **writes a plan and
prints its exact submission command; it does not submit**:

```bash
python -m src.cluster.benchmark_jobs --phase experiment_2 --model tabpfn_3_5 --profile wice_h100 --concurrency 8 --write
```

The manifest assigns one unique dataset/representation per array item, all five folds.
Only CPU resources are requested for CatBoost/linear; GPU predictors request one GPU per
item. Use `--profile wice_h100` for a separately verified runtime or `mindwell_b200` after
GPFS setup. Submit each logical array once, on one chosen profile. The per-array concurrency
is not a global quota: several arrays add their active tasks. Check MaxSubmitJobsPerUser
before submitting large arrays. `--profile-fold` routes first-fold resource tests to
Experiment 0 instead of benchmark output. Full-run gates and frozen-source checks still
apply, and no feature/row cap is introduced. Array jobs resume checksummed completed folds.

For the final reset, retain weights, containers and permanent response/embedding stores.
Use the existing preview/apply cleanup only after setup fixes and runtime tests pass, with
no active JEVPFN writers. Then rebuild questions, upload verified inputs, freeze once, rerun
0A, create features, prepare PCA, run 0B, and only then enable the benchmark. A clean output
directory is not itself evidence that the code is bug-free.
