# {{PROJECT_NAME}} on the KU Leuven VSC

How to run this project on the cluster. The authoritative cluster documentation is in the
`tfm-library/` submodule — read it there and keep this file as *this project's* answer to it, not a
copy. Fill in the `TODO`s the first time you submit.

**Log every run** in the Runs table of [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md): one row with the
config, the outcome and the headline number. That table is what stops a configuration being
resubmitted months after it already failed.

## The machines

| Host | Role |
|---|---|
| **Genius** | login. SSH lands here — editing, submitting and small tests only, never a run. |
| **wICE** | general-purpose compute partitions |
| **Mindwell** | GPU partitions |

```bash
ssh vsc<number>@login.hpc.kuleuven.be
```

Compute nodes have **no outbound internet**. Anything that downloads — datasets, weights, a pip
package — happens on a login node first. A job that fetches at runtime works locally and hangs
here.

## Storage — two tiers, one resolver

On **both** tiers everything lives inside a folder named after the project. Never build these
paths by hand: `src/utils/paths.py` is the only module that constructs them, and it collapses both
tiers to the repository root off-cluster, so the same code runs on a laptop unconfigured.

| tier | path | holds | quota |
|---|---|---|---|
| **project storage** | `/lustre1/project/stg_00211/{{PROJECT_NAME}}/` | datasets, checkpoints, caches, **`output/results/`** | large, **low inodes** |
| **personal data** | `$VSC_DATA/{{PROJECT_NAME}}/` | the repository, and the rest of `output/` | 75 GiB |
| scratch | `$VSC_SCRATCH/` | working scratch only | **purged after 30 days** |

- **Both tiers are backed up.** They differ in size and in convenience: you can browse `$VSC_DATA`
  directly, while anything on project storage has to be pulled down locally first (PowerShell,
  `scp`/`rsync`) before you can look at it. So the big, rarely-read things go to project storage and
  everything you actually want to open stays on `$VSC_DATA`.
- `$VSC_DATA` is 75 GiB. One forgotten checkpoint directory fills it, and then every job that
  writes a log fails too.
- Project storage has a **low inode budget** — few big files, not a hundred thousand small ones.
  Per-step records therefore go to `$VSC_DATA`.
- Scratch's purge is on **access** time, and `mv` and timestamp-preserving `rsync` do **not** count
  as an access, so freshly staged data can vanish almost immediately. Copy, then
  `paths.touch_tree()`.
- Project storage is on **Lustre**: read a checkpoint from it once at job start, never stream from
  it in a training loop.

Override the big-file tier without editing code — how the tests exercise it, and how to put data
on an external drive locally:

```bash
export {{PROJECT_UPPER}}_STAGING_ROOT=/some/other/place
```

## Environment

```bash
module load Python/3.11.3-GCCcore-12.3.0      # TODO confirm against the current module list
python -m venv $VSC_DATA/{{PROJECT_NAME}}/.venv
source $VSC_DATA/{{PROJECT_NAME}}/.venv/bin/activate
pip install -e ".[dev]"
```

The venv lives on `$VSC_DATA`: it must survive the scratch purge and be backed up, and it is small.

## Submitting

```bash
sbatch scripts/slurm/job.slurm
squeue --me
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>
```

| Knob | Where | TODO |
|---|---|---|
| partition | `#SBATCH --partition` | which partition this project uses |
| walltime | `#SBATCH --time` | the limit on that partition |
| credits | `#SBATCH --account` | the credit account to charge |
| GPUs | `#SBATCH --gpus-per-node` | what this project actually needs |

When the job comes back, add its row to the Runs table in
[`AGENTS_MEMORY.md`](AGENTS_MEMORY.md) — `done` / `walltime` / `OOM` / `crashed` / `diverged`, plus
one line of notes.

## Outliving the walltime

Any run that can exceed the limit must be **resumable** — a job killed at the walltime is
otherwise a total loss.

1. Checkpoint on a fixed interval to project storage, plus a small `state.json` on `$VSC_DATA`
   naming the latest complete one. Write the pointer **last**, so a job killed mid-write points at
   the previous complete checkpoint rather than a truncated one.
2. On start, read the pointer and resume. A fresh run and a resumed run take the **same code
   path** — a resume path only exercised after a crash is a resume path that does not work.
3. Requeue rather than resubmitting by hand: `#SBATCH --signal=B:USR1@300` gives SIGUSR1 five
   minutes before the kill. Trap it, checkpoint, then `scontrol requeue $SLURM_JOB_ID`.
4. Make each step idempotent: re-running a completed step overwrites its own output rather than
   appending, so a requeue never doubles a result file.

## Getting results back, and cleaning up

Everything generated is under `output/` — one directory to copy, which is the point of that rule.

```bash
rsync -av vsc<number>@login.hpc.kuleuven.be:$VSC_DATA/{{PROJECT_NAME}}/output/ ./output/
```

`output/results/` is on project storage, so it is a second copy from
`/lustre1/project/stg_00211/{{PROJECT_NAME}}/output/results/`.

```bash
python -m src.utils.clean_run                       # list what the previous run left
python -m src.utils.clean_run --clean               # wipe output/ on both tiers
python -m src.utils.clean_run --clean --processed    # ...and the data/processed cache
```

One invocation covers `$VSC_DATA` and project storage, and it can never remove `data/raw/`,
`checkpoints/`, or `tfm-library/` — it only ever looks inside `output/`.
