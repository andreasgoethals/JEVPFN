# {{PROJECT_NAME}} on the KU Leuven VSC

How to run this project on the cluster. The authoritative cluster documentation lives in
the literature submodule — read it there and keep this file as the *project's* answer to
it, not a copy of it:

    tfm-library/  → the VSC documentation it carries

Fill in the `TODO` markers from that documentation the first time you submit a job, and
record what you learned in [`CHANGELOG.md`](CHANGELOG.md).

---

## The machines

| Host | Role | Notes |
|---|---|---|
| **Genius** | login | SSH lands here. Login nodes are for editing, submitting and small tests only — never a training run. |
| **wICE** | compute | the general-purpose partitions |
| **Mindwell** | compute | GPU partitions |

```bash
ssh vsc<number>@login.hpc.kuleuven.be
```

Compute nodes have **no outbound internet**. Anything that downloads — datasets, model
weights, a pip package — happens on a login node first, into the storage tier below.
A job that fetches at runtime works locally and hangs on the cluster.

## Storage — two tiers, one resolver

On **both** tiers everything lives inside a folder named after the project. Never build
these paths by hand: `src/utils/paths.py` is the only module that constructs them, and it
collapses both tiers to the repository root when you are not on the cluster, so the same
code runs on a laptop with no configuration.

| tier | path | holds | backed up | quota |
|---|---|---|---|---|
| **project storage** | `/lustre1/project/stg_00211/{{PROJECT_NAME}}/` | big files: datasets, checkpoints, generated caches, **`output/results/`** | no | large, **low inode budget** |
| **personal data** | `$VSC_DATA/{{PROJECT_NAME}}/` | the repository, and the rest of `output/` — figures, logs, manifests, the two `.md` summaries | **yes** | 75 GiB |
| scratch | `$VSC_SCRATCH/` | working scratch only | no | **purged after 30 days without access** |

Consequences worth remembering:

- `$VSC_DATA` is 75 GiB. Nothing large goes there. One forgotten checkpoint directory
  fills it and every job that writes a log then fails.
- Project storage has a **low inode budget** — it wants few big files, not a hundred
  thousand small ones. Per-step metrics therefore go to `$VSC_DATA`, not here.
- Scratch's purge is based on **access** time, and `mv` and timestamp-preserving `rsync`
  do **not** count as an access. Freshly staged data can be deleted almost immediately.
  Copy, then call `paths.touch_tree()`.
- Project storage is on **Lustre**. Read a checkpoint from it once at job start; do not
  stream from it in a training loop.

Override the big-file tier without editing code — useful for a laptop with an external
drive, and how the tests exercise it:

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

The venv lives on `$VSC_DATA`, not in the repository and not on scratch: it must survive
the scratch purge and be backed up, and it is small.

## Submitting

```bash
sbatch scripts/slurm/job.slurm
squeue --me
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>
```

| Knob | Where it is set | TODO |
|---|---|---|
| partition | `#SBATCH --partition` | which partition this project uses |
| walltime | `#SBATCH --time` | the limit on that partition |
| credits | account in `#SBATCH --account` | the credit account to charge |
| GPUs | `#SBATCH --gpus-per-node` | what this project actually needs |

## Outliving the walltime

Any run that can exceed the partition's limit must be **resumable**, because a job killed
at the walltime is a total loss otherwise. The pattern:

1. Write a checkpoint on a fixed interval to project storage, plus a small `state.json`
   on `$VSC_DATA` naming the latest complete one. Write the pointer **last**, so a job
   killed mid-write leaves a pointer to the previous complete checkpoint rather than a
   truncated one.
2. On start, read the pointer and resume from it. A fresh run and a resumed run take the
   same code path — a resume path that is only exercised after a crash is a resume path
   that does not work.
3. Requeue rather than resubmit by hand:

```bash
#SBATCH --signal=B:USR1@300     # SIGUSR1 five minutes before the kill
```

   Trap it, checkpoint, then `scontrol requeue $SLURM_JOB_ID`.

4. Make the job idempotent: re-running a completed step overwrites its own output rather
   than appending to it, so a requeue never doubles a result file.

## Getting results back

Everything the code generates is under `output/`, which is the whole point of that rule —
one directory to copy.

```bash
rsync -av vsc<number>@login.hpc.kuleuven.be:$VSC_DATA/{{PROJECT_NAME}}/output/ ./output/
```

`output/results/` lives on project storage, so it is a second copy from
`/lustre1/project/stg_00211/{{PROJECT_NAME}}/output/results/`.

## Cleaning up

```bash
python scripts/clean_run.py                    # list what a previous run left, delete nothing
python scripts/clean_run.py --clean            # delete the cheap categories
```

It walks both tiers and can never remove `data/raw/`, `checkpoints/`, or `tfm-library/`.
