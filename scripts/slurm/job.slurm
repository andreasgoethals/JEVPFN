#!/bin/bash
# ---------------------------------------------------------------------------
# {{PROJECT_NAME}} — SLURM job template.
#
# Copy this per experiment; do not parameterise one script into unreadability.
#   sbatch scripts/slurm/job.slurm
#
# THIS IS BASH ON LINUX, not PowerShell. `&&` is fine here.
#
# Fill in the TODOs from docs/VSC.md the first time you submit.
# ---------------------------------------------------------------------------
#SBATCH --job-name={{PACKAGE_NAME}}
#SBATCH --account=TODO_credit_account
#SBATCH --partition=TODO_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Uncomment for a GPU partition:
##SBATCH --gpus-per-node=1

# Logs go under output/logs/ on $VSC_DATA, NOT next to the submitting directory. WHY: the
# repository is on the backed-up tier, and after a requeue SLURM writes to a different file
# — a log the code controls is the one that survives the job that produced it.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Ask for SIGUSR1 five minutes before the walltime kill, so a long run can checkpoint and
# requeue itself instead of being lost. See docs/VSC.md § Outliving the walltime.
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT="{{PROJECT_NAME}}"
REPO="$VSC_DATA/$PROJECT"

module load Python/3.11.3-GCCcore-12.3.0   # TODO confirm against the current module list
source "$REPO/.venv/bin/activate"
cd "$REPO"

# Record where this run will write, before it writes anything. A run whose output cannot be
# found six months later is a run that did not happen.
python -c "from src.utils.paths import describe; [print(f'{k:>16}: {v}') for k, v in describe().items()]"

# --- resume-on-signal --------------------------------------------------------
# Trap USR1, let the training loop finish its checkpoint, then requeue. The job comes back
# with the same ID and resumes from the pointer it wrote.
checkpoint_and_requeue() {
    echo "$(date) received USR1 — checkpointing and requeueing job $SLURM_JOB_ID"
    # The application writes its own checkpoint on SIGUSR1; give it a moment to finish.
    sleep 30
    scontrol requeue "$SLURM_JOB_ID"
}
trap checkpoint_and_requeue USR1

# --- the actual work ---------------------------------------------------------
# TODO replace with this project's entry point. `&` plus `wait` is required, not stylistic:
# a foreground process makes bash defer the trap until it exits, so the signal would arrive
# only after the job was already killed.
srun python scripts/run_notebooks.py &
wait
