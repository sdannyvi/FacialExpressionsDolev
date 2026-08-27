#!/bin/bash
# =============================================================================
#  FacialExpressionsDolev - batch experiment launcher (ECE-HPC / BGU)
#
#  Layout produced - log and csv side by side in one folder:
#      experiments/<EXP_GROUP>/<run_name>_<jobid>.log
#      experiments/<EXP_GROUP>/<run_name>_<jobid>.csv
#  Same stem, same job id - the log and the csv always pair up.
#
#  Run with:
#      mkdir -p experiments/generator_comparison/runs
#      sbatch zeroshot_gemma_4_31b_no_thinking.sh
#
#  TWO THINGS TO SET:
#    1. run name  -> --job-name (below, or on the sbatch command line)
#    2. exp group -> BOTH the --output line below AND the EXP_GROUP variable.
#       They must say the same thing. SLURM opens the log before bash runs, so
#       the --output path cannot use a variable - it has to be literal text.
#
#  !! The folder must exist BEFORE you submit, or SLURM cannot open the log
#  !! and the job dies with nothing to explain why:
#  !!     mkdir -p experiments/<EXP_GROUP>
# =============================================================================

#SBATCH --job-name=zero_shot_gemma_4_31b_no_thinking
#SBATCH --output=/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/gemma_4_31b/%x_%j.log
#SBATCH --partition=vilenchik_part
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45000M
#SBATCH --time=20:00:00

# No --error line on purpose: without it SLURM merges stderr into stdout, so
# prints + warnings + tracebacks all land in the ONE log file above.
# %x = job name, %j = job id. Absolute path, so the submit directory never matters.

set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH

# ----------------------------------------------------------------- FIXED SETUP
PROJECT=/truenas/home/sdolev/FacialExpressionsDolev
IMAGE=/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif
CONDA_ENV=rag
CONDA_SH=/truenas/home/sdolev/miniconda3/etc/profile.d/conda.sh

# ------------------------------------------------------------- PER-EXPERIMENT
# Experiment group = the folder under experiments/.
# MUST match the --output line above.
EXP_GROUP="generator_comparison/runs/gemma_4_31b"

# Which pipeline: zero_shot | rag
PIPELINE="zero_shot"

# Arguments for that pipeline. Use absolute paths ($PROJECT/...).
# Do NOT pass --results_path here; it is derived from the job name.
PIPELINE_ARGS=(
  --test_path    "$PROJECT/rag_thresholds/train_test_sets/public_test_50%.csv"
  --generator_id "google/gemma-4-31B-it"

  # zero_shot.py takes ONLY --test_path, --generator_id, --results_path (added
  # automatically below) and the bare --enable_thinking flag. No knowledge base, and
  # no --top_k / --dim_reduction / --prompt / --start_batch - those are rag.py only.
  #
  # --enable_thinking is a bare on/off flag: present = thinking on, absent = off.
  # Deliberately OFF for this run: it is the no-thinking half of the gemma-4-31B pair.
  # The checkpoint is registered "thinking": "optional", so omitting the flag is legal
  # and simply means no reasoning - and the results csv gets no thinking column.
  # --enable_thinking
)
# --------------------------------------------------------- END PER-EXPERIMENT

RUN_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
EXP_DIR="$PROJECT/experiments/$EXP_GROUP"
RESULTS_CSV="$EXP_DIR/${RUN_ID}.csv"
LOG_FILE="$EXP_DIR/${RUN_ID}.log"

# --------------------------------------------------------- provenance / git
# Run on the host, before entering the container (git may not exist in the image).
GIT_BRANCH=$(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git -C "$PROJECT" rev-parse HEAD)
GIT_DIRTY=$(git -C "$PROJECT" status --porcelain)

echo "=============================================================="
echo " run name    : $SLURM_JOB_NAME"
echo " exp group   : $EXP_GROUP"
echo " job id      : $SLURM_JOB_ID"
echo " started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo " node        : $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}"
echo "--------------------------------------------------------------"
echo " git branch  : $GIT_BRANCH"
echo " git commit  : $GIT_COMMIT"
if [ -n "$GIT_DIRTY" ]; then
  echo " git state   : DIRTY - the commit above does NOT match what ran:"
  echo "$GIT_DIRTY" | sed 's/^/               /'
else
  echo " git state   : clean"
fi
echo "--------------------------------------------------------------"
echo " image       : $IMAGE"
echo " conda env   : $CONDA_ENV"
echo " results csv : $RESULTS_CSV"
echo " log file    : $LOG_FILE"
echo "=============================================================="
nvidia-smi || true
echo "=============================================================="

# ------------------------------------------------------------------- run it
# The pipelines use relative imports (from ..generators), so they must be run
# as MODULES with src/ on PYTHONPATH - `python /path/to/zero_shot.py` fails.
apptainer exec --nv --bind /truenas "$IMAGE" bash -lc "
  set -eo pipefail
  source '$CONDA_SH'
  # conda's activate.d scripts read unset vars (qt-main_activate.sh line 5 reads
  # QT_XCB_GL_INTEGRATION with no default), which is fatal under -u. Relax it just for
  # the activation, then restore it so the pipeline command runs with the check on.
  # NOTE: no dollar signs in this block - it is inside a double-quoted string, so the
  # OUTER shell expands them before the container ever sees the text.
  set +u
  conda activate '$CONDA_ENV'
  set -u
  export PYTHONPATH='$PROJECT/src'
  export PYTHONUNBUFFERED=1
  cd '$PROJECT'
  exec python -u -m fer_rag.pipelines.$PIPELINE ${PIPELINE_ARGS[*]@Q} --results_path '$RESULTS_CSV'
"

echo "=============================================================="
echo " finished    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " results     : $RESULTS_CSV"
echo "=============================================================="
