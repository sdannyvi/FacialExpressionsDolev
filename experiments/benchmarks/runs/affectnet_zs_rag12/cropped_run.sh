#!/bin/bash
# =============================================================================
#  FacialExpressionsDolev - experiment launcher (RunPod, single GPU, no SLURM)
#
#  Continues the AffectNet run affectnet_zs_rag12_10086 (lab server), which
#  stopped after test rows 0-1699. cropped_test.csv holds the remaining rows
#  1700-3999 (2300 samples) of data/affectnet/train_test_set/validation_set.csv.
#  All other flags are identical to affectnet_full_data_zs_rag12.sh.
#
#  Layout produced - log and csv side by side in one folder:
#      experiments/<EXP_GROUP>/<RUN_ID>.log
#      experiments/<EXP_GROUP>/<RUN_ID>.csv
#
#  Run with (inside tmux, so the run survives closing the browser):
#      tmux new -s run
#      cd /workspace/FacialExpressionsDolev
#      bash experiments/benchmarks/runs/affectnet_zs_rag12/cropped_run.sh
#      # detach: Ctrl+B, then D     re-attach: tmux attach -t run
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------- FIXED SETUP
PROJECT=/workspace/FacialExpressionsDolev
# model weights are cached on the persistent volume, not the container disk
export HF_HOME=/workspace/hf_cache
export PYTHONPATH="$PROJECT/src"
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------- PER-EXPERIMENT
# Run name: replaces SLURM's --job-name. A timestamp replaces the SLURM job id,
# so the log and the csv still pair up and a rerun never overwrites them.
RUN_NAME="affectnet_zs_rag12_cropped"
# Experiment group = the folder under experiments/.
EXP_GROUP="benchmarks/runs/affectnet_zs_rag12"

# Which pipeline: zero_shot | rag
PIPELINE="rag"

# Arguments for that pipeline. Use absolute paths ($PROJECT/...).
# Do NOT pass --results_path here; it is derived from the run name.
PIPELINE_ARGS=(
  # the REMAINING AffectNet validation (test) rows 1700-3999 that the lab-server run
  # affectnet_zs_rag12_10086 did not reach; merged with its rows 0-1699 afterwards.
  # rag.py routes every sample itself: samples whose top-1 and top-2 labels agree run
  # the original RAG, the rest run the framework
  --test_path    "$PROJECT/experiments/benchmarks/runs/affectnet_zs_rag12/cropped_test.csv"
  # frameworks run on llava-v1.6-34b only (checked by validate_framework_request before load)
  --generator_id "llava-hf/llava-v1.6-34b-hf"

  # --- rag.py only: uncomment when PIPELINE="rag" ---
  # the full AffectNet train set as the knowledge base
  --knowledge_base_path "$PROJECT/data/affectnet/train_test_set/train_set.csv"
  # retrieval: CLIP large embeddings reduced by LDA. Both are the defaults; passed explicitly so the
  # log records the retriever this run used - the LDA is re-fitted from the knowledge base here
  --clip_model_id "openai/clip-vit-large-patch14"
  --dim_reduction lda
  # the framework from src/fer_rag/frameworks/registry.py
  --framework    rules_direct_zs_rag12
  # the inference-time gate (not the oracle): different top-1 and top-2 labels go to the framework
  --gate          gate
  # frameworks need --top_k 2 (the default); passed explicitly so the log records it
  --top_k         2
  # --enable_thinking is a bare on/off flag: present = thinking on, absent = off.
  # llava-v1.6-34b has NO thinking key in the registry, and frameworks reject
  # --enable_thinking before load. The results csv therefore has no thinking column.
  # --enable_thinking
  # --prompt        single-user-message
  # --start_batch   0
)
# --------------------------------------------------------- END PER-EXPERIMENT

RUN_ID="${RUN_NAME}_$(date '+%Y%m%d_%H%M')"
EXP_DIR="$PROJECT/experiments/$EXP_GROUP"
RESULTS_CSV="$EXP_DIR/${RUN_ID}.csv"
LOG_FILE="$EXP_DIR/${RUN_ID}.log"
mkdir -p "$EXP_DIR"

# Replaces SLURM's --output: from here on, everything this script and python print
# (stdout + stderr: prints, warnings, tracebacks) goes to the screen AND the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

# --------------------------------------------------------- provenance / git
GIT_BRANCH=$(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git -C "$PROJECT" rev-parse HEAD)
GIT_DIRTY=$(git -C "$PROJECT" status --porcelain)

echo "=============================================================="
echo " run name    : $RUN_NAME"
echo " exp group   : $EXP_GROUP"
echo " run id      : $RUN_ID"
echo " started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo " node        : $(hostname)   platform: RunPod"
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
echo " python      : $(python --version 2>&1)   ($(command -v python))"
echo " HF_HOME     : $HF_HOME"
echo " results csv : $RESULTS_CSV"
echo " log file    : $LOG_FILE"
echo "=============================================================="
nvidia-smi || true
echo "=============================================================="

# ------------------------------------------------------------------- run it
# The pipelines use relative imports (from ..generators), so they must be run
# as MODULES with src/ on PYTHONPATH - `python /path/to/zero_shot.py` fails.
cd "$PROJECT"
python -u -m "fer_rag.pipelines.$PIPELINE" "${PIPELINE_ARGS[@]}" --results_path "$RESULTS_CSV"

echo "=============================================================="
echo " finished    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " results     : $RESULTS_CSV"
echo "=============================================================="
