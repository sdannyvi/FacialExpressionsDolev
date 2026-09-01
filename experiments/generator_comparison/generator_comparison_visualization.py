"""
Retriever vs RAG comparison across the generator_comparison runs.

Validates the RAG results csvs listed in RAG_CSVS and feeds them all to compare_retriever_rag()
as a single dict, so the whole comparison lands in one table under "figures/", plus a pair of
confusion matrices per run.

Run name (the key in dfs, used in the table and in every file name) = the csv name without the
"rag_" pipeline prefix and the "_<job id>" suffix. That keeps the two gemma_4_31b runs apart,
which the folder name alone cannot do.

Run as a module from the project root, so that config and fer_rag both import:
    PYTHONPATH=src python -m experiments.generator_comparison.generator_comparison_visualization
"""

import os

import pandas as pd

from config import resolve_path
from fer_rag.evaluation.vis_results import validate_results
from fer_rag.evaluation.analysis import (compare_retriever_rag, compare_zeroshot_rag,
                                        error_analysis_retrieval_cases)

EXP_DIR = resolve_path("experiments/generator_comparison")
RUNS_DIR = EXP_DIR / "runs"

# the dataset every run was evaluated on, prefixes every file written
DATASET_NAME = "FER+"

# the RAG runs to compare, as paths under "runs/" - edit by hand to add or drop a run.
# deliberately left out: qwen_3_32b_thinking/rag_qwen_3_32b_thinking_8703.csv, that run died
# partway and was restarted, the resume run below is the one that counts.
RAG_CSVS = [
    "llava_next_34b/rag_llava_next_34b_8704.csv",
    "gemma_3_27b/rag_gemma_3_27b_8716.csv",
    "gemma_4_31b/rag_gemma_4_31b_no_thinking_8702.csv",
    "gemma_4_31b/rag_gemma_4_31b_thinking_8701.csv",
    "qwen_3_32b_instruct/rag_qwen_3_32b_instruct_8717.csv",
    "qwen_3_32b_thinking/rag_qwen_3_32b_thinking_resume_run_8757.csv",
]

# the zero-shot run of each model, keyed by the run name RAG_CSVS produced above, so the two
# frameworks of a pair are matched by name and not by position. a run with no entry here is
# simply left out of the zero-shot comparison, which is what makes this list independent of
# RAG_CSVS: add or drop a model in one without touching the other.
ZERO_SHOT_CSVS = {
    "llava_next_34b": "llava_next_34b/zero_shot_llava_next_34b_8730.csv",
    "gemma_3_27b": "gemma_3_27b/zero_shot_gemma_3_27b_8731.csv",
    "gemma_4_31b_no_thinking": "gemma_4_31b/zero_shot_gemma_4_31b_no_thinking_8734.csv",
    "gemma_4_31b_thinking": "gemma_4_31b/zero_shot_gemma_4_31b_thinking_8732.csv",
    "qwen_3_32b_instruct": "qwen_3_32b_instruct/zero_shot_qwen_3_32b_instruct_8729.csv",
    "qwen_3_32b_thinking_resume_run": "qwen_3_32b_thinking/zero_shot_qwen_3_32b_thinking_8755.csv",
}


# compare_retriever_rag writes to a relative "figures" folder, put it next to this script
os.chdir(EXP_DIR)

# validate every run before any metric is derived from it.
dfs = {}
for rel_path in RAG_CSVS:
    csv_path = RUNS_DIR / rel_path
    name = csv_path.stem.removeprefix("rag_").rsplit("_", 1)[0]
    print(f"\n=================== validating: {name} ===================")
    dfs[name] = validate_results(pd.read_csv(csv_path)).dropna(subset=["prediction"])

# one comparison table for all runs together, plus 2 confusion matrices per run
print(f"\n=================== retriever vs RAG ===================")
compare_retriever_rag(dfs, dataset_name=DATASET_NAME)

# error analysis of the 8 retrieval cases, one table per run. the function takes a single
# dataframe, so the loop is here: that keeps it working unchanged for the ablation scripts that
# call it with one df, and it costs nothing to go from six runs to one - just edit RAG_CSVS.
# the run name has to go into dataset_name because the function builds its file name out of
# that argument alone, so a shared value would have every run overwrite the previous one's pdf.
for name, df in dfs.items():
    print(f"\n=================== retrieval cases: {name} ===================")
    error_analysis_retrieval_cases(df, dataset_name=f"{DATASET_NAME} - {name}")



# zero-shot vs RAG for the same generator, one table for all models together.
# the two frameworks ran as separate jobs, and a row dropped from one side is not dropped from
# the other, so the pair is cut down to the rows both scored and both are sorted by "file_path".
# without that the pair fails the is_same_dataset check inside the function and the model is
# left out: qwen_3_32b_thinking has one row fewer on the RAG side, which pushes every row after
# it out of alignment.
zero_shot_rag_pairs = {}
for name, rel_path in ZERO_SHOT_CSVS.items():
    csv_path = RUNS_DIR / rel_path
    print(f"\n=================== validating: zero-shot {name} ===================")
    df_zero_shot = validate_results(pd.read_csv(csv_path)).dropna(subset=["prediction"])
    df_rag = dfs[name]

    shared_paths = set(df_zero_shot["file_path"]) & set(df_rag["file_path"])
    if len(shared_paths) < len(df_rag):
        print(f"'{name}': the two frameworks share {len(shared_paths)} rows, out of "
              f"{len(df_zero_shot)} zero-shot and {len(df_rag)} RAG rows. only the shared rows "
              f"are compared.")
    align = lambda df: (df[df["file_path"].isin(shared_paths)]
                        .sort_values("file_path").reset_index(drop=True))
    zero_shot_rag_pairs[name] = (align(df_zero_shot), align(df_rag))

print(f"\n=================== zero-shot vs RAG ===================")
compare_zeroshot_rag(zero_shot_rag_pairs, dataset_name=DATASET_NAME)
