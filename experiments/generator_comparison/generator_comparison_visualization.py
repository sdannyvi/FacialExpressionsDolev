"""
Retriever vs RAG, retrieval-case error analysis, and zero-shot vs RAG, across the
generator_comparison runs.

RUNS lists every run by name, and holds the two csvs of that run - the RAG one and the zero-shot
one - in the same entry, so the two are paired by where they are written and not by a name parsed
out of a file name. that is what keeps the pairing correct where the file names alone would
mislead: gemma_4_31b has a thinking and a no-thinking run in one folder, and the two csvs of a
run carry different job ids.

Every csv in RUNS is validated before any metric is derived from it. The RAG-only analyses are
then handed just the RAG dataframes, and the zero-shot comparison the pairs.

Run name (the key in RUNS, used in every table and file name) is written by hand, so it says
what the run is rather than what its file happens to be called.

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

# the runs to analyze: {run name: {method: csv path under "runs/"}}
RUNS = {
    "llava_next_34b": {
        "rag": "llava_next_34b/rag_llava_next_34b_8704.csv",
        "zero-shot": "llava_next_34b/zero_shot_llava_next_34b_8730.csv",
    },
    "gemma_3_27b": {
        "rag": "gemma_3_27b/rag_gemma_3_27b_8716.csv",
        "zero-shot": "gemma_3_27b/zero_shot_gemma_3_27b_8731.csv",
    },
    "gemma_4_31b_no_thinking": {
        "rag": "gemma_4_31b/rag_gemma_4_31b_no_thinking_8702.csv",
        "zero-shot": "gemma_4_31b/zero_shot_gemma_4_31b_no_thinking_8734.csv",
    },
    "gemma_4_31b_thinking": {
        "rag": "gemma_4_31b/rag_gemma_4_31b_thinking_8701.csv",
        "zero-shot": "gemma_4_31b/zero_shot_gemma_4_31b_thinking_8732.csv",
    },
    "qwen_3_32b_instruct": {
        "rag": "qwen_3_32b_instruct/rag_qwen_3_32b_instruct_8717.csv",
        "zero-shot": "qwen_3_32b_instruct/zero_shot_qwen_3_32b_instruct_8729.csv",
    },
    "qwen_3_32b_thinking": {
        "rag": "qwen_3_32b_thinking/rag_qwen_3_32b_thinking_8757.csv",
        "zero-shot": "qwen_3_32b_thinking/zero_shot_qwen_3_32b_thinking_8755.csv",
    },
}

# the analysis functions write to a relative "figures" folder, put it next to this script
os.chdir(EXP_DIR)

# validate every csv 
dfs = {}
for name, csvs in RUNS.items():
    dfs[name] = {}
    for method, rel_path in csvs.items():
        print(f"\n=================== validating: {method} {name} ===================")
        df = validate_results(pd.read_csv(RUNS_DIR / rel_path))
        # a row with no prediction cannot be scored and makes the analyses raise, so its dropped
        dfs[name][method] = df.dropna(subset=["prediction"])

# the RAG-only analyses take one dataframe per run, so hand them just the RAG side
rag_dfs = {}
for name, methods in dfs.items():
    rag_dfs[name] = methods["rag"]

# one comparison table for all runs together, plus 2 confusion matrices per run
print(f"\n=================== retriever vs RAG ===================")
compare_retriever_rag(rag_dfs, dataset_name=DATASET_NAME)

# error analysis of the 8 retrieval cases, one table per run. 
for name, df in rag_dfs.items():
    print(f"\n=================== retrieval cases: {name} ===================")
    error_analysis_retrieval_cases(df, dataset_name=f"{DATASET_NAME} - {name}")

# zero-shot vs RAG for the same generator, one table for all runs together.
zero_shot_rag_pairs = {}
for name, methods in dfs.items():
    zero_shot_rag_pairs[name] = (methods["zero-shot"], methods["rag"])

print(f"\n=================== zero-shot vs RAG ===================")
compare_zeroshot_rag(zero_shot_rag_pairs, dataset_name=DATASET_NAME)
