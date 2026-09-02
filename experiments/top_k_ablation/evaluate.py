"""
Retriever vs RAG, retrieval-case error analysis, and zero-shot vs RAG, across the
top_k_ablation runs.

The ablation varies ONE thing: how many retrieved neighbours the generator sees
(top_k = 2, 3, 4, 5). Generator, knowledge base and test set are identical in all four runs, so a
difference between two rows of any table below is a difference caused by k and by nothing else.

k=2 is the original run from generator_comparison rather than a run made for this ablation - it is
the same generator on the same test set and knowledge base, with the same lda reduction, prompt
and greedy decoding, so it belongs in the series and re-running it would only add sampling noise.
It is the k the earlier "RAG barely beats the retriever" reading was based on, which makes it the
point the other three have to be read against.

RUNS lists every run by name, and holds the two csvs of that run - the RAG one and the zero-shot
one - in the same entry, so the two are paired by where they are written and not by a name parsed
out of a file name.

The zero-shot side is deliberately the SAME csv in all four entries: zero-shot never retrieves,
so it has no k to vary, and re-running it per k would only add sampling noise to a baseline that
is by construction constant. It therefore lives in generator_comparison (where it was produced)
and is reused from here - which is why every path in RUNS is project-relative rather than relative
to this experiment's own "runs" folder.

Every csv in RUNS is validated before any metric is derived from it. The RAG-only analyses are
then handed just the RAG dataframes, and the zero-shot comparison the pairs.

Run name (the key in RUNS, used in every table and file name) is written by hand, so it says
what the run is rather than what its file happens to be called.

Run as a module from the project root, so that config and fer_rag both import:
    PYTHONPATH=src python -m experiments.top_k_ablation.evaluate
"""

import os

import pandas as pd

from config import resolve_path
from fer_rag.evaluation.vis_results import validate_results
from fer_rag.evaluation.analysis import (compare_retriever_rag, compare_zeroshot_rag,
                                         error_analysis_retrieval_cases)

EXP_DIR = resolve_path("experiments/top_k_ablation")

# the dataset every run was evaluated on, prefixes every file written
DATASET_NAME = "FER+"

# the zero-shot baseline shared by all four runs - same generator, no retrieval, so no k.
ZERO_SHOT_CSV = ("experiments/generator_comparison/runs/gemma_4_31b/"
                 "zero_shot_gemma_4_31b_no_thinking_8734.csv")

# the runs to analyze: {run name: {method: csv path from the project root}}
# ordered by k, so every table below reads as the ablation curve
RUNS = {
    "top_k_2": {
        "rag": ("experiments/generator_comparison/runs/gemma_4_31b/"
                "rag_gemma_4_31b_no_thinking_8702.csv"),
        "zero-shot": ZERO_SHOT_CSV,
    },
    "top_k_3": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_3_8827.csv",
        "zero-shot": ZERO_SHOT_CSV,
    },
    "top_k_4": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_4_8828.csv",
        "zero-shot": ZERO_SHOT_CSV,
    },
    "top_k_5": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_5_8829.csv",
        "zero-shot": ZERO_SHOT_CSV,
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
        df = validate_results(pd.read_csv(resolve_path(rel_path)))
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
