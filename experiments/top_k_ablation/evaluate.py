"""
Retriever vs RAG, evidence-regime error analysis, and zero-shot vs RAG, across the
top_k_ablation runs.

The retriever is scored twice on purpose. As 1-NN (compare_retriever_rag) it is the simplest
model anyone would deploy instead of a 31B generator, and it does not move with k, so it is the
fixed line everything else is read against. As k-NN (compare_knn_retriever_rag) it sees exactly
the examples the generator saw, so the gap to RAG is what the GENERATOR added rather than what
the extra retrieval added. Only the second one can answer whether the generator is reasoning over
the examples or just counting them.

The 8-case error table is replaced by the evidence-regime table: enumerating every combination of
the retrieved labels needs 2^k * 2 rows, which is readable at k=2 and useless by k=5.

The ablation varies ONE thing: how many retrieved neighbours the generator sees
(top_k = 2 through 8). Generator, knowledge base and test set are identical in all seven runs, so a
difference between two rows of any table below is a difference caused by k and by nothing else.

k=2 is the original run from generator_comparison rather than a run made for this ablation - it is
the same generator on the same test set and knowledge base, with the same lda reduction, prompt
and greedy decoding, so it belongs in the series and re-running it would only add sampling noise.
It is the k the earlier "RAG barely beats the retriever" reading was based on, which makes it the
point the other six have to be read against.

RUNS lists every run by name, and holds the two csvs of that run - the RAG one and the zero-shot
one - in the same entry, so the two are paired by where they are written and not by a name parsed
out of a file name.

The zero-shot side is deliberately the SAME csv in all seven entries: zero-shot never retrieves,
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
from fer_rag.evaluation.analysis import (compare_knn_retriever_rag, compare_retriever_rag,
                                         compare_zeroshot_rag, error_analysis_evidence_regimes)

EXP_DIR = resolve_path("experiments/top_k_ablation")

# the dataset every run was evaluated on, prefixes every file written
DATASET_NAME = "FER+"

# how the k-NN retriever turns its k retrieved labels into one prediction: "cosine" weights each
# neighbour by its similarity, "majority" gives each one vote. the cosines here all sit in
# [0.80, 1.00], so the two rules agree almost everywhere.
KNN_VOTING = "cosine"

# the zero-shot baseline shared by all seven runs - same generator, no retrieval, so no k.
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
    "top_k_6": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_6_8851.csv",
        "zero-shot": ZERO_SHOT_CSV,
    },
    "top_k_7": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_7_8852.csv",
        "zero-shot": ZERO_SHOT_CSV,
    },
    "top_k_8": {
        "rag": "experiments/top_k_ablation/runs/rag_gemma_4_31b_no_thinking_top_k_8_8853.csv",
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

# the retriever again, this time as a k-NN classifier over all the neighbours the generator saw,
# so RAG and the baseline are information-matched at every k.
print(f"\n=================== k-NN retriever vs RAG ===================")
compare_knn_retriever_rag(rag_dfs, dataset_name=DATASET_NAME, voting=KNN_VOTING)

# evidence regimes - high retrieval / conflicting / low retrieval - one table per run. the number
# the ablation exists to move is the conditional percentage of K1: of the queries whose evidence
# disagreed, how often the generator still landed on the truth.
for name, df in rag_dfs.items():
    print(f"\n=================== evidence regimes: {name} ===================")
    error_analysis_evidence_regimes(df, dataset_name=f"{DATASET_NAME} - {name}")

# zero-shot vs RAG for the same generator, one table for all runs together.
zero_shot_rag_pairs = {}
for name, methods in dfs.items():
    zero_shot_rag_pairs[name] = (methods["zero-shot"], methods["rag"])

print(f"\n=================== zero-shot vs RAG ===================")
compare_zeroshot_rag(zero_shot_rag_pairs, dataset_name=DATASET_NAME)
