"""
Per retrieval case, the share of samples the gate keeps on the original RAG and the share it routes to
the alternative method. Percentages are out of the retrieval case's own samples.

Run from the project root:
    PYTHONPATH=src python -m experiments.gated_framework_comparison.gate_analysis --rag_path <rag results.csv>
"""

import argparse
import os

import pandas as pd

from fer_rag.evaluation.analysis import add_gate_column, split_by_retrieval_case, table_gate_routing, table_override_rescue
from fer_rag.evaluation.vis_results import validate_results

parser = argparse.ArgumentParser()
parser.add_argument("--rag_path", help="Path to the RAG predictions CSV",
                    default="/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/llava_next_34b/rag_llava_next_34b_8704.csv")
parser.add_argument("--zs_path", help="Path to the zero-shot predictions CSV",
                    default="/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/llava_next_34b/zero_shot_llava_next_34b_8730.csv")
args = parser.parse_args()

rag_df = validate_results(pd.read_csv(args.rag_path))
zs_df = validate_results(pd.read_csv(args.zs_path))

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "gate_analysis")
os.makedirs(figures_dir, exist_ok=True)

df = split_by_retrieval_case(add_gate_column(rag_df))
table_gate_routing(df, figures_dir)

# the same override / rescue table as in failure_analysis, once per gate side; percentages are out of all
# the RAG results' samples, so each side's cells are accuracy points of the whole set
table_override_rescue(df[df["passed_gate"]], zs_df, figures_dir, total=len(df),
                      title="Override and rescue by retrieval case - table - gated samples, out of all samples")
table_override_rescue(df[~df["passed_gate"]], zs_df, figures_dir, total=len(df),
                      title="Override and rescue by retrieval case - table - non-gated samples, out of all samples")
