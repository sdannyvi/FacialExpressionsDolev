"""
Generator comparison - qwen3-32b thinking.

For now this script does one thing: confirm that the RAG run of the qwen3-32b thinking
generator was evaluated on exactly the rows of the public_test_50% split - the same rows, in
the same order - so that a later comparison against another generator's run compares
generators and not samples.

Run from the repo root:

    python -m experiments.generator_comparison.generator_comparison_visualization
"""

import pandas as pd

from config import resolve_path
from src.fer_rag.evaluation.analysis import is_same_dataset

# the RAG run to check, and the test split it was supposed to have been run on
RAG_RESULTS_PATH = resolve_path("experiments/generator_comparison/runs/qwen_3_32b_thinking/"
                                "rag_qwen_3_32b_thinking_resume_run_8757.csv")
TEST_SET_PATH = resolve_path("rag_thresholds/train_test_sets/public_test_50%.csv")


def main():
    df_rag = pd.read_csv(RAG_RESULTS_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)

    is_same_dataset(df_rag, df_test,
                    name_a="qwen3-32b thinking RAG run",
                    name_b="public_test_50%")


if __name__ == "__main__":
    main()
