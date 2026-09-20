"""
Split a test set to the samples the gate sends to a framework, and merge a framework run back.

This saves running time only for this experiment and is not part of the framework: given a full test
set, ``rag.py --framework <name>`` routes every sample itself. Samples whose top-1 and top-2 labels
agree get the original RAG prediction in every framework run, so they are taken from one original RAG
run instead of being run again.

    split  the rows of --test_path whose top-1 and top-2 labels differ in an original RAG results csv
           (top_k=2) of that test set, with the test csv's columns and row order. Run
           rag.py --framework <name> on the output: every sample passes the gate.
    merge  a framework run on the split test csv + the original RAG results -> results for the full
           test set: gated rows from the framework run, the others from the original RAG run.
           Checks that the framework run holds exactly the gated samples, that the pipeline routed all
           of them to the framework, and that it retrieved the same top-1/top-2 labels.

The gate is imported from fer_rag, so the split can never disagree with what the pipeline routes.

Run as a module from the project root, so that config and fer_rag both import:
    PYTHONPATH=src python -m experiments.gated_framework_comparison.split_merge_gated split \\
        --rag_results <original rag results.csv> --test_path <test.csv> \\
        --output_path experiments/gated_framework_comparison/gated_val_set/<dataset>_val_gated.csv
    (without --output_path the split csv is written to gated_val_set/ferplus_val_gated.csv)
    PYTHONPATH=src python -m experiments.gated_framework_comparison.split_merge_gated merge \\
        --framework_results runs/<framework>/<results.csv> --rag_results <original rag results.csv> \\
        --output_path full_val_set_results/<framework>.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from fer_rag.frameworks.registry import needs_new_framework

# where every split csv of this experiment is stored; the default --output_path is in it
GATED_VAL_SET_DIR = Path(__file__).resolve().parent / "gated_val_set"


def split_gated_samples(rag_results_path, test_path, output_path):
    """
    rag_results_path: original RAG results csv (top_k=2) of the test set.
    test_path: the test csv to split.
    output_path: where to write the test csv holding only the gated samples.
    """
    rag_df = pd.read_csv(rag_results_path)
    test_df = pd.read_csv(test_path)

    if rag_df["file_path"].duplicated().any():
        raise ValueError(f"{rag_results_path} has duplicated file_path rows.")
    if rag_df[["top_label_1", "top_label_2"]].isna().any().any():
        raise ValueError(f"{rag_results_path} has rows without a top-1 or top-2 label; it must come from a "
                         f"finished run with top_k=2.")
    not_in_results = set(test_df["file_path"]) - set(rag_df["file_path"])
    if not_in_results:
        raise ValueError(f"{len(not_in_results)} test samples are not in {rag_results_path}; the results csv "
                         f"must come from a run on this test set.")

    gated_paths = set()
    for row in rag_df.itertuples():
        # the gate compares this sample's own top-1 and top-2 labels
        if needs_new_framework([row.top_label_1, row.top_label_2]):
            gated_paths.add(row.file_path)
    subset = test_df[test_df["file_path"].isin(gated_paths)]
    subset.to_csv(output_path, index=False)

    print(f"gated samples: {len(subset)} of {len(test_df)} test samples have different top-1 and top-2 labels")
    print(f"true_label counts: {subset['true_label'].value_counts().to_dict()}")
    print(f"saved to {output_path}")


def merge_framework_results(framework_results_path, rag_results_path, output_path):
    """
    framework_results_path: rag.py --framework results on the split test csv.
    rag_results_path: original RAG results csv (top_k=2) of the full test set.
    output_path: where to write the results for the full test set.
    """
    # round_trip reads the stored float values back exactly
    framework_df = pd.read_csv(framework_results_path, float_precision="round_trip")
    rag_df = pd.read_csv(rag_results_path, float_precision="round_trip")
    for path, df in [(framework_results_path, framework_df), (rag_results_path, rag_df)]:
        if df["file_path"].duplicated().any():
            raise ValueError(f"{path} has duplicated file_path rows.")

    gated_paths = set()
    for row in rag_df.itertuples():
        # the gate compares this sample's own top-1 and top-2 labels
        if needs_new_framework([row.top_label_1, row.top_label_2]):
            gated_paths.add(row.file_path)
    gated_df = rag_df[rag_df["file_path"].isin(gated_paths)]

    # the framework run must hold exactly the gated samples
    missing = set(gated_df["file_path"]) - set(framework_df["file_path"])
    extra = set(framework_df["file_path"]) - set(gated_df["file_path"])
    if missing or extra:
        raise ValueError(f"the framework results do not match the gated samples: {len(missing)} gated samples "
                         f"missing, {len(extra)} samples that the gate keeps on the original RAG.")

    # every sample must have been routed to the framework by the pipeline's own retrieval
    not_routed = framework_df["route"] != "framework"
    if not_routed.any():
        raise ValueError(f"{int(not_routed.sum())} rows of {framework_results_path} were not routed to the "
                         f"framework (route != 'framework').")
    retrieved = framework_df.set_index("file_path")[["top_label_1", "top_label_2"]]
    baseline = gated_df.set_index("file_path")[["top_label_1", "top_label_2"]].reindex(retrieved.index)
    changed = (retrieved != baseline).any(axis=1)
    if changed.any():
        raise ValueError(f"{int(changed.sum())} samples retrieved different top-1/top-2 labels in the framework "
                         f"run than in the original RAG run, so the two runs are not comparable.")

    original_df = rag_df[~rag_df["file_path"].isin(gated_paths)].assign(route="original")
    columns = list(framework_df.columns) + [c for c in original_df.columns if c not in framework_df.columns]
    merged = pd.concat([original_df, framework_df], ignore_index=True)[columns]
    # back to the original RAG results' row order
    order = {file_path: position for position, file_path in enumerate(rag_df["file_path"])}
    merged = merged.sort_values("file_path", key=lambda paths: paths.map(order)).reset_index(drop=True)

    # ===== TEMPORARY VALIDATION - START (delete this block once a real merge has passed it) =====
    # the reordering: the merged rows must be the original RAG results' samples, each exactly once, in the
    # original RAG results' row order
    if merged["file_path"].tolist() != rag_df["file_path"].tolist():
        raise AssertionError("the merged rows are not in the original RAG results' row order.")
    print("validation passed: merged rows are in the original RAG results' row order")
    # every row of the merged file must hold exactly the values of the row it came from, under the same
    # columns: the original RAG rows against original_df, the framework rows against framework_df
    merged_by_path = merged.set_index("file_path")
    for source_name, source_df in [("original RAG", original_df), ("framework", framework_df)]:
        source_by_path = source_df.set_index("file_path")
        shared_columns = [column for column in source_by_path.columns if column in merged_by_path.columns]
        pd.testing.assert_frame_equal(merged_by_path.loc[source_by_path.index, shared_columns],
                                      source_by_path[shared_columns], check_dtype=False,
                                      obj=f"merged rows vs {source_name} rows")
    print("validation passed: every merged row matches its source row, column by column")
    # ===== TEMPORARY VALIDATION - END =====

    merged.to_csv(output_path, index=False)

    print(f"merged: {len(framework_df)} framework rows + {len(original_df)} original RAG rows = {len(merged)} rows")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a test set by the gate, or merge a framework run back.")
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("split", help="write the gated rows of a test csv.")
    split_parser.add_argument("--rag_results", required=True, help="original RAG results csv (top_k=2) of the test set.")
    split_parser.add_argument("--test_path", required=True, help="the test csv to split.")
    split_parser.add_argument("--output_path", default=str(GATED_VAL_SET_DIR / "ferplus_val_gated.csv"),
                              help=f"full path of the gated test csv, in the gated_val_set folder, e.g. "
                                   f"{GATED_VAL_SET_DIR / 'fer_plus_gated.csv'}. "
                                   f"Default: {GATED_VAL_SET_DIR / 'ferplus_val_gated.csv'}")

    merge_parser = commands.add_parser("merge", help="merge a gated framework run with the original RAG results.")
    merge_parser.add_argument("--framework_results", required=True, help="rag.py --framework results on the gated test csv.")
    merge_parser.add_argument("--rag_results", required=True, help="original RAG results csv (top_k=2) of the full test set.")
    merge_parser.add_argument("--output_path", required=True, help="where to write the full-test-set results csv.")

    args = parser.parse_args()
    if args.command == "split":
        split_gated_samples(args.rag_results, args.test_path, args.output_path)
    else:
        merge_framework_results(args.framework_results, args.rag_results, args.output_path)
