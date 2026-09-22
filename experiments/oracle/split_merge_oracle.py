"""
Split a test set to the samples only the oracle gate sends to a framework, and merge an oracle run back.

This saves running time only for the oracle experiment and is not part of the framework: given a full test
set, ``rag.py --framework <name> --gate oracle`` routes every sample itself. The oracle gate differs from
the gate only on Low retrieval samples whose top-1 and top-2 labels are the same wrong label: the gate
keeps them on the original RAG, the oracle sends them to the framework. Every other sample is routed the
same by both, so its result is taken from the runs that already exist instead of being run again.

    split  the rows of --test_path that the oracle gate sends to the framework and the gate does not (both
           top-1 and top-2 labels wrong and equal), in an original RAG results csv (top_k=2) of that test
           set, with the test csv's columns and row order. Run
           rag.py --framework <name> --gate oracle on the output: every sample goes to the framework.
    merge  an oracle framework run on the split test csv + the gated framework run of the same framework
           + the original RAG results -> oracle results for the full test set: High retrieval rows from
           the original RAG run, gated rows from the gated framework run, the rows only the oracle routes
           from the oracle framework run. Checks that each framework run holds exactly its samples, that
           the pipeline routed all of them to the framework, that both are runs of the same framework, and
           that they retrieved the same top-1/top-2 labels as the original RAG run.

Both gates are imported from fer_rag, so the split can never disagree with what the pipeline routes.

Run as a module from the project root, so that config and fer_rag both import:
    PYTHONPATH=src python -m experiments.oracle.split_merge_oracle split \\
        --rag_results <original rag results.csv> --test_path <test.csv> \\
        --output_path experiments/oracle/subst_to_test/<dataset>_val_both_wrong_ungated.csv
    (without --output_path the split csv is written to subst_to_test/ferplus_val_both_wrong_ungated.csv)
    PYTHONPATH=src python -m experiments.oracle.split_merge_oracle merge \\
        --oracle_results experiments/oracle/runs/<framework>/<oracle results.csv> \\
        --gated_results <gated framework results.csv> --rag_results <original rag results.csv> \\
        --output_path <oracle full-test-set results.csv>
"""

import argparse
from pathlib import Path

import pandas as pd

from fer_rag.frameworks.registry import needs_new_framework, oracle_needs_new_framework

# where every split csv of this experiment is stored; the default --output_path is in it
SUBSET_TO_TEST_DIR = Path(__file__).resolve().parent / "subst_to_test"


def route_samples(rag_df):
    """
    rag_df: original RAG results (top_k=2) of the test set.
    returns: (gated_paths, oracle_only_paths, high_paths): the file paths the gate sends to the framework,
             the ones only the oracle gate sends to the framework, and the ones both keep on the original RAG.
    """
    gated_paths, oracle_only_paths, high_paths = set(), set(), set()
    for row in rag_df.itertuples():
        # both gates compare this sample's own top-1 and top-2 labels; the oracle also its true label
        top_labels = [row.top_label_1, row.top_label_2]
        gate = needs_new_framework(top_labels)
        oracle = oracle_needs_new_framework(top_labels, row.true_label)
        # every sample the gate lets through, the oracle must let through too, so the gated run is reused as is
        if gate and not oracle:
            raise ValueError(f"{row.file_path} passes the gate but not the oracle gate; the gated run cannot be "
                             f"reused for the oracle results.")
        if gate:
            gated_paths.add(row.file_path)
        elif oracle:
            oracle_only_paths.add(row.file_path)
        else:
            high_paths.add(row.file_path)
    return gated_paths, oracle_only_paths, high_paths


def validate_rag_results(rag_df, rag_results_path):
    """Reject an original RAG results csv the routing cannot be read from."""
    if rag_df["file_path"].duplicated().any():
        raise ValueError(f"{rag_results_path} has duplicated file_path rows.")
    if rag_df[["top_label_1", "top_label_2", "true_label"]].isna().any().any():
        raise ValueError(f"{rag_results_path} has rows without a top-1 label, top-2 label or true label; it must "
                         f"come from a finished run with top_k=2.")


def split_oracle_samples(rag_results_path, test_path, output_path):
    """
    rag_results_path: original RAG results csv (top_k=2) of the test set.
    test_path: the test csv to split.
    output_path: where to write the test csv holding only the samples the oracle gate adds.
    """
    rag_df = pd.read_csv(rag_results_path)
    test_df = pd.read_csv(test_path)

    validate_rag_results(rag_df, rag_results_path)
    not_in_results = set(test_df["file_path"]) - set(rag_df["file_path"])
    if not_in_results:
        raise ValueError(f"{len(not_in_results)} test samples are not in {rag_results_path}; the results csv "
                         f"must come from a run on this test set.")

    gated_paths, oracle_only_paths, high_paths = route_samples(rag_df)
    subset = test_df[test_df["file_path"].isin(oracle_only_paths)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)

    print(f"routing of {len(rag_df)} samples: {len(high_paths)} High retrieval (original RAG in both gates), "
          f"{len(gated_paths)} pass the gate, {len(oracle_only_paths)} pass only the oracle gate")
    print(f"oracle-only samples: {len(subset)} of {len(test_df)} test samples have the same wrong top-1 and "
          f"top-2 label")
    print(f"true_label counts: {subset['true_label'].value_counts().to_dict()}")
    print(f"saved to {output_path}")


def check_framework_run(framework_df, framework_results_path, expected_paths, rag_df, run_name):
    """
    Check that a framework run holds exactly expected_paths, that the pipeline routed all of them to the
    framework, and that it retrieved the same top-1/top-2 labels as the original RAG run.
    """
    if framework_df["file_path"].duplicated().any():
        raise ValueError(f"{framework_results_path} has duplicated file_path rows.")

    missing = expected_paths - set(framework_df["file_path"])
    extra = set(framework_df["file_path"]) - expected_paths
    if missing or extra:
        raise ValueError(f"the {run_name} results do not match its samples: {len(missing)} samples missing, "
                         f"{len(extra)} samples that do not belong to it.")

    # every sample must have been routed to the framework by the pipeline's own retrieval
    not_routed = framework_df["route"] != "framework"
    if not_routed.any():
        raise ValueError(f"{int(not_routed.sum())} rows of {framework_results_path} were not routed to the "
                         f"framework (route != 'framework'). An oracle run must use --gate oracle.")
    retrieved = framework_df.set_index("file_path")[["top_label_1", "top_label_2"]]
    baseline = rag_df.set_index("file_path")[["top_label_1", "top_label_2"]].reindex(retrieved.index)
    changed = (retrieved != baseline).any(axis=1)
    if changed.any():
        raise ValueError(f"{int(changed.sum())} samples retrieved different top-1/top-2 labels in the {run_name} "
                         f"than in the original RAG run, so the two runs are not comparable.")


def merge_oracle_results(oracle_results_path, gated_results_path, rag_results_path, output_path):
    """
    oracle_results_path: rag.py --framework <name> --gate oracle results on the split test csv.
    gated_results_path: rag.py --framework <name> results of the same framework on the gated samples.
    rag_results_path: original RAG results csv (top_k=2) of the full test set.
    output_path: where to write the oracle results for the full test set.
    """
    # round_trip reads the stored float values back exactly
    oracle_df = pd.read_csv(oracle_results_path, float_precision="round_trip")
    gated_df = pd.read_csv(gated_results_path, float_precision="round_trip")
    rag_df = pd.read_csv(rag_results_path, float_precision="round_trip")
    validate_rag_results(rag_df, rag_results_path)

    # the two framework runs must be runs of the same framework: the framework columns are its own
    if set(oracle_df.columns) != set(gated_df.columns):
        raise ValueError(f"{oracle_results_path} and {gated_results_path} have different columns, so they are "
                         f"not runs of the same framework.")

    gated_paths, oracle_only_paths, high_paths = route_samples(rag_df)
    check_framework_run(gated_df, gated_results_path, gated_paths, rag_df, "gated framework run")
    check_framework_run(oracle_df, oracle_results_path, oracle_only_paths, rag_df, "oracle framework run")

    original_df = rag_df[rag_df["file_path"].isin(high_paths)].assign(route="original")
    columns = list(gated_df.columns) + [c for c in original_df.columns if c not in gated_df.columns]
    merged = pd.concat([original_df, gated_df, oracle_df], ignore_index=True)[columns]
    # back to the original RAG results' row order
    order = {file_path: position for position, file_path in enumerate(rag_df["file_path"])}
    merged = merged.sort_values("file_path", key=lambda paths: paths.map(order)).reset_index(drop=True)

    # no sample may end up twice in the merged results, e.g. if two of the three sources held the same row
    duplicated = merged["file_path"].duplicated()
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} file_path rows appear more than once in the merged results.")

    # ===== TEMPORARY VALIDATION - START (delete this block once a real merge has passed it) =====
    # the reordering: the merged rows must be the original RAG results' samples, each exactly once, in the
    # original RAG results' row order
    if merged["file_path"].tolist() != rag_df["file_path"].tolist():
        raise AssertionError("the merged rows are not in the original RAG results' row order.")
    print("validation passed: merged rows are in the original RAG results' row order")
    # every row of the merged file must hold exactly the values of the row it came from, under the same
    # columns: the High retrieval rows against original_df, the framework rows against their own run
    merged_by_path = merged.set_index("file_path")
    for source_name, source_df in [("original RAG", original_df), ("gated framework", gated_df),
                                   ("oracle framework", oracle_df)]:
        source_by_path = source_df.set_index("file_path")
        shared_columns = [column for column in source_by_path.columns if column in merged_by_path.columns]
        pd.testing.assert_frame_equal(merged_by_path.loc[source_by_path.index, shared_columns],
                                      source_by_path[shared_columns], check_dtype=False,
                                      obj=f"merged rows vs {source_name} rows")
    print("validation passed: every merged row matches its source row, column by column")
    # ===== TEMPORARY VALIDATION - END =====

    merged.to_csv(output_path, index=False)

    print(f"merged: {len(gated_df)} gated framework rows + {len(oracle_df)} oracle framework rows + "
          f"{len(original_df)} original RAG rows = {len(merged)} rows")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a test set to the samples only the oracle gate routes, "
                                                 "or merge an oracle framework run back.")
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("split", help="write the rows of a test csv that only the oracle gate "
                                                     "sends to the framework.")
    split_parser.add_argument("--rag_results", required=True, help="original RAG results csv (top_k=2) of the test set.")
    split_parser.add_argument("--test_path", required=True, help="the test csv to split.")
    split_parser.add_argument("--output_path", default=str(SUBSET_TO_TEST_DIR / "ferplus_val_both_wrong_ungated.csv"),
                              help=f"full path of the oracle-only test csv, in the subst_to_test folder, e.g. "
                                   f"{SUBSET_TO_TEST_DIR / 'fer_plus_both_wrong_ungated.csv'}. "
                                   f"Default: {SUBSET_TO_TEST_DIR / 'ferplus_val_both_wrong_ungated.csv'}")

    merge_parser = commands.add_parser("merge", help="merge an oracle framework run with the gated framework run "
                                                     "and the original RAG results.")
    merge_parser.add_argument("--oracle_results", required=True,
                              help="rag.py --framework <name> --gate oracle results on the split test csv.")
    merge_parser.add_argument("--gated_results", required=True,
                              help="rag.py --framework <name> results of the same framework on the gated samples.")
    merge_parser.add_argument("--rag_results", required=True, help="original RAG results csv (top_k=2) of the full test set.")
    merge_parser.add_argument("--output_path", required=True, help="where to write the oracle full-test-set results csv.")

    args = parser.parse_args()
    if args.command == "split":
        split_oracle_samples(args.rag_results, args.test_path, args.output_path)
    else:
        merge_oracle_results(args.oracle_results, args.gated_results, args.rag_results, args.output_path)
