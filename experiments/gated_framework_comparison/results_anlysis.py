"""
Analysis of one merged full-test-set results csv of a gated framework run (see split_merge_gated.py),
against the original RAG run the merge took its non-gated rows from.

Run from the project root:
    PYTHONPATH=src python -m experiments.gated_framework_comparison.results_anlysis \
        --gated_framework experiments/gated_framework_comparison/full_val_set_results/<framework>.csv
    (without --original_framework the llava-next-34b RAG run of the full validation set is used,
     and without --zs_path its zero-shot run)
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fer_rag.evaluation.analysis import TEXT_SIZE, is_same_dataset, split_by_retrieval_case, table_override_rescue
from fer_rag.evaluation.vis_results import validate_framework_results, validate_results
from fer_rag.frameworks.branches import BRANCH_EXAMPLES
from fer_rag.frameworks.decision_rules import RULES, apply_rule

# this experiment's folder, so that every figure is written next to this script
EXP_DIR = Path(__file__).resolve().parent
FIGURES_DIR = EXP_DIR / "figures"
# the original RAG run of the full validation set: the run the merge took the non-gated rows from
ORIGINAL_RAG_RESULTS = ("/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/"
                        "llava_next_34b/rag_llava_next_34b_8704.csv")
# the zero-shot run of the same model over the same test set, the ZS side of the override / rescue table
ZERO_SHOT_RESULTS = ("/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/"
                     "llava_next_34b/zero_shot_llava_next_34b_8730.csv")
# the decision rule whose answer is the prediction of the gated samples in the override / rescue table
OVERRIDE_RESCUE_RULE = "sum_class_conf"

COMPARISON_COLUMNS = ["Method", "Accuracy (%)", "Accuracy gap", "Macro F1 (%)", "Macro F1 gap"]

parser = argparse.ArgumentParser(description="Analyze a merged full-test-set framework results csv.")
parser.add_argument("--gated_framework", required=True,
                    help="Path to the merged full-test-set results CSV of one gated framework.")
parser.add_argument("--original_framework", default=ORIGINAL_RAG_RESULTS,
                    help=f"Path to the original RAG results CSV of the same test set. "
                         f"Default: {ORIGINAL_RAG_RESULTS}")
parser.add_argument("--zs_path", default=ZERO_SHOT_RESULTS,
                    help=f"Path to the zero-shot results CSV of the same test set. Default: {ZERO_SHOT_RESULTS}")
args = parser.parse_args()



# round_trip reads the stored float values back exactly, so the confidences and class scores are the
# run's own values and a rule recomputed from them gives the run's answer
gated_df = pd.read_csv(args.gated_framework, float_precision="round_trip")
original_df = pd.read_csv(args.original_framework, float_precision="round_trip")
zs_df = pd.read_csv(args.zs_path, float_precision="round_trip")

# plot_metrics and plot_classification_report write to a relative "figures" folder, put it next to this
# script. after the reads above, whose paths are relative to the project root the script is run from
os.chdir(EXP_DIR)

def _majority_vote(row, branches, classes_list):
    """
    The label most of the branches answered, as a decision rule the run did not store.
    row: one results row of a framework sample.
    branches: the branches that vote, e.g. ["zs", "rag_top1", "rag_top2"].
    classes_list: the classes, in the order used to break ties.
    returns: the label with the most votes. A tie goes to the tied label with the highest summed
             confidence, and then to the label listed first in classes_list. None when no branch answered.
    """
    votes = {}
    confidences = {}
    for branch in branches:
        prediction = row[f"{branch}__prediction"]
        confidence = row[f"{branch}__confidence"]
        # a branch with no answer, and so no confidence, does not vote (as in decision_rules._answers)
        if isinstance(prediction, str) and prediction and pd.notna(confidence):
            votes[prediction] = votes.get(prediction, 0) + 1
            confidences[prediction] = confidences.get(prediction, 0.0) + float(confidence)
    if not votes:
        return None

    most_votes = max(votes.values())
    tied = [label for label, count in votes.items() if count == most_votes]
    if len(tied) > 1:
        # first tie break: the summed confidence of the branches that answered the label
        best_confidence = max(confidences[label] for label in tied)
        tied = [label for label in tied if confidences[label] == best_confidence]
    if len(tied) > 1:
        # second tie break: the class order, as in the rules of decision_rules.py
        order = {class_name: position for position, class_name in enumerate(classes_list)}
        return min(tied, key=lambda label: order.get(label, len(classes_list)))
    return tied[0]


def _route_predictions(gated_df, framework_predictions):
    """
    The full-test-set prediction column of one decision rule: the rule's answer on the rows the gate
    routed to the framework, and the original RAG's prediction on the rows it kept, which the merge
    already wrote into "prediction".
    framework_predictions: a Series over the framework rows only.
    """
    framework_mask = gated_df["route"] == "framework"
    return gated_df["prediction"].where(~framework_mask, framework_predictions.reindex(gated_df.index))


def _score(true_labels, predictions, classes_list, method):
    """Accuracy and macro F1 as percentages, rounded where they are derived so every number reported
    here is the number the table shows. A row with no prediction is not scored either way: it stops the
    comparison, so that a table is never drawn over a method that answered fewer samples than the others."""
    n_missing = int(predictions.isna().sum())
    if n_missing:
        raise ValueError(f"'{method}' has no prediction in {n_missing} of {len(predictions)} rows, so it "
                         f"cannot be scored over the same samples as the other rows of the table. A rule "
                         f"returns no answer when no branch answered, or when a class score is missing.")
    accuracy = round(accuracy_score(true_labels, predictions) * 100, 2)
    macro_f1 = round(f1_score(true_labels, predictions, labels=classes_list, average="macro",
                              zero_division=0) * 100, 2)
    return accuracy, macro_f1


def compare_rules_to_rag(gated_df, original_df, figures_dir, title, classes_list=None):
    """
    Accuracy and macro F1 of the original RAG and of every decision rule of a gated framework run, in
    one table: the original RAG first, then one row per rule, then majority vote when the framework has
    at least three branches.
    Every rule is scored over the full test set the way the pipeline would have routed it had that rule
    been the framework's decision rule: the rule's answer on the samples the gate routed to the
    framework, the original RAG's prediction on the samples it kept.

    gated_df (DataFrame): a merged framework results file, validated with validate_framework_results.
    original_df (DataFrame): the original RAG results of the same test set, validated with validate_results.
    figures_dir: where the pdf is written.
    title (String): the table's title, e.g. the results file's name without ".csv". The pdf is saved as
                    <figures_dir>/<title> - rag and decision rules comparison.pdf
    classes_list: the classes, in the order the rules use to break ties. Default: the sorted true labels.
    returns: the table as a dataframe.
    """
    if classes_list is None:
        classes_list = sorted(original_df["true_label"].dropna().unique().tolist())

    # the rules this run stored; a run with none (an aggregator framework) has nothing to compare here
    rule_columns = [rule for rule in RULES if f"rule__{rule}" in gated_df.columns]
    if not rule_columns:
        raise ValueError(f"'{title}' holds none of the rule columns {[f'rule__{rule}' for rule in RULES]}, "
                         f"so there are no decision rules to compare. An aggregator framework stores its "
                         f"answer in the aggregator's columns instead.")
    branches = [branch for branch in BRANCH_EXAMPLES if f"{branch}__prediction" in gated_df.columns]

    # the two runs are separate jobs over the same test set, so report whether they scored the same rows
    if not is_same_dataset(original_df, gated_df, name_a="original RAG", name_b=title):
        warnings.warn(f"the original RAG run and '{title}' do not hold the same rows "
                      f"({len(original_df)} original RAG, {len(gated_df)} gated), so the rows behind the "
                      f"table's rows differ and every gap carries that difference.", UserWarning)

    # the first row: the original RAG, the baseline every gap below is measured from
    rag_accuracy, rag_f1 = _score(original_df["true_label"], original_df["prediction"], classes_list,
                                  method="Original RAG")
    print(f"Original RAG: accuracy {rag_accuracy}%, macro F1 {rag_f1}%")
    rows = [["Original RAG", f"{rag_accuracy:.2f}", "", f"{rag_f1:.2f}", ""]]

    # one row per decision rule, and majority vote when there are enough voters for it to decide
    framework_rows = gated_df[gated_df["route"] == "framework"]
    predictions_per_method = {rule: gated_df[f"rule__{rule}"] for rule in rule_columns}
    if len(branches) >= 3:
        predictions_per_method["majority"] = framework_rows.apply(
            lambda row: _majority_vote(row, branches, classes_list), axis=1)
    else:
        print(f"majority vote is not derived: the framework has {len(branches)} branches "
              f"({branches}), so every disagreement between them is a tie")

    for method, framework_predictions in predictions_per_method.items():
        predictions = _route_predictions(gated_df, framework_predictions)
        accuracy, macro_f1 = _score(gated_df["true_label"], predictions, classes_list,
                                    method=f"Gated Framework - {method}")
        # the gap against the original RAG: the difference between the two values the table shows
        accuracy_gap = round(accuracy - rag_accuracy, 2)
        f1_gap = round(macro_f1 - rag_f1, 2)
        print(f"Gated Framework - {method}: accuracy {accuracy}% ({accuracy_gap:+.2f}), "
              f"macro F1 {macro_f1}% ({f1_gap:+.2f})")
        rows.append([f"Gated Framework - {method}", f"{accuracy:.2f}", f"({accuracy_gap:+.2f})",
                     f"{macro_f1:.2f}", f"({f1_gap:+.2f})"])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(rows) + 2))
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(cellText=rows, colLabels=COMPARISON_COLUMNS, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(TEXT_SIZE)
    table.auto_set_column_width(col=list(range(len(COMPARISON_COLUMNS))))
    table.scale(1, 2.4)
    for (row, column), cell in table.get_celld().items():
        # only horizontal lines between rows, like the other tables of this experiment
        cell.visible_edges = "BT" if row == 0 else "B"
        cell.set_edgecolor("#d9d9d9")
        cell.PAD = 0.08

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{title} - rag and decision rules comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out_path}")
    plt.close(fig)

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def override_rescue_on_gated(gated_df, zs_df, figures_dir, title, rule=OVERRIDE_RESCUE_RULE):
    """
    The override / rescue table (table_override_rescue) over the gated samples only: the rows the gate
    routed to the framework, with the decision rule's answer as their prediction. Every percentage is out
    of the whole test set, so each cell is accuracy points of the full set.

    gated_df (DataFrame): a merged framework results file, validated with validate_framework_results.
    zs_df (DataFrame): the zero-shot results of the same test set, validated with validate_results.
    figures_dir: where the pdf is written.
    title (String): the table's file name without ".pdf", saved as <figures_dir>/<title>.pdf
    rule: the decision rule whose column, rule__<rule>, is the prediction of the gated samples.
    """
    rule_column = f"rule__{rule}"
    if rule_column not in gated_df.columns:
        raise ValueError(f"the gated framework results hold no '{rule_column}' column.")

    framework_rows = gated_df[gated_df["route"] == "framework"].copy()
    # a row the rule did not answer would be counted as wrong by the table, so it stops the table instead
    n_missing = int(framework_rows[rule_column].isna().sum())
    if n_missing:
        raise ValueError(f"'{rule_column}' has no answer in {n_missing} of {len(framework_rows)} gated rows.")
    framework_rows["prediction"] = framework_rows[rule_column]
    print(f"gated samples: {len(framework_rows)} of {len(gated_df)}, predicted by '{rule_column}'")

    table_override_rescue(split_by_retrieval_case(framework_rows), zs_df, figures_dir, title=title,
                          total=len(gated_df))


# the original RAG run has no framework columns, so only the checks every results file gets
print(f"\n=================== validating: original RAG ===================")
original_df = validate_results(original_df)

print(f"\n=================== validating: zero-shot ===================")
zs_df = validate_results(zs_df)

# the columns every results file has, then the framework's own columns: the second normalizes the label
# columns of the branches and the rules, and recomputes rule__sum_conf on the normalized answers
print(f"\n=================== validating: gated framework ===================")
gated_df = validate_results(gated_df)
gated_df = validate_framework_results(gated_df)

print(f"\n=================== original RAG vs the decision rules ===================")
compare_rules_to_rag(gated_df, original_df, FIGURES_DIR, title=Path(args.gated_framework).stem)

print(f"\n=================== override / rescue on the gated samples ===================")
override_rescue_on_gated(gated_df, zs_df, FIGURES_DIR,
                         title=f"{Path(args.gated_framework).stem} - Override rescue table out of total samples")
