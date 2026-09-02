"""
Analysis functions for RAG results: retriever (1-NN) vs RAG comparison, retrieval-case
error analysis, RAG vs zero-shot on retrieval failures, and retrieval tie statistics.

Working copy of the functions in ablation/visualize_results_part2.py. The originals stay
frozen under ablation/ because they produced the figures in the submitted draft.

Every function takes dataframes (not paths), and writes its figures to a relative
"figures" folder, so the output lands next to whichever experiment script called it.
Validate each results CSV with validate_results() (in vis_results) before passing it in,
and use is_same_dataset() to confirm two dataframes hold the exact same rows, in the
same order.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import warnings

from .vis_results import plot_confusion_matrix


def _head(items, max_report):
    """
    format a list for printing, capped at max_report entries, so a pair of dataframes that
    have nothing in common does not dump thousands of lines. the caller prints the full count.
    items (list): the values to print.
    max_report (int): how many entries to show.
    returns: a String, with "...and N more" appended when the list was cut.
    """
    shown = list(items[:max_report])
    if len(items) > max_report:
        return f"{shown} ...and {len(items) - max_report} more"
    return f"{shown}"


def is_same_dataset(df_a, df_b, name_a="", name_b="", max_report=50):
    """
    check that two dataframes hold the exact same rows, using "file_path" as the row id.
    three checks, and every one that fails is reported: both hold the same number of rows,
    both hold the same set of file paths (checked in both directions), and the file paths sit
    in the same order row by row. a mismatch warns, since comparisons that merge the two on
    "file_path" silently drop the unmatched rows, and comparisons that line the two up by
    position silently compare different samples.
    df_a, df_b (DataFrame): dataframes to compare, must contain "file_path".
    name_a, name_b (String): labels for the printed output only.
    max_report (int): how many entries of each reported list to print. the counts are always
                      printed in full, only the listings are capped.
    returns: True only if all three checks pass.
    """
    label_a = name_a or "first dataframe"
    label_b = name_b or "second dataframe"

    for df, label in ((df_a, label_a), (df_b, label_b)):
        if "file_path" not in df.columns:
            raise KeyError(f"'{label}' has no 'file_path' column, so its rows cannot be "
                           f"identified. columns found: {df.columns.tolist()}")

    # check 1: the same number of rows
    same_length = len(df_a) == len(df_b)

    # check 2: the same set of file paths
    paths_a = set(df_a["file_path"])
    paths_b = set(df_b["file_path"])
    same_paths = paths_a == paths_b

    # the file paths behind a set mismatch, in both directions, for the report below
    only_a = sorted(paths_a - paths_b)
    only_b = sorted(paths_b - paths_a)

    # check 3: the same order. compared by position (the index is dropped, so a non-default
    # index cannot make identical files look different), over the rows both dataframes have,
    # so a length mismatch still reports the order of the rows they share
    list_a = df_a["file_path"].reset_index(drop=True).tolist()
    list_b = df_b["file_path"].reset_index(drop=True).tolist()
    n_shared = min(len(list_a), len(list_b))
    row_matches = [list_a[row] == list_b[row] for row in range(n_shared)]
    same_order = all(row_matches)

    # the row numbers behind an order mismatch, for the report below
    mismatched_rows = [row for row, is_row_match in enumerate(row_matches) if not is_row_match]

    is_match = same_length and same_paths and same_order

    print(f"are '{label_a}' ({len(df_a)} rows) and '{label_b}' ({len(df_b)} rows) the exact "
          f"same dataset? {is_match}")

    if is_match:
        print(f"they match: same number of rows ({len(df_a)}), same set of file paths, and "
              f"the file paths are in the same order.")
        return True

    # report every check that failed
    if not same_length:
        print(f"the number of rows differs: '{label_a}' has {len(df_a)} rows, "
              f"'{label_b}' has {len(df_b)} rows.")

    if not same_paths:
        print(f"file paths that are only in '{label_a}' ({len(only_a)}): "
              f"{_head(only_a, max_report)}")
        print(f"file paths that are only in '{label_b}' ({len(only_b)}): "
              f"{_head(only_b, max_report)}")

    if not same_order:
        first_row = mismatched_rows[0]
        print(f"the first row where the file paths differ is row {first_row}: "
              f"'{label_a}' has '{list_a[first_row]}', '{label_b}' has '{list_b[first_row]}'")
        print(f"rows where the file paths differ ({len(mismatched_rows)} of the {n_shared} "
              f"rows the two dataframes share): {_head(mismatched_rows, max_report)}")

    warnings.warn(
        f"'{label_a}' and '{label_b}' are not the same dataset - comparisons that merge them "
        f"on 'file_path' will drop the unmatched rows, and comparisons that line them up by "
        f"position will compare different samples. row counts: {len(df_a)} and {len(df_b)}. "
        f"file paths only in '{label_a}': {len(only_a)}. "
        f"file paths only in '{label_b}': {len(only_b)}. "
        f"rows in a different order: {len(mismatched_rows)}. see the printed output above.",
        UserWarning,
    )

    return False


def cosine_tie_stats(df, dataset_name=""):
    """
    count how many times the top-2 retrieved examples had the exact same cosine similarity,
    and among those, how many times the two retrieved examples carried different labels.
    a tie with different labels means the order of the two examples shown to the generator
    is arbitrary.
    :param df: rag results dataframe
    :param dataset_name: used to label the printed output only
    :return: dict with the counts and percentages
    """
    print(dataset_name)

    # how many times the top-2 retrieved examples had the same cosine similarity
    # exact cosine ties
    same_cos = df["top_cosine_1"] == df["top_cosine_2"]
    same_dist_count = same_cos.sum()
    same_dist_pct = same_dist_count / len(df)
    print(f"same_dist_count: {same_dist_count}")
    print(f"same_dist_pct: {same_dist_pct:.6f} ({same_dist_pct*100:.4f}%)")

    # how many times the top-2 retrieved examples had the same cosine similarity AND different labels
    # among the ties, different labels
    diff_labels_in_ties = same_cos & (df["top_label_1"] != df["top_label_2"])
    count_diff_labels_in_ties = diff_labels_in_ties.sum()
    pct_diff_labels_in_ties = count_diff_labels_in_ties / len(df)
    print(f"count_diff_labels_in_ties: {count_diff_labels_in_ties}")
    print(f"pct_diff_labels_in_ties: {pct_diff_labels_in_ties:.6f} ({pct_diff_labels_in_ties*100:.4f}%)")

    return {
        "same_dist_count": same_dist_count,
        "same_dist_pct": same_dist_pct,
        "count_diff_labels_in_ties": count_diff_labels_in_ties,
        "pct_diff_labels_in_ties": pct_diff_labels_in_ties,
    }


def compare_retriever_rag(dfs, dataset_name=""):
    """
    compare the retriever used as a 1-NN classifier (its top-1 retrieved label) against the
    full RAG pipeline (the generator prediction), across one or more runs. the two compared
    methods are fixed and only the runs vary, so any ablation whose results are RAG CSVs fits
    here (duplicate removal, top-k, thresholds...).
    for each run, derives accuracy and macro F1, and saves a confusion matrix for both methods.
    dfs (dict): {run name: rag results dataframe}. the key labels the run, e.g. "before
                duplicate removal" / "after duplicate removal", or "top-1" / "top-2". it is
                used for output only, so it must be safe to put in a file name.
                each dataframe must contain "true_label", "top_label_1" and "prediction" cols.
    dataset_name (String): the dataset the runs come from, e.g. "FER+". every file written
                starts with it, so all the figures of one dataset sort and scan together.
    returns: nothing. writes 2 confusion matrices per run plus one comparison table, all as
             pdf files under "figures".
    """
    # calculate retriever and generator performance
    rows = []
    for name, df in dfs.items():
        # calculate retriever accuracy and RAG accuracy
        retriever_acc = accuracy_score(df["true_label"], df["top_label_1"])
        rag_acc = accuracy_score(df["true_label"], df["prediction"])
        retriever_f1 = f1_score(df["true_label"], df["top_label_1"], average="macro")
        rag_f1 = f1_score(df["true_label"], df["prediction"], average="macro")
        # append results to table
        rows.append(["Retriever (Top-1)", name, retriever_acc * 100, retriever_f1 * 100])
        rows.append(["RAG", name, rag_acc * 100, rag_f1 * 100])
        # print results
        print(f"results for run: {name}")
        print(f"retriever accuracy: {retriever_acc:.6f} ({retriever_acc*100:.2f}%)")
        print(f"rag accuracy: {rag_acc:.6f} ({rag_acc*100:.3f}%)")
        print(f"retriever Macro F1: {retriever_f1:.6f} ({retriever_f1*100:.2f}%)")
        print(f"rag Macro F1: {rag_f1:.6f} ({rag_f1*100:.3f}%)")

        # retriever confusion matrix
        plot_confusion_matrix(true_classes=df["true_label"],pred_classes=df["top_label_1"],formats=("pdf",),
                              plot_name=f"{dataset_name} - {name} - Retriever confusion matrix")

        # RAG confusion matrix
        plot_confusion_matrix(true_classes=df["true_label"],pred_classes=df["prediction"],formats=("pdf",),
                              plot_name=f"{dataset_name} - {name} - RAG confusion matrix")

    # visualize table
    # create dataframe
    table_df = pd.DataFrame(rows, columns=["Method", "Run", "Accuracy", "Macro F1"])

    # round numeric columns
    table_df["Accuracy"] = table_df["Accuracy"].round(2)
    table_df["Macro F1"] = table_df["Macro F1"].round(2)

    print(table_df)

    # visualize table and save as pdf
    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.8 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.1, 2.0)

    # optional: make header a bit bolder
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # save only pdf
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name} - Retriever and RAG comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)


# error analysis
def error_analysis_retrieval_cases(df, dataset_name=""):
    """
    create a table of retrieval cases, and the proportion of
    correct generator prediction under each retrieval case.
    df_path: path to csv results, after duplicate removal (no leakage)
    """
    total = len(df)
    # A1: top1 correct, top2 correct, generator correct
    case_A1 = ((df["top_label_1"] == df["true_label"]) &
               (df["top_label_2"] == df["true_label"]) &
               (df["prediction"] == df["true_label"])).sum()

    # A2: top1 correct, top2 correct, generator incorrect
    case_A2 = ((df["top_label_1"] == df["true_label"]) &
               (df["top_label_2"] == df["true_label"]) &
               (df["prediction"] != df["true_label"])).sum()

    # B1: top1 correct, top2 incorrect, generator correct
    case_B1 = ((df["top_label_1"] == df["true_label"]) &
               (df["top_label_2"] != df["true_label"]) &
               (df["prediction"] == df["true_label"])).sum()

    # B2: top1 correct, top2 incorrect. generator incorrect
    case_B2 = ((df["top_label_1"] == df["true_label"]) &
               (df["top_label_2"] != df["true_label"]) &
               (df["prediction"] != df["true_label"])).sum()

    # C1: top1 incorrect, top2 correct, generator correct
    case_C1 = ((df["top_label_1"] != df["true_label"]) &
               (df["top_label_2"] == df["true_label"]) &
               (df["prediction"] == df["true_label"])).sum()

    # C2: top1 incorrect, top2 correct, generator incorrect
    case_C2 = ((df["top_label_1"] != df["true_label"]) &
               (df["top_label_2"] == df["true_label"]) &
               (df["prediction"] != df["true_label"])).sum()

    # D1: top1 incorrect, top2 incorrect, generator correct
    case_D1 = ((df["top_label_1"] != df["true_label"]) &
               (df["top_label_2"] != df["true_label"]) &
               (df["prediction"] == df["true_label"])).sum()

    # D2: top1 incorrect, top2 incorrect, generator incorrect
    case_D2 = ((df["top_label_1"] != df["true_label"]) &
               (df["top_label_2"] != df["true_label"]) &
               (df["prediction"] != df["true_label"])).sum()


    # print percentages
    sum_cases = case_A1 + case_A2 + case_B1 + case_B2 + case_C1 + case_C2 + case_D1 + case_D2
    print("Sum of all cases:", sum_cases)
    print("Total:", total)

    print("Case A1 (Top-1 correct, Top-2 correct, generator correct):", round(case_A1 / total * 100, 2), "%")
    print("Case A2 (Top-1 correct, Top-2 correct, generator incorrect):", round(case_A2 / total * 100, 2), "%")

    print("Case B1 (Top-1 correct, Top-2 incorrect, generator correct):", round(case_B1 / total * 100, 2), "%")
    print("Case B2 (Top-1 correct, Top-2 incorrect, generator incorrect):", round(case_B2 / total * 100, 2), "%")

    print("Case C1 (Top-1 incorrect, Top-2 correct, generator correct):", round(case_C1 / total * 100, 2), "%")
    print("Case C2 (Top-1 incorrect, Top-2 correct, generator incorrect):", round(case_C2 / total * 100, 2), "%")

    print("Case D1 (Top-1 incorrect, Top-2 incorrect, generator correct):", round(case_D1 / total * 100, 2), "%")
    print("Case D2 (Top-1 incorrect, Top-2 incorrect, generator incorrect):", round(case_D2 / total * 100, 2), "%")


    # conditional totals per retriever scenario
    total_A = case_A1 + case_A2
    total_B = case_B1 + case_B2
    total_C = case_C1 + case_C2
    total_D = case_D1 + case_D2

    # helper to avoid division by zero
    def cond_pct(count, subtotal):
        return round(count / subtotal * 100, 2) if subtotal != 0 else 0


    # visualize table
    rows = [
        ["A1",
         "Top-1 correct, Top-2 correct,\ngenerator correct",
         case_A1,
         round(case_A1 / total * 100, 2),
         cond_pct(case_A1, total_A),
         "Retriever and generator both correct"],

        ["A2",
         "Top-1 correct, Top-2 correct,\ngenerator incorrect",
         case_A2,
         round(case_A2 / total * 100, 2),
         cond_pct(case_A2, total_A),
         "Generator failed despite both retrieved labels being correct"],

        ["B1",
         "Top-1 correct, Top-2 incorrect,\ngenerator correct",
         case_B1,
         round(case_B1 / total * 100, 2),
         cond_pct(case_B1, total_B),
         "Generator correct when only Top-1 is correct"],

        ["B2",
         "Top-1 correct, Top-2 incorrect,\ngenerator incorrect",
         case_B2,
         round(case_B2 / total * 100, 2),
         cond_pct(case_B2, total_B),
         "Generator failed despite correct Top-1 retrieval"],

        ["C1",
         "Top-1 incorrect, Top-2 correct,\ngenerator correct",
         case_C1,
         round(case_C1 / total * 100, 2),
         cond_pct(case_C1, total_C),
         "Generator correct when only Top-2 is correct"],

        ["C2",
         "Top-1 incorrect, Top-2 correct,\ngenerator incorrect",
         case_C2,
         round(case_C2 / total * 100, 2),
         cond_pct(case_C2, total_C),
         "Generator failed despite correct Top-2 retrieval"],

        ["D1",
         "Top-1 incorrect, Top-2 incorrect,\ngenerator correct",
         case_D1,
         round(case_D1 / total * 100, 2),
         cond_pct(case_D1, total_D),
         "Generator correct despite incorrect retrieval"],

        ["D2",
         "Top-1 incorrect, Top-2 incorrect,\ngenerator incorrect",
         case_D2,
         round(case_D2 / total * 100, 2),
         cond_pct(case_D2, total_D),
         "Both retriever and generator failed"]
    ]

    # dataframe
    table_df = pd.DataFrame(
        rows,
        columns=["Case", "Condition", "Count", "Percentage", "Conditional Percentage", "Interpretation"]
    )


    print(table_df)

    # visualize table and save as pdf
    fig, ax = plt.subplots(figsize=(18, 2.5 + 0.9 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.8)

    # make header bold
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # save pdf
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name}- Error_analysis_8_cases.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)





def compare_rag_vs_zeroshot_on_failure(df_rag, df_zero_shot, dataset_name=""):
    """
    df_zero_shot =
    :param df_rag: rag results after duplicate removal
    :param df_zero: zero shot results
    :param dataset_name:
    :return:
    """
    # case 4 - zero-shot vs RAG when retrieval fails
    # D1 + D2 in the RAG dataframe:
    # top1 incorrect and top2 incorrect
    mask_D = (
        (df_rag["top_label_1"] != df_rag["true_label"]) &
        (df_rag["top_label_2"] != df_rag["true_label"])
    )

    # keep only D1 + D2 samples from RAG
    df_rag_D = df_rag.loc[mask_D, ["file_path", "true_label", "prediction"]].copy()
    df_rag_D = df_rag_D.rename(columns={"prediction": "prediction_rag"})

    # keep matching samples from zero-shot
    df_zero_D = df_zero_shot.loc[:, ["file_path", "true_label", "prediction"]].copy()
    df_zero_D = df_zero_D.rename(columns={"prediction": "prediction_zero"})

    # merge by file_path
    df_compare = df_rag_D.merge(
        df_zero_D[["file_path", "prediction_zero"]],
        on="file_path",
        how="inner"
    )

    # total number of matched D1 + D2 samples
    total_D_cases = len(df_compare)

    print("Matched D1+D2 samples:", total_D_cases)
    print("Original D1+D2 samples in RAG:", len(df_rag_D))

    # correctness of each framework
    rag_correct = df_compare["prediction_rag"] == df_compare["true_label"]
    zero_correct = df_compare["prediction_zero"] == df_compare["true_label"]

    # 4 comparison cases
    case_1 = ((~rag_correct) & (zero_correct)).sum()   # RAG wrong, zero-shot correct
    case_2 = ((rag_correct) & (~zero_correct)).sum()   # RAG correct, zero-shot wrong
    case_3 = ((rag_correct) & (zero_correct)).sum()    # both correct
    case_4 = ((~rag_correct) & (~zero_correct)).sum()  # both wrong

    # sanity check
    sum_cases = case_1 + case_2 + case_3 + case_4
    print("Sum of all comparison cases:", sum_cases)
    print("Total D1+D2 samples:", total_D_cases)

    # print percentages
    print("RAG wrong, zero-shot correct:", round(case_1 / total_D_cases * 100, 2), "%")
    print("RAG correct, zero-shot wrong:", round(case_2 / total_D_cases * 100, 2), "%")
    print("Both correct:", round(case_3 / total_D_cases * 100, 2), "%")
    print("Both wrong:", round(case_4 / total_D_cases * 100, 2), "%")

    # create table
    rows = [
        ["RAG wrong, zero-shot correct", f"{round(case_1 / total_D_cases * 100, 2)}% ({case_1})"],
        ["RAG correct, zero-shot wrong", f"{round(case_2 / total_D_cases * 100, 2)}% ({case_2})"],
        ["Both correct", f"{round(case_3 / total_D_cases * 100, 2)}% ({case_3})"],
        ["Both wrong", f"{round(case_4 / total_D_cases * 100, 2)}% ({case_4})"],
    ]

    table_df = pd.DataFrame(rows, columns=["Condition", "Percentage"])

    print(table_df)

    # visualize table and save as pdf
    fig, ax = plt.subplots(figsize=(10, 2.5 + 0.9 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.8)

    # make header bold
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # save pdf
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name}- RAG_vs_ZeroShot_on_D1_D2.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)


def compare_zeroshot_rag(dfs, dataset_name=""):
    """
    compare zero-shot against the full RAG pipeline for the same generator, across one or more
    models. the two compared methods are fixed and only the models vary, so any pair of runs
    that differ in nothing but the framework fits here.
    for each model, derives accuracy and macro F1 of both frameworks and the gap between them,
    always RAG minus zero-shot, so a positive gap means retrieval helped and a reader can see
    at a glance which generators retrieval helps and which it hurts. the gap is written on the
    RAG row of the pair, the row it is measured from.
    the two runs of a pair are separate jobs over the same test set, so they are checked with
    is_same_dataset() first: a pair whose rows do not line up is reported and skipped, since
    scoring it would compare two different sample sets and read as a difference in framework.
    dfs (dict): {model name: (zero-shot dataframe, rag dataframe)}. the key labels the model,
                e.g. "gemma_3_27b" / "llava_next_34b". it is used for output only, so it must
                be safe to put in a file name. both dataframes must contain "true_label",
                "prediction" and "file_path".
    dataset_name (String): the dataset the runs come from, e.g. "FER+". every file written
                starts with it, so all the figures of one dataset sort and scan together.
    returns: nothing. writes one comparison table as a pdf file under "figures".
    """
    # calculate zero-shot and RAG performance
    rows = []
    for name, (df_zero_shot, df_rag) in dfs.items():
        print(f"results for model: {name}")

        # the two frameworks ran as separate jobs, so report whether they scored the same rows.
        if not is_same_dataset(df_zero_shot, df_rag,
                               name_a=f"{name} zero-shot", name_b=f"{name} RAG"):
            warnings.warn(
                f"the zero-shot and RAG runs of '{name}' do not hold the same rows "
                f"({len(df_zero_shot)} zero-shot, {len(df_rag)} RAG), so the two are scored "
                f"over different samples and the gap between them carries that difference. "
                f"the model is kept in the table, see the printed report above for what "
                f"differs.",
                UserWarning,
            )

        # calculate zero-shot accuracy and RAG accuracy. rounded to 2 decimals here, at the
        # point they are derived, and not only where they are printed: every number this
        # function reports is then the same number, so the gap below is exactly the difference
        # between the two values the table shows, and a reader recomputing it by hand agrees.
        zero_shot_acc = round(accuracy_score(df_zero_shot["true_label"],
                                             df_zero_shot["prediction"]) * 100, 2)
        rag_acc = round(accuracy_score(df_rag["true_label"], df_rag["prediction"]) * 100, 2)
        zero_shot_f1 = round(f1_score(df_zero_shot["true_label"], df_zero_shot["prediction"],
                                      average="macro") * 100, 2)
        rag_f1 = round(f1_score(df_rag["true_label"], df_rag["prediction"],
                                average="macro") * 100, 2)

        # what retrieval was worth for this generator, always RAG minus zero-shot, so a positive
        # gap means retrieval helped and the sign is the point. rounded again because
        # subtracting two rounded values can leave a float artefact (0.1+0.2 is not 0.3)
        acc_gap = round(rag_acc - zero_shot_acc, 2)
        f1_gap = round(rag_f1 - zero_shot_f1, 2)

        # append results to table. the gap belongs to the pair rather than to either row, so it
        # is written on the RAG row, the one it is measured from, and left blank on the
        # zero-shot row instead of repeating the same value twice
        rows.append([name, "Zero-shot", zero_shot_acc, zero_shot_f1, "", ""])
        rows.append([name, "RAG", rag_acc, rag_f1, acc_gap, f1_gap])

        # print results
        print(f"zero-shot accuracy: {zero_shot_acc}%")
        print(f"rag accuracy: {rag_acc}%")
        print(f"zero-shot Macro F1: {zero_shot_f1}%")
        print(f"rag Macro F1: {rag_f1}%")
        print(f"RAG - zero-shot: {acc_gap} accuracy points, {f1_gap} macro F1 points")

    if not rows:
        warnings.warn("no model could be scored, so no table was written.", UserWarning)
        return

    # visualize table
    # create dataframe
    # the values were rounded where they were derived, so nothing is rounded here
    table_df = pd.DataFrame(rows, columns=["Model", "Method", "Accuracy", "Macro F1",
                                           "Accuracy gap\n(RAG - Zero-shot)",
                                           "Macro F1 gap\n(RAG - Zero-shot)"])

    print(table_df)

    # visualize table and save as pdf
    fig, ax = plt.subplots(figsize=(14, 2.2 + 0.8 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.1, 2.4)

    # optional: make header a bit bolder
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # save only pdf
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name} - Zero-shot and RAG comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)


def error_analysis_evidence_regimes(df, dataset_name="", top_k=None):
    """
    the same question the 8-case table asks, but asked in a way that survives a growing k.

    error_analysis_retrieval_cases enumerates every combination of the retrieved labels, so it
    needs 2^k * 2 rows: fine at k=2 (8 rows), unreadable at k=5 (64) and useless at k=8 (512).
    this collapses the k retrieved labels into the only distinction the top-k ablation is about -
    whether the evidence AGREES with itself - and keeps the table at 6 rows for every k, so the
    tables of different k can be read side by side.

    the three regimes, from the k retrieved labels and the query's true label:

      High retrieval  - every retrieved label IS the true label. clean, unanimous evidence.
      Conflicting     - at least one retrieved label is the true label and at least one is not.
                        the evidence points in more than one direction, and the right answer is
                        somewhere in it.
      Low retrieval   - no retrieved label is the true label. the right answer is not in the
                        evidence at all, so the generator can only get there on its own.

    the three are mutually exclusive and cover every row, at any k.

    the number to read is the Conditional Percentage of the "generator correct" row of the
    Conflicting regime: of the queries where the evidence disagreed, how often the generator still
    landed on the truth. that is the quantity the ablation exists to move. comparing it across k
    answers the actual hypothesis - whether a model that sees more examples handles contradictory
    evidence better than one that sees two.

    df: results dataframe, already validated, with top_label_1..top_label_k columns.
    dataset_name (String): labels the printed output and prefixes the file written.
    top_k (int): use only the first top_k label columns. None (default) uses every one present,
        which is what you want unless you are deliberately re-scoring a k=5 run as if it were k=3.

    returns: the table as a dataframe, so several runs can be stacked into a cross-k comparison.
        also writes the table as a pdf under "figures".
    """
    # the label columns, ordered by their number and not by string order, so that top_label_10
    # sorts after top_label_9 rather than after top_label_1.
    label_cols = [c for c in df.columns
                  if c.startswith("top_label_") and c[len("top_label_"):].isdigit()]
    label_cols.sort(key=lambda c: int(c[len("top_label_"):]))
    if top_k is not None:
        label_cols = label_cols[:top_k]
    k = len(label_cols)
    if k == 0:
        raise ValueError("no top_label_* columns in the dataframe")

    labels = df[label_cols]
    true_label = df["true_label"]

    # rag.py pads the neighbour columns with None when the index returns fewer than k, so a row
    # can hold fewer than k labels. count what was actually retrieved rather than assuming k,
    # otherwise a padded row could never be "every retrieved label is correct".
    retrieved = labels.notna()
    n_retrieved = retrieved.sum(axis=1)
    # a None pads to NaN, and NaN == true_label is False, so padding never counts as a match
    n_correct = labels.eq(true_label, axis=0).sum(axis=1)

    # a row with nothing retrieved has no evidence to classify, so it is out of the table
    has_evidence = n_retrieved > 0
    n_empty = int((~has_evidence).sum())
    if n_empty:
        warnings.warn(f"{n_empty} rows retrieved no neighbours at all and are excluded")

    high = has_evidence & (n_correct == n_retrieved)
    low = has_evidence & (n_correct == 0)
    conflicting = has_evidence & (n_correct > 0) & (n_correct < n_retrieved)

    generator_correct = df["prediction"] == true_label
    total = int(has_evidence.sum())

    def cond_pct(count, subtotal):
        return round(count / subtotal * 100, 2) if subtotal != 0 else 0

    regimes = [
        ("H", "High retrieval", high,
         f"all {k} retrieved labels\nare the true label",
         "Generator kept the answer the evidence already agreed on",
         "Generator lost an answer that every example handed it"),
        ("K", "Conflicting", conflicting,
         "retrieved labels disagree,\nthe true label is among them",
         "Generator resolved the conflict correctly",
         "Generator was pulled off by the contradicting examples"),
        ("L", "Low retrieval", low,
         f"none of the {k} retrieved\nlabels is the true label",
         "Generator overrode the evidence and was right anyway",
         "Generator followed evidence that did not contain the answer"),
    ]

    rows = []
    for code, regime_name, mask, condition, interp_correct, interp_wrong in regimes:
        subtotal = int(mask.sum())
        for suffix, correct, interpretation in [("1", True, interp_correct),
                                                ("2", False, interp_wrong)]:
            count = int((mask & (generator_correct == correct)).sum())
            rows.append([
                code + suffix,
                f"{regime_name},\ngenerator {'correct' if correct else 'incorrect'}",
                condition,
                count,
                round(count / total * 100, 2) if total else 0,
                cond_pct(count, subtotal),
                interpretation,
            ])

    table_df = pd.DataFrame(
        rows,
        columns=["Case", "Regime", "Condition", "Count", "Percentage",
                 "Conditional Percentage", "Interpretation"]
    )

    print(f"{dataset_name} - evidence regimes at k={k}")
    print("Rows with evidence:", total, "of", len(df))
    for code, regime_name, mask, _, _, _ in regimes:
        subtotal = int(mask.sum())
        print(f"  {regime_name}: {subtotal} rows ({cond_pct(subtotal, total)}% of all), "
              f"generator correct on {cond_pct(int((mask & generator_correct).sum()), subtotal)}% "
              f"of them")

    # within Low retrieval, evidence that is unanimously wrong misleads far more consistently than
    # evidence that is merely scattered, so the split is worth seeing even though both are "Low".
    unanimous = labels.nunique(axis=1, dropna=True) == 1
    unanimous_wrong = low & unanimous
    scattered_wrong = low & ~unanimous
    print(f"  (of Low retrieval: {int(unanimous_wrong.sum())} rows unanimously wrong, generator "
          f"correct on {cond_pct(int((unanimous_wrong & generator_correct).sum()), int(unanimous_wrong.sum()))}%"
          f" | {int(scattered_wrong.sum())} rows scattered wrong, generator correct on "
          f"{cond_pct(int((scattered_wrong & generator_correct).sum()), int(scattered_wrong.sum()))}%)")
    print(table_df)

    # visualize table and save as pdf
    fig, ax = plt.subplots(figsize=(20, 2.5 + 0.9 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.8)

    # make header bold
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # save pdf
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name}- Evidence_regimes.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)

    return table_df


def _neighbour_columns(df, top_k=None):
    """
    the top_label_* / top_cosine_* columns of a rag results dataframe, ordered by neighbour
    number rather than by string order, so top_label_10 sorts after top_label_9 and not after
    top_label_1.
    top_k (int): keep only the first top_k neighbours. None uses every one present.
    returns: (label columns, cosine columns), the two aligned by position.
    """
    def numbered(prefix):
        cols = [c for c in df.columns
                if c.startswith(prefix) and c[len(prefix):].isdigit()]
        cols.sort(key=lambda c: int(c[len(prefix):]))
        return cols[:top_k] if top_k is not None else cols

    label_cols = numbered("top_label_")
    cosine_cols = numbered("top_cosine_")
    if not label_cols:
        raise ValueError("no top_label_* columns in the dataframe")
    return label_cols, cosine_cols[:len(label_cols)]


def knn_predict(df, top_k=None, voting="cosine"):
    """
    turn the retrieved neighbours into a k-NN classifier prediction, one label per row.

    compare_retriever_rag uses top_label_1 alone, which is 1-NN and ignores k entirely. that is
    the right baseline for asking "is RAG worth it over the simplest model", but it is the wrong
    one for asking "does the generator do anything with the extra examples": at k=5 the generator
    sees 5 neighbours and 1-NN sees 1, so the comparison is not information-matched. this gives
    the retriever the same evidence the generator got.

    voting (String): how the k labels become one prediction.
        "cosine"   - each neighbour votes with its cosine similarity, and the label with the
                     largest total wins. the default: it is defined at every k, and it degrades
                     smoothly to 1-NN rather than needing an arbitrary rule.
        "majority" - each neighbour votes once, ties broken by the nearest neighbour among the
                     tied labels. plain k-NN, and the thing to check the default against.

    ties are broken by neighbour order (nearest first) under both rules, so the result never
    depends on column order or on dictionary ordering.

    df: results dataframe with top_label_1..k and, for "cosine", top_cosine_1..k.
    top_k (int): use only the first top_k neighbours. None (default) uses every one present.
    returns: a Series of predicted labels, indexed like df.
    """
    if voting not in ("cosine", "majority"):
        raise ValueError(f"voting must be 'cosine' or 'majority', got {voting!r}")

    label_cols, cosine_cols = _neighbour_columns(df, top_k)
    if voting == "cosine" and len(cosine_cols) != len(label_cols):
        raise ValueError("cosine voting needs a top_cosine_* column per top_label_* column")

    labels = df[label_cols].to_numpy()
    weights = df[cosine_cols].to_numpy() if voting == "cosine" else None

    predictions = []
    for row in range(len(df)):
        # totals per label, and the position of the nearest neighbour carrying that label, so a
        # tie can be settled by distance instead of arbitrarily
        scores = {}
        first_seen = {}
        for pos in range(len(label_cols)):
            label = labels[row, pos]
            # rag.py pads with None when the index returned fewer than k neighbours; a padded
            # slot is not a vote
            if pd.isna(label):
                continue
            if voting == "cosine":
                weight = weights[row, pos]
                if pd.isna(weight):
                    continue
                # cosines here are all positive, but a negative one would otherwise count as a
                # vote AGAINST its own label, so floor it at zero
                weight = max(float(weight), 0.0)
            else:
                weight = 1.0
            scores[label] = scores.get(label, 0.0) + weight
            if label not in first_seen:
                first_seen[label] = pos
        if not scores:
            predictions.append(None)
            continue
        # highest total wins; on a tie the label whose nearest neighbour came first
        predictions.append(max(scores, key=lambda label: (scores[label], -first_seen[label])))

    return pd.Series(predictions, index=df.index, name=f"knn_{voting}_prediction")


def compare_knn_retriever_rag(dfs, dataset_name="", voting="cosine", top_k=None,
                              include_1nn=True):
    """
    the same comparison compare_retriever_rag makes, with the retriever scored as a k-NN
    classifier over all of its retrieved neighbours instead of as 1-NN over the top one.

    k is read from each dataframe, so a top_k=3 run is scored as 3-NN and a top_k=5 run as 5-NN.
    that is the point: RAG and the retriever then see exactly the same evidence in every run, and
    the gap between them is what the GENERATOR added rather than what the extra retrieval added.

    1-NN is kept in the table as well (include_1nn), because it is the only baseline that does not
    move with k. it gives a flat reference line the k-NN and RAG curves can both be read against.

    reading the result:
      RAG minus k-NN widens with k -> the generator genuinely exploits the extra examples.
      RAG tracks k-NN               -> the gain came from retrieval, and the generator is
                                       an expensive way to take a vote.

    dfs (dict): {run name: rag results dataframe}. the key labels the run and goes into file
                names. each dataframe needs "true_label", "prediction", top_label_1..k and,
                for cosine voting, top_cosine_1..k.
    dataset_name (String): the dataset the runs come from, e.g. "FER+". every file written
                starts with it.
    voting (String): "cosine" (default) or "majority", passed to knn_predict.
    top_k (int): score every run with this many neighbours instead of all of them. None
                (default) uses each run's own k, which is what the ablation wants.
    include_1nn (bool): also report the top-1 label as a fixed reference row.
    returns: the comparison table as a dataframe. writes one confusion matrix per method per run
             plus one comparison table, all as pdf files under "figures".
    """
    rows = []
    for name, df in dfs.items():
        label_cols, _ = _neighbour_columns(df, top_k)
        k = len(label_cols)
        knn_name = f"Retriever ({k}-NN, {voting})"

        knn_pred = knn_predict(df, top_k=top_k, voting=voting)
        # a row where nothing was retrieved has no k-NN prediction and cannot be scored
        scorable = knn_pred.notna()
        if not scorable.all():
            warnings.warn(f"{name}: {int((~scorable).sum())} rows had no neighbours to vote "
                          f"with and are excluded from the {k}-NN score")
        true_label = df["true_label"][scorable]

        knn_acc = accuracy_score(true_label, knn_pred[scorable])
        knn_f1 = f1_score(true_label, knn_pred[scorable], average="macro")
        rag_acc = accuracy_score(df["true_label"], df["prediction"])
        rag_f1 = f1_score(df["true_label"], df["prediction"], average="macro")

        rows.append([knn_name, name, knn_acc * 100, knn_f1 * 100])
        if include_1nn:
            nn1_acc = accuracy_score(df["true_label"], df[label_cols[0]])
            nn1_f1 = f1_score(df["true_label"], df[label_cols[0]], average="macro")
            rows.append(["Retriever (1-NN)", name, nn1_acc * 100, nn1_f1 * 100])
        rows.append(["RAG", name, rag_acc * 100, rag_f1 * 100])

        print(f"results for run: {name}  (k={k}, voting={voting})")
        print(f"{k}-NN retriever accuracy: {knn_acc:.6f} ({knn_acc*100:.2f}%)")
        if include_1nn:
            print(f"1-NN retriever accuracy: {nn1_acc:.6f} ({nn1_acc*100:.2f}%)")
        print(f"rag accuracy: {rag_acc:.6f} ({rag_acc*100:.3f}%)")
        print(f"{k}-NN retriever Macro F1: {knn_f1:.6f} ({knn_f1*100:.2f}%)")
        if include_1nn:
            print(f"1-NN retriever Macro F1: {nn1_f1:.6f} ({nn1_f1*100:.2f}%)")
        print(f"rag Macro F1: {rag_f1:.6f} ({rag_f1*100:.3f}%)")
        # how often the generator simply reproduced the vote: if this is high, the generator is
        # not reasoning over the examples so much as counting them
        agreement = (df["prediction"][scorable] == knn_pred[scorable]).mean()
        print(f"RAG agrees with the {k}-NN vote on {agreement*100:.2f}% of rows")

        # k-NN retriever confusion matrix
        plot_confusion_matrix(true_classes=true_label, pred_classes=knn_pred[scorable],
                              formats=("pdf",),
                              plot_name=f"{dataset_name} - {name} - {k}-NN retriever confusion matrix")

        # RAG confusion matrix
        plot_confusion_matrix(true_classes=df["true_label"], pred_classes=df["prediction"],
                              formats=("pdf",),
                              plot_name=f"{dataset_name} - {name} - RAG confusion matrix")

    # visualize table
    table_df = pd.DataFrame(rows, columns=["Method", "Run", "Accuracy", "Macro F1"])
    table_df["Accuracy"] = table_df["Accuracy"].round(2)
    table_df["Macro F1"] = table_df["Macro F1"].round(2)

    print(table_df)

    fig, ax = plt.subplots(figsize=(11, 2.2 + 0.8 * len(table_df)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.1, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{dataset_name} - kNN retriever and RAG comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

    print(f"Saved: {out_path}")
    plt.close(fig)

    return table_df
