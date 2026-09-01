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
