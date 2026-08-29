"""
Analysis functions for RAG results: retriever (1-NN) vs RAG comparison, retrieval-case
error analysis, RAG vs zero-shot on retrieval failures, and retrieval tie statistics.

Working copy of the functions in ablation/visualize_results_part2.py. The originals stay
frozen under ablation/ because they produced the figures in the submitted draft.

Every function takes dataframes (not paths), and writes its figures to a relative
"figures" folder, so the output lands next to whichever experiment script called it.
Validate each results CSV with validate_results_df() before passing it in, and use
is_same_dataset() to confirm two results dataframes cover the same samples.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import warnings

from .vis_results import plot_confusion_matrix


def is_same_dataset(df_a, df_b, name_a="", name_b=""):
    """
    check that two results dataframes cover the same samples, by comparing their sets of
    "file_path" values. a mismatch warns, since comparisons that merge the two on
    "file_path" silently drop the unmatched rows.
    df_a, df_b (DataFrame): results dataframes, must contain "file_path".
    name_a, name_b (String): labels for the printed output only.
    returns: True if both hold the same set of file paths.
    """
    label_a = name_a or "first dataframe"
    label_b = name_b or "second dataframe"

    paths_a = set(df_a["file_path"])
    paths_b = set(df_b["file_path"])
    is_match = paths_a == paths_b

    print(f"do '{label_a}' ({len(df_a)} rows) and '{label_b}' ({len(df_b)} rows) "
          f"cover the same set of file paths? {is_match}")

    if is_match:
        print("note: this only confirms both dataframes hold the exact same file paths. "
              "row order and duplicated file paths are not checked - "
              "if either matters for your comparison, check it yourself.")

    if not is_match:
        only_a = sorted(paths_a - paths_b)
        only_b = sorted(paths_b - paths_a)
        warnings.warn(
            f"'{label_a}' and '{label_b}' do not cover the same samples - comparisons that "
            f"merge them on 'file_path' will drop those rows. "
            f"file paths that are only in '{label_a}' ({len(only_a)}): {only_a}. "
            f"file paths that are only in '{label_b}' ({len(only_b)}): {only_b}.",
            UserWarning,
        )

    return is_match


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
    gets a dictionary with keys (names of figures) values (datasets).
    calculate accuracy and macro F1 of the retriever classifier and RAG.
    creates confusion metrix for each and saves under figure folder.
    :param dfs:
    :return:
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
        print(name)
        print(f"retriever accuracy: {retriever_acc:.6f} ({retriever_acc*100:.2f}%)")
        print(f"rag accuracy: {rag_acc:.6f} ({rag_acc*100:.3f}%)")
        print(f"retriever Macro F1: {retriever_f1:.6f} ({retriever_f1*100:.2f}%)")
        print(f"rag Macro F1: {rag_f1:.6f} ({rag_f1*100:.3f}%)")

        # retriever confusion matrix
        plot_confusion_matrix(true_classes=df["true_label"],pred_classes=df["top_label_1"],formats=("pdf",),
                              plot_name=f"{name}-{dataset_name} - Retriever confusion matrix")

        # RAG confusion matrix
        plot_confusion_matrix(true_classes=df["true_label"],pred_classes=df["prediction"],formats=("pdf",),
                              plot_name=f"{name}-{dataset_name} - RAG confusion matrix")

    # visualize table
    # create dataframe
    table_df = pd.DataFrame(rows, columns=["Method", "Duplicate removal", "Accuracy", "Macro F1"])

    # create duplicate-removal column
    #table_df["Duplicate removal"] = table_df["Duplicate removal"].replace({
    #    "before duplicate removal": "-",
    #    "after duplicate removal": "V"
    #})

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
    out_path = os.path.join("figures", f"{dataset_name}- Retriever and RAG comparison (before and after duplicates).pdf")
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
