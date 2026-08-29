"""
functions to visualize classification results:
plot classification report -> get table plot of recall, precision, f1, accuracy (overall and per class).
hist_visualization - > gets a df  (must have a column name "true_label") and a title for the plot.
                        and plots a histogram to shoe the data distribution.

"""

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from PIL import Image
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import random
import textwrap



def validate_results(df):
    # validate results
    # if anger is in prediction, normalize to angry
    df.loc[df["prediction"] == "anger", "prediction"] = "angry"
    # check if all predictions are valid
    print(f"label list: {sorted(df['true_label'].unique().tolist())}")
    print(f"prediction list: {sorted(df['prediction'].unique().tolist())}")
    print(f"same unique labels? {set(df['true_label'].dropna().unique()) == set(df['prediction'].dropna().unique())}")
    # are there nulls?
    print(f"are there nulls in true label? {df['true_label'].isna().any()}")
    print(f"are there nulls in prediction? {df['prediction'].isna().any()}")
    # print columns
    print(f"columns: {df.columns.tolist()}")
    # validate file paths
    all_match = (df["file_path"].astype(str) == df["query_file_path"].astype(str)).all()
    print(f"does file paths match? {all_match}")

    return df


def ambiguity_distribution(df, plot_name):
    if "votes_percentage" not in df.columns:
        raise ValueError("Column 'votes_percentage' not found in df.")

    vp = df["votes_percentage"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vp, bins=30)
    ax.set_xlabel("votes_percentage")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of votes_percentage")

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{plot_name}.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path

def plot_ambiguity_image(df, seed=42, file_name=""):

    # ambiguity bins
    bins = [
        ("Low agreement", 0.0, 0.335),
        ("Medium agreement", 0.50, 0.6),
        ("High agreement", 0.85, None)
    ]

    random.seed(seed)

    results = []
    for bin_name, low, high in bins:
        vp = df["votes_percentage"]
        # build mask
        if high is None:
            mask = (vp >= low)  # 0.80 included and above
        elif bin_name == "Low agreement":
            mask = (vp >= low) & (vp <= high)  # 0.0–0.50 included
        else:
            mask = (vp > low) & (vp < high)

        # filter df by ambiguity
        bin_df = df[mask]

        n_img = 3
        sampled = bin_df.sample(n=n_img, random_state=seed)

        bin_items = []
        for _, row in sampled.iterrows():
            img_path = row["file_path"]
            img = Image.open(img_path).convert("RGB")
            bin_items.append((img, row["true_label"], float(row["votes_percentage"])))
        results.append({"bin": bin_name, "items": bin_items})

    n_rows, n_cols = 3, len(results)
    # visualize
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 8.4))

    # if only 1 label, axes shape is different
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, bin_data in enumerate(results):
        bin_name = bin_data["bin"]
        items = bin_data["items"]

        # title (bold) above the column
        axes[0, col].set_title(bin_name, fontweight="bold", fontsize=22, pad=15)

        for row in range(n_rows):
            ax = axes[row, col]
            ax.axis("off")

            if row < len(items):
                img, true_label, vote_pct = items[row]
                ax.imshow(img)

                # text below each image: label + percent
                ax.text(
                    0.5, -0.08,
                    f"{str(true_label).capitalize()} ({vote_pct * 100:.1f}%)",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=20
                )
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center", fontsize=12)

    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{file_name}.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path}")







def class_distribution_table(dfs, label_col="true_label", file_name="", show_removed_percentage=False):
    """
    dfs: dict. "split": dataframe
    label_col: default "true_label"
    show_removed_percentage: False. whether adding a column "removed samples (%)".
    """
    first_key = next(iter(dfs))
    first_split = dfs[first_key]
    classes_list = sorted(first_split['true_label'].unique().tolist())
    reference_total = len(first_split)

    rows = []
    for split, df in dfs.items():
        counts = df[label_col].value_counts(dropna=True)
        perc = df[label_col].value_counts(normalize=True, dropna=True) * 100
        split_wrapped = "\n".join(textwrap.wrap(str(split), width=12))
        current_total = int(counts.sum())
        row = {"split": split_wrapped, "total": current_total}

        if show_removed_percentage:
            removed_percentage = ((reference_total - current_total) / reference_total) * 100
            row["removed samples (%)"] = f"{removed_percentage:.2f}%"

        # fill class columns
        for cls in classes_list:
            count = int(counts.get(cls,0))
            percent = float(perc.get(cls,0.0))
            row[cls] =f"{count}\n({percent:.2f}%)"


        rows.append(row)

    # create a dataframe with number of rows and the cols
    columns = ["split"] + classes_list + ["total"]
    if show_removed_percentage:
        columns.append("removed samples (%)")
    table_df = pd.DataFrame(rows, columns=columns)

    # visualize table
    fig_w = max(8.0, 1.2 * len(table_df.columns))
    fig_h = max(2.6, 1.2 * (len(table_df) + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table_vis = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table_vis.auto_set_font_size(False)
    table_vis.set_fontsize(12)
    table_vis.scale(1.5, 3.0)
    plt.tight_layout()

    # save to pdf
    os.makedirs("figures", exist_ok=True)

    out_path = os.path.join("figures", f"{file_name}.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path}")

    return table_df







def plot_metrics(y_true, y_pred, digits=2, plot_name="", classes_list=None, formats=("pdf", "png")):
    """
    classes_list: if given, use it. if not, create classes list using true label column.
    """
    if classes_list is None:
        classes_list = sorted(y_true.unique().tolist())

    # create class report
    report = classification_report(
        y_true, y_pred,
        labels=classes_list,
        output_dict=True,
        zero_division=0
    )
    print(report)
    df = pd.DataFrame(report).T

    # Add a dedicated column for overall accuracy (so it won't get wiped)
    df["accuracy_pct"] = np.nan
    if "accuracy" in df.index:
        acc_col = df.loc["accuracy"].first_valid_index()  # where sklearn put the accuracy number
        df.loc["accuracy", "accuracy_pct"] = float(df.loc["accuracy", acc_col]) * 100
        df.loc["accuracy", ["precision", "recall", "f1-score"]] = np.nan
        df.loc["accuracy", "support"] = len(y_true)


    # TP
    classes = [c for c in df.index if c not in ["accuracy", "macro avg", "weighted avg"]]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm)
    N = len(y_true)

    # Add extra columns for class rows
    df.loc[classes, "tp"] = tp
    df.loc[classes, "support_pct"] = (df.loc[classes, "support"] / N) * 100
    df.loc[classes, "acc_contrib_pct"] = (tp / N) * 100  # TP / total samples

    # Convert report metrics to percentages (0–100)
    metric_cols = ["precision", "recall", "f1-score"]
    df.loc[:, metric_cols] = df.loc[:, metric_cols].astype(float) * 100

    df = df.round(digits)

    df = df.rename(columns={
        "support_pct": "support\npercentage",
        "acc_contrib_pct": "accuracy\ncontrib(%)",
        "accuracy_pct": "total\naccuracy(%)"
    })
    # Choose columns to be presented LaTeX table
    # cols = ["precision", "recall", "f1-score", "support", "support_pct", "acc_contrib_pct", "tp"]
    # df = df[cols]

    fig, ax = plt.subplots(figsize=(14, 0.8*len(df)+2))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.0, 1.95)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.05)

    # save
    os.makedirs("figures", exist_ok=True)
    saved_paths = []
    for f in formats:
        f = f.lstrip(".").lower()
        out_path = os.path.join("figures", f"{plot_name}.{f}")
        if f == "png":
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
        saved_paths.append(out_path)

    for p in saved_paths:
        print(f"Saved: {p}")

    return df





# def plot_classification_report(true_class_list, predicted_class_list, classes_list):
def plot_classification_report(true_classes_column, pred_classes_column, plot_name, formats=("pdf", "png")):
    """
    Gets lists of true labels and predicted labels length N (the data),
    creates a plot table of classification report and returns it.
    true_class (Series/list/array): list of all samples (their true labels).
    predicted_class (Series/list/array): list of all samples (their predicted label by the model).
    title_text (String): the tile of the plot.
    returns: plotly table (classification report).
    """
    class_report = classification_report(true_classes_column, pred_classes_column,
                                         zero_division=0, output_dict=True)


    # for each key, loop through it keys and print the key1 , key2s and the values. just to understand the structure.
    for key1, key2_dict in class_report.items():
        print(f'Values of {key1}: {key2_dict}')

    # Calculate total instances and matches for TP calculation
    total_instances = len(true_classes_column)
    matches = Counter((t, p) for t, p in zip(true_classes_column, pred_classes_column))

    accuracies = []
    true_positives = []
    for class_name in class_report.keys():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        if matches.get((class_name, class_name)) is not None:
            true_positive = matches[(class_name, class_name)]
        else:
            true_positive = 0
        true_positives.append(true_positive)
        accuracy = true_positive / total_instances
        accuracies.append(accuracy)

    # Now create rows, combining class metrics and summary metrics in one loop
    rows = []
    for idx, (class_name, metric_dict) in enumerate(class_report.items()):
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            # Row for actual classes
            row = {
                'Class': class_name,
                'Precision': metric_dict['precision'],
                'Recall': metric_dict['recall'],
                'F1_Score': metric_dict['f1-score'],
                'Accuracy': accuracies[idx],
                'TP': true_positives[idx],
                'Support': f"{round((metric_dict['support']/total_instances)*100,3)}%<br>({int(metric_dict['support'])})",
                'Support (percentage)': (metric_dict['support'] / total_instances) * 100
                # 'Support': metric_dict['support'],
                # 'Support_percentage': (metric_dict['support'] / total_instances) * 100,
            }
        elif class_name in ['macro avg', 'weighted avg']:
            # Row for macro avg and weighted avg
            row = {
                'Class': class_name,
                'Precision': metric_dict['precision'],
                'Recall': metric_dict['recall'],
                'F1_Score': metric_dict['f1-score'],
                'Accuracy': None,
                'TP': None,
                'Support': f"100%\n({round(total_instances,0)})",
                'Support (percentage)': None,
                # 'Support': total_instances,
                # 'Support_percentage': None,
            }
        else:
            # Row for total accuracy (class_name == "accuracy":)
            row = {
                'Class': 'Total Accuracy',
                'Precision': None,
                'Recall': None,
                'F1_Score': None,
                'Accuracy': metric_dict,
                'TP': None,
                'Support': f"100%\n({round(total_instances,0)})",
                'Support (percentage)': None,
                # 'Support': total_instances,
                # 'Support_percentage': None,
            }
        rows.append(row)

    # Create DataFrame
    class_report_df = pd.DataFrame(rows)
    class_report_df = class_report_df.round(4)

    # Convert selected metrics to percentages (0–100) and keep 2 decimals
    percent_cols = ['Precision', 'Recall', 'F1_Score', 'Accuracy']
    class_report_df[percent_cols] = class_report_df[percent_cols].astype(float) * 100
    class_report_df[percent_cols] = class_report_df[percent_cols].round(2)

    # sort the rows of classes by their proportion in the data in descending order
    # all rows that are actual classes
    classes_df = class_report_df[~class_report_df['Class'].isin(['Total Accuracy', 'macro avg', 'weighted avg'])]
    # all rows that are not classes
    summary_df = class_report_df[class_report_df['Class'].isin(['Total Accuracy', 'macro avg', 'weighted avg'])]
    # sort
    classes_df = classes_df.sort_values(by='Support (percentage)', ascending=False)
    # combine the dfs again now without this column
    class_report_df = pd.concat([classes_df, summary_df], ignore_index=True)
    class_report_df.drop(columns='Support (percentage)', inplace=True)

    # Create a Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{col}</b>" for col in class_report_df.columns],
                    align='center',
                    font=dict(color='black', size=14)),
        cells=dict(values=[class_report_df[col] for col in class_report_df.columns],
                   align='center',
                   height=25,
                   font=dict(size=14))
                   # font=dict(color=colors_list, size=14))
    )])

    fig.update_layout(template='plotly_white',
                      # title="Classification Report",
                      title=f"<b>{plot_name}</b>",
                      title_x=0.5,
                      title_y=0.91,
                      title_font=dict(size=22, weight='bold'),
                      height=850,
                      width=1100
                      )

    # save pdf
    os.makedirs("figures", exist_ok=True)
    saved_paths = []
    for f in formats:
        f = f.lstrip(".").lower()
        out_path = os.path.join("figures", f"{plot_name}.{f}")
        if f in ("pdf", "png", "svg", "jpeg", "jpg", "webp"):
            fig.write_image(out_path)   # requires kaleido
        elif f == "html":
            fig.write_html(out_path)
        else:
            raise ValueError(f"Unsupported format: {f}")
        saved_paths.append(out_path)

    for p in saved_paths:
        print(f"Saved: {p}")

    return fig

# confusion matrix
def plot_confusion_matrix(true_classes,pred_classes,classes_list=None, plot_name="", formats=("pdf","png")):
    """
    Gets true labels and prediction lists, creates a confusion matrix
    and returns the plot.
    true_classes_column (Series/list/array):
    pred_classes_column (Series/list/array):
    title_text (String):
    is_normalize (Boolean): if False shows counts, if true shows percentage out of rows
                            (out of each class).
    returns: confusion matrix plotly plot
    """

    if classes_list is None:
        classes_list = sorted(true_classes.unique().tolist())
    cm = confusion_matrix(y_true=true_classes, y_pred=pred_classes,
                          labels=classes_list, normalize="true")

    cm_rounded = cm.round(5)
    for i, cls in enumerate(classes_list):
        print(f"{cls}:")
        print(cm_rounded[i].tolist())


    cm = cm * 100

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
        xticklabels=classes_list,
        yticklabels=classes_list,
        cbar=False,
        cmap="Blues"
    )
    ax.set_xlabel("Predicted", fontsize=16,labelpad=10)
    ax.set_ylabel("True", fontsize=16,labelpad=10)
    ax.set_xticklabels(classes_list, rotation=0)
    ax.set_yticklabels(classes_list, rotation=90)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_aspect("equal")
    fig.tight_layout()

    # save
    os.makedirs("figures", exist_ok=True)
    saved_paths = []
    for f in formats:
        f = f.lstrip(".").lower()
        out_path = os.path.join("figures", f"{plot_name}.{f}")
        if f == "png":
            fig.savefig(out_path, dpi=300)
        else:
            fig.savefig(out_path)
        saved_paths.append(out_path)

    for p in saved_paths:
        print(f"Saved: {p}")
    return fig, ax



#
# # confusion matrix
# def plot_confusion_matrix(true_classes_column,pred_classes_column, title_text):
#     """
#     Gets true labels and prediction lists, creates a confusion matrix
#     and returns the plot.
#     true_classes_column (Series/list/array):
#     pred_classes_column (Series/list/array):
#     title_text (String):
#     is_normalize (Boolean): if False shows counts, if true shows percentage out of rows
#                             (out of each class).
#     returns: confusion matrix plotly plot
#     """
#
#     classes_list = sorted(true_classes_column.unique().tolist())
#     cm = confusion_matrix(y_true=true_classes_column, y_pred=pred_classes_column, normalize=True)
#     print(cm)
#     fig = px.imshow(cm, text_auto=".3f",color_continuous_scale="Blues",
#                     labels=dict(x="Predicted", y="True", color="Count"),
#                     x=classes_list, y=classes_list)
#     fig.update_layout(
#         title=dict(
#             text=f"<b>{title_text}</b>",
#             x=0.5,
#             xanchor='center',
#             font=dict(size=22)
#         )
#     )
#     return fig

# evaluate the retriever
def precision_at_k(res_df):
    """
    :param res_df: results dataframe
    :return: precision at k
    """
    # identify example labels
    top_cols = [c for c in res_df.columns if c.startswith("top_example_")]

    # if no examples
    k = len(top_cols)

    # precision for each instance
    precisions = []
    for _, row in res_df.iterrows():
        relevant_count = sum(row[col] == row["true_label"] for col in top_cols)
        precision = relevant_count / float(k)
        precisions.append(precision)
    # mean precision
    return float(np.mean(precisions))

def hit_at_k(res_df):
    """
    :param res_df: results dataframe
    :return: hit at k
    """
    # identify example labels
    top_cols = [c for c in res_df.columns if c.startswith("top_example_")]

    # hit per samples
    hits = []
    for _, row in res_df.iterrows():
        hit = 1 if any(row[col] == row["true_label"] for col in top_cols) else 0
        hits.append(hit)

    return float(np.mean(hits))

def retrieval_quality_eval(csv_paths, plot_name=""):
    """
    gets csv results files, calculate retrieval quality metrics, visualize the table
    and save as a pdf in figure folder. returns a dataframe containing retrieval quality.
    cav_paths: a list of paths to results csvs
    file_name: the name of the save plot
    return dataframe with precision#k and hit#k for each result csv
    """

    rows = []

    for path in csv_paths:
        df = pd.read_csv(path)
        run_name = os.path.splitext(os.path.basename(path))[0]

        # validity
        # identify example labels
        top_cols = [c for c in df.columns if c.startswith("top_example_")]

        # if null label, return
        for col in top_cols:
            if df[col].isnull().any():
                raise ValueError(f"missing value in column: {col}.")

        # if no examples
        k = len(top_cols)
        if k == 0:
            raise ValueError("No top_example_i columns were found.")

        # metrics
        precision = precision_at_k(df)
        hit = hit_at_k(df)

        rows.append({
            "run": run_name,
            "csv_path": path,
            "precision_at_k": precision,
            "hit_at_k": hit
        })

    results_df = pd.DataFrame(rows)

    # format values to percentage
    display_df = results_df.copy()
    cols = ["precision_at_k", "hit_at_k"]
    display_df[cols] = (display_df[cols] * 100).round(2)
    display_df = display_df.drop(columns=["csv_path"])

    # visualize
    fig, ax = plt.subplots(figsize=(14, 0.8 * len(display_df) + 2))
    ax.axis("off")

    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.0, 1.95)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

    # save table plot
    figures_dir = os.path.join(os.getcwd(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f"{plot_name}.pdf"), bbox_inches="tight")

    plt.close(fig)
    return results_df



def recall_at_k(res_df, kb_df, k=3):
    """
    gets a path to the results dataframe and a k (top k examples retrieved) and compute recall at k.
    :param res_df: results dataframe
    :param kb_df (DataFrame): df of kb, to calculate how many instances we have
                                from the same label as the query.
    :param k: top k examples retrieved
    :return: recall at k
    """
    # insert precision calculated for each instance in data
    recalls = []

    # looping through instances, calculate its precision
    for _, row in res_df.iterrows():
        # number of relevant items within top k retrieved
        relevant_count = sum(row[f"top_example_{i+1}"] == row["true_label"] for i in range(k))

        # number of relevant items in kb
        total_relevant_in_knowledge_base = kb_df[kb_df["true_label"] == row["true_label"]].shape[0]

        # recall
        recall = relevant_count / total_relevant_in_knowledge_base
        recalls.append(recall)
    # calculate the mean recalls
    return np.mean(recalls)


def f1_at_k(res_df, kb_df, k=3, title_text=""):
    """
    computing recall and precision and compute f1 score.
    res_df (DataFrame): dataframe of rag results. must have columns:
                        top_example_{k}, and true_label.
    kb_df (DataFrame): dataframe of knowledge base, for recall calculation.
                        must have column "true_label".
    k (int): top k examples retrieved.
    return: plotly table that shows recall at k, precision at k and f1.
    """

    # calculate recall and precision
    precision = precision_at_k(res_df, k=k)
    recall = recall_at_k(res_df,kb_df, k=k)

    # calculate f1
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    # visualize
    metrics_df = pd.DataFrame({
        "Metric": ["Precision at k", "Recall at K", "F1-score at K"],
        "Score": [f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    })

    # plot a table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(metrics_df.columns),
                    # fill_color='#636EFA',  # Default header color
                    align='center',
                    font=dict(color='black', size=15)),
        cells=dict(values=[metrics_df["Metric"], metrics_df["Score"]],
                   fill_color='lavender',  # Default cell color
                   align='center',
                   # height=25,
                   font=dict(color='black', size=14)))
    ])

    # Layout styling
    fig.update_layout(template='plotly_white',
                      # title=f"Retrieval Evaluation at K={k}",
                      title=title_text,
                      title_x=0.5,
                      title_font=dict(size=16, weight='bold'),
                      height=450,
                      width=400)

    return fig
    # fig.write_html(f"Retrival evaluation - KB_size{kb_size}.html")

def hist_visualization(df, plot_title):
    """
    gets a dataframe(must have a columns name "true_label", and a title for the plot,
    then creates a histogram for data distribution and returns a plotly hist.
    df (DataFrame): dataframe to plot, must contain a column name "true_label".
    plot_title (String): the title of the plot.
    returns: plotly plot hist.
    """
    # calculate the number and percentage of each label in the data
    label_count = df['true_label'].value_counts()
    label_percentage = (label_count / label_count.sum()) * 100

    # create a text to be shown above each bar
    text = label_percentage.round(2).astype(str) + '%<br>(' + label_count.astype(str) + ')'

    fig = px.bar(x=label_percentage.index, y=label_percentage.values, text=text)
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(width=900, height=650,
                      title_text=f"<b>{plot_title}</b>",  # <<< Important: use title_text, not title!
                      title_x=0.5,
                      title_font=dict(size=24),
                      # title=dict(text=f"<b>{plot_title}</b>", x=0.5, font=dict(size=24)),
                      xaxis_title=dict(text="Class", font=dict(size=18)),
                      yaxis_title=dict(text="Percentage", font=dict(size=18)),
                      # yaxis=dict(range=[0, 100]),
                      yaxis=dict(range=[0, 101], tickmode='linear', tick0=0, dtick=20),
                      xaxis=dict(tickfont=dict(size=15))
                      )
    return fig


def get_misclassified_vs_correct(df_rag, df_zs, sample_size=10, is_rag_wrong=True):
    """
    df_rag (DataFrame): A dataframe contains results. Must contain the columns: file path, true label, prediction.
    df_zs (DataFrame): A dataframe contains results. Must contain the columns: file path, true label, prediction.
    sampled_size (int): A number of images to sample from each class for visualization.
    is_rag_wrong (Boolean): if True -> then plot misclassified images from df1 where df2 classified correctly.
                            if False -> then plot  misclassified images from d2 where df1 classified correctly.
    Returns: filtered_df (DataFrame)
    """

    resulted_rows = []
    labels = df_rag['true_label'].unique().tolist()

    # for each class, filter and find the relevant rows and store in result
    for label in labels:
        # take all relevant rows by label
        rag_label = df_rag[df_rag['true_label'] == label].reset_index(drop=True)
        zs_label = df_zs[df_zs['true_label'] == label].reset_index(drop=True)

        # if df1 is wrong (true)
        if is_rag_wrong:
            # rows where rag is wrong and zs correct
            condition = (rag_label['prediction'] != rag_label['true_label']) & (
                    zs_label['prediction'] == zs_label['true_label'])
            filtered = rag_label[condition]
        else:
            condition = (rag_label['prediction'] == rag_label['true_label']) & (
                        zs_label['prediction'] != zs_label['true_label'])
            # copy relevant rows from RAG and add the prediction of zero shot
            filtered = rag_label[condition].copy()
            # add zero shot prediction to filtered
            zs_filtered = zs_label[condition][['file_path', 'predictions']].copy()
            zs_filtered.rename(columns={'prediction': 'zs_prediction'}, inplace=True)
            filtered = filtered.merge(zs_filtered, on='file_path', how='inner')


        if len(filtered) > sample_size:
            filtered = filtered.sample(sample_size, random_state=42)

        # append the resulted rows from the current label
        resulted_rows.append(filtered)

    # after looping through labels and collect all convert to df and return
    filtered_df = pd.concat(resulted_rows).reset_index(drop=True)
    return filtered_df

def plot_images_from_classes(df, title, is_rag_wrong=True):
    """
    df (DataFrame): the dataframe to plot
    title (String): The title of each plot, will be changed according to is_rag_wrong also.
                    The final title of the plot will look like: title (label).

    """
    # create a dictionary to return later with keys as labels and figures asvalues
    figures = {}
    labels = df['true_label'].unique().tolist()

    for label in labels:
        label_df = df[df['true_label'] == label].reset_index(drop=True)
        # the number of rows in the grip plot based on the number of images
        n = len(label_df)
        rows = n
        cols = 4

        # create the plot
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        fig.suptitle(f"{title} ({label})", fontsize=24, fontweight='bold', y=0.95)

        #  if I only have 1 image to iterate on (one row) then make it iterable
        if rows == 1:
            axes = [axes]
        # loop through rows
        for i, (_, row) in enumerate(label_df.iterrows()):
            # Query image
            path = row['file_path']
            if os.path.exists(path):
                img = Image.open(path)
                axes[i][0].imshow(img, cmap='gray')
                axes[i][0].axis('off')
                pred = row['prediction'] if is_rag_wrong else row['zs_prediction']
                axes[i][0].set_title(f"Prediction: {pred}", fontsize=16)
                axes[i][0].add_patch(Rectangle(
                    (0, 0), 1, 1,
                    transform=axes[i][0].transAxes,
                    linewidth=3,
                    edgecolor='green',
                    facecolor='none'
                ))

            # RAG examples
            for j, (path_key, label_key, sim_key) in enumerate(zip(
                    ['top_example_1_path', 'top_example_2_path', 'top_example_3_path'],
                    ['top_example_1', 'top_example_2', 'top_example_3'],
                    ['top_example_1_similarity', 'top_example_2_similarity', 'top_example_3_similarity']
            )):
                ex_path = row.get(path_key)
                ex_label = row.get(label_key)
                sim_val = row.get(sim_key)
                if ex_path and os.path.exists(ex_path):
                    ex_img = Image.open(ex_path)
                    axes[i][j + 1].imshow(ex_img, cmap='gray')
                    axes[i][j + 1].axis('off')
                    sim_str = f"{sim_val:.3f}"
                    axes[i][j + 1].set_title(f"Label: {ex_label} (Sim: {sim_str})", fontsize=16)
                    axes[i][j + 1].add_patch(Rectangle(
                        (0, 0), 1, 1,
                        transform=axes[i][j + 1].transAxes,
                        linewidth=2,
                        edgecolor='blue',
                        facecolor='none'
                    ))

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        figures[label] = fig
    return figures



def plot_classified_images_by_label(df, title, misclassified=True, sample_size=10):
    """
    Plot misclassified or correctly classified images per label.

    Args:
        df (DataFrame): The input dataframe with at least columns:
                        - 'file_path', 'true_label', 'prediction',
                        - optionally: top_example_X_path, top_example_X, top_example_X_similarity
        title (str): Title prefix for each plot.
        misclassified (bool): Whether to show misclassified (True) or correctly classified (False) images.
        sample_size (int): Number of images per class to sample.

    Returns:
        figures (dict): Dictionary of matplotlib figures per class.
    """
    figures = {}
    labels = df['true_label'].unique().tolist()

    for label in labels:
        # Filter rows for the current class
        label_df = df[df['true_label'] == label]

        if misclassified:
            label_df = label_df[label_df['prediction'] != label_df['true_label']]
        else:
            label_df = label_df[label_df['prediction'] == label_df['true_label']]

        if label_df.empty:
            continue

        # Sample if needed
        if len(label_df) > sample_size:
            label_df = label_df.sample(sample_size, random_state=42)

        label_df = label_df.reset_index(drop=True)
        n = len(label_df)
        rows = n
        cols = 4

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        fig.suptitle(f"{title} ({label})", fontsize=24, fontweight='bold', y=0.95)

        if rows == 1:
            axes = [axes]

        for i, (_, row) in enumerate(label_df.iterrows()):
            # Main image
            path = row['file_path']
            if os.path.exists(path):
                img = Image.open(path)
                axes[i][0].imshow(img, cmap='gray')
                axes[i][0].axis('off')
                pred = row['prediction']
                axes[i][0].set_title(f"Prediction: {pred}", fontsize=16)
                axes[i][0].add_patch(Rectangle(
                    (0, 0), 1, 1,
                    transform=axes[i][0].transAxes,
                    linewidth=3,
                    edgecolor='green',
                    facecolor='none'
                ))

            # Top-k examples (if available)
            for j, (path_key, label_key, sim_key) in enumerate(zip(
                    ['top_path_1', 'top_path_2', 'top_path_3'],
                    ['top_label_1', 'top_label_2', 'top_label_3'],
                    ['top_cosine_1', 'top_cosine_2', 'top_cosine_3']
            )):
                ex_path = row.get(path_key)
                ex_label = row.get(label_key)
                sim_val = row.get(sim_key)
                if ex_path and os.path.exists(ex_path):
                    ex_img = Image.open(ex_path)
                    axes[i][j + 1].imshow(ex_img, cmap='gray')
                    axes[i][j + 1].axis('off')
                    sim_str = f"{sim_val:.3f}" if sim_val is not None else "N/A"
                    axes[i][j + 1].set_title(f"Label: {ex_label} (Sim: {sim_str})", fontsize=14)
                    axes[i][j + 1].add_patch(Rectangle(
                        (0, 0), 1, 1,
                        transform=axes[i][j + 1].transAxes,
                        linewidth=2,
                        edgecolor='blue',
                        facecolor='none'
                    ))

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        figures[label] = fig

    return figures


def plot_label_distribution(dfs, label_col="true_label", normalize="percent", plot_name=""):
    """
    dfs: dict like {"train": df_train, "val": df_val, "test": df_test}
    normalize: "percent" (recommended) or "count"
    """
    # Count labels per split
    counts = {}
    for split, df in dfs.items():
        vc = df[label_col].value_counts(dropna=False)
        counts[split] = vc

    dist = pd.DataFrame(counts).fillna(0)

    # Keep a stable class order (by total frequency across splits)
    dist = dist.loc[dist.sum(axis=1).sort_values(ascending=False).index]

    if normalize == "percent":
        dist = dist.div(dist.sum(axis=0), axis=1) * 100
        ylabel = "Percent (%)"
    else:
        ylabel = "Count"

    # captalize
    dist.index = dist.index.astype(str).str.capitalize()
    dist.columns = dist.columns.astype(str).str.capitalize()
    # Create figure/axis explicitly so we can save it
    fig, ax = plt.subplots(figsize=(8, 5))
    dist.plot(kind="bar", ax=ax)

    ax.set_xlabel("Class", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.legend(title="Split", frameon=False, fontsize=10, title_fontsize=10)
    plt.xticks(rotation=20, ha="right", fontsize=11)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"{plot_name}.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out_path}")


def visualize_sample_images(df, seed=42, n_images=2, plot_name=""):
    labels = sorted(df["true_label"].dropna().unique())
    random.seed(seed)
    label_to_paths = {}
    for label in labels:
        paths = df.loc[df["true_label"] == label, "file_path"].dropna().tolist()

        # sample up to 2 (if there are fewer than 2, take what exists)
        k = min(n_images, len(paths))
        label_to_paths[label] = random.sample(paths, k) if k > 0 else []

    labels = list(label_to_paths.keys())  # same order you built the dict with
    n_cols = len(labels)
    n_rows = n_images

    # visualize
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 4.2))

    # if only 1 label, axes shape is different
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, label in enumerate(labels):
        # title (bold) above the column
        axes[0, col].set_title(str(label).capitalize(), fontweight="bold", fontsize=20, pad=14)

        paths = label_to_paths[label]  # list of up to 2 file paths

        for row in range(n_rows):
            ax = axes[row, col]
            ax.axis("off")

            if row < len(paths):
                img_path = paths[row]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img, cmap="gray" if img.mode in ["L", "I;16", "I"] else None)
                except Exception:
                    ax.text(0.5, 0.5, "Could not load", ha="center", va="center")

    plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)

    os.makedirs("figures", exist_ok=True)
    if not plot_name:
        plot_name = "sample_images"
    out_path = os.path.join("figures", f"{plot_name}.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out_path}")
