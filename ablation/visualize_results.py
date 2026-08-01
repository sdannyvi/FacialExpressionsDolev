import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

df_clip = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/rag_kb_50%_private_test_50%_no_lda.csv")
df_one_exp = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/rag_kb_50%_private_test_50%_top_1.csv")

# validate results
print("RAG - CLIP only:")
validate_results(df_clip)

# visualize results
plot_metrics(y_true=df_clip["true_label"], y_pred=df_clip["prediction"], digits=2, formats=("pdf",),
             plot_name= "RAG (No LDA)- Report")
plot_confusion_matrix(true_classes=df_clip["true_label"],
                      pred_classes=df_clip["prediction"], formats=("pdf",),
                      plot_name="RAG (No LDA)- Confusion matrix")


# validate results
print("RAG - Top 1:")
validate_results(df_one_exp)

# visualize results
plot_metrics(y_true=df_one_exp["true_label"], y_pred=df_one_exp["prediction"], digits=2, formats=("pdf",),
             plot_name= "RAG (Top-1)- Report")
plot_confusion_matrix(true_classes=df_one_exp["true_label"],
                      pred_classes=df_one_exp["prediction"], formats=("pdf",),
                      plot_name="RAG (Top-1)- Confusion matrix")


# retriever compared to RAG
df_with_dup = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/kb_50%_private_test_50%.csv")
df_no_dup  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/rag_kb_50%_private_test_50%_no_leakage.csv")

# validate results
print("before duplicate removal:")
validate_results(df_with_dup)
print("after duplicate removal:")
validate_results(df_no_dup)

# validate both datasets are the same
is_math=  set(df_with_dup["file_path"]) == set(df_no_dup["file_path"])
print(f"does both dataframes before and after duplicates are match? {is_math}")

# how many times the top-2 retrieved examples had the same cosine similarity on dataset with duplicates
same_dist = df_with_dup["top_cosine_1"] == df_with_dup["top_cosine_2"]
same_dist_count = same_dist.sum()
same_dist_pct = same_dist_count / len(df_with_dup)
print(f"same_dist_count: {same_dist_count}")
print(f"same_dist_pct: {same_dist_pct:.6f} ({same_dist_pct*100:.4f}%)")

# how many times the top-2 retrieved examples had the same cosine similarity AND different labels
# exact cosine ties
same_cos = df_with_dup["top_cosine_1"] == df_with_dup["top_cosine_2"]
# among those, different labels
diff_labels_in_ties = same_cos & (df_with_dup["top_label_1"] != df_with_dup["top_label_2"])
count_diff_labels_in_ties = diff_labels_in_ties.sum()
pct_diff_labels_in_ties = count_diff_labels_in_ties / len(df_with_dup)
print(f"count_diff_labels_in_ties: {count_diff_labels_in_ties}")
print(f"pct_diff_labels_in_ties: {pct_diff_labels_in_ties:.6f} ({pct_diff_labels_in_ties*100:.4f}%)")


# calculate retriever and generator performance
rows = []
dfs = {"before duplicate removal": df_with_dup, "after duplicate removal": df_no_dup}
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
                          plot_name=f"{name} - Retriever confusion matrix")

    # RAG confusion matrix
    plot_confusion_matrix(true_classes=df["true_label"],pred_classes=df["prediction"],formats=("pdf",),
                          plot_name=f"{name} - RAG confusion matrix")

# visualize table
# create dataframe
table_df = pd.DataFrame(rows, columns=["Method", "Duplicate removal", "Accuracy", "Macro F1"])

# make duplicate-removal column look like the table in your image
table_df["Duplicate removal"] = table_df["Duplicate removal"].replace({
    "before duplicate removal": "-",
    "after duplicate removal": "V"
})

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
out_path = os.path.join("figures", "Retriever and RAG comparison (before and after duplicates).pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

print(f"Saved: {out_path}")
plt.close(fig)


# class distribution of before and after duplicate removal
kb_full = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_50%.csv")
kb_clean = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/kb_50%_no_leakage.csv")
plot_label_distribution(dfs={"before duplicate removal": kb_full, "after duplicate removal": kb_clean},
                        label_col="true_label", normalize="percent",
                        plot_name="class distribution (bar plot)- before and after duplicates")

class_distribution_table(dfs={"before duplicate removal": kb_full, "after duplicate removal": kb_clean},
                         label_col="true_label",
                         file_name="class distribution (table) - before and after duplicates")

# error analysis
total = len(df_no_dup)
# A1: top1 correct, top2 correct, generator correct
case_A1 = ((df_no_dup["top_label_1"] == df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] == df_no_dup["true_label"]) &
           (df_no_dup["prediction"] == df_no_dup["true_label"])).sum()

# A2: top1 correct, top2 correct, generator incorrect
case_A2 = ((df_no_dup["top_label_1"] == df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] == df_no_dup["true_label"]) &
           (df_no_dup["prediction"] != df_no_dup["true_label"])).sum()

# B1: top1 correct, top2 incorrect, generator correct
case_B1 = ((df_no_dup["top_label_1"] == df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] != df_no_dup["true_label"]) &
           (df_no_dup["prediction"] == df_no_dup["true_label"])).sum()

# B2: top1 correct, top2 incorrect. generator incorrect
case_B2 = ((df_no_dup["top_label_1"] == df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] != df_no_dup["true_label"]) &
           (df_no_dup["prediction"] != df_no_dup["true_label"])).sum()

# C1: top1 incorrect, top2 correct, generator correct
case_C1 = ((df_no_dup["top_label_1"] != df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] == df_no_dup["true_label"]) &
           (df_no_dup["prediction"] == df_no_dup["true_label"])).sum()

# C2: top1 incorrect, top2 correct, generator incorrect
case_C2 = ((df_no_dup["top_label_1"] != df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] == df_no_dup["true_label"]) &
           (df_no_dup["prediction"] != df_no_dup["true_label"])).sum()

# D1: top1 incorrect, top2 incorrect, generator correct
case_D1 = ((df_no_dup["top_label_1"] != df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] != df_no_dup["true_label"]) &
           (df_no_dup["prediction"] == df_no_dup["true_label"])).sum()

# D2: top1 incorrect, top2 incorrect, generator incorrect
case_D2 = ((df_no_dup["top_label_1"] != df_no_dup["true_label"]) &
           (df_no_dup["top_label_2"] != df_no_dup["true_label"]) &
           (df_no_dup["prediction"] != df_no_dup["true_label"])).sum()


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
out_path = os.path.join("figures", "Error_analysis_8_cases.pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

print(f"Saved: {out_path}")
plt.close(fig)






# case 4 - zero-shot vs RAG when retrieval fails
df_zero_shot_50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/kb_50%_private_test_50%.csv")


# D1 + D2 in the RAG dataframe:
# top1 incorrect and top2 incorrect
mask_D = (
    (df_no_dup["top_label_1"] != df_no_dup["true_label"]) &
    (df_no_dup["top_label_2"] != df_no_dup["true_label"])
)

# keep only D1 + D2 samples from RAG
df_rag_D = df_no_dup.loc[mask_D, ["file_path", "true_label", "prediction"]].copy()
df_rag_D = df_rag_D.rename(columns={"prediction": "prediction_rag"})

# keep matching samples from zero-shot
df_zero_D = df_zero_shot_50.loc[:, ["file_path", "true_label", "prediction"]].copy()
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
out_path = os.path.join("figures", "RAG_vs_ZeroShot_on_D1_D2.pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)

print(f"Saved: {out_path}")
plt.close(fig)







#
#
# # RAF DB
# df_radboud = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/results/rag_radboud.csv")
#
# # calculate metrics
# retriever_acc = accuracy_score(df_radboud["true_label"], df_radboud["top_label_1"])
# rag_acc = accuracy_score(df_radboud["true_label"], df_radboud["prediction"])
#
# retriever_f1 = f1_score(df_radboud["true_label"], df_radboud["top_label_1"], average="macro")
# rag_f1 = f1_score(df_radboud["true_label"], df_radboud["prediction"], average="macro")
#
# # print results
# print(f"Retriever accuracy: {retriever_acc:.6f} ({retriever_acc*100:.2f}%)")
# print(f"RAG accuracy: {rag_acc:.6f} ({rag_acc*100:.2f}%)")
# print(f"Retriever Macro F1: {retriever_f1:.6f} ({retriever_f1*100:.2f}%)")
# print(f"RAG Macro F1: {rag_f1:.6f} ({rag_f1*100:.2f}%)")
#
# # create rows for the table
# rows = [
#     ["Retriever (Top-1)", retriever_acc * 100, retriever_f1 * 100],
#     ["RAG", rag_acc * 100, rag_f1 * 100]
# ]
#
# # create dataframe for table
# table_df = pd.DataFrame(rows, columns=["Method", "Accuracy", "Macro F1"])
#
# # round numeric columns
# table_df["Accuracy"] = table_df["Accuracy"].round(2)
# table_df["Macro F1"] = table_df["Macro F1"].round(2)
#
# print(table_df)
#
# # visualize table and save as pdf
# fig, ax = plt.subplots(figsize=(8, 2.2 + 0.8 * len(table_df)))
# ax.axis("off")
#
# tbl = ax.table(
#     cellText=table_df.values,
#     colLabels=table_df.columns,
#     loc="center",
#     cellLoc="center"
# )
#
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(13)
# tbl.scale(1.1, 2.0)
#
# # make header bold
# for (row, col), cell in tbl.get_celld().items():
#     if row == 0:
#         cell.set_text_props(weight="bold")
#
# # save file
# os.makedirs("figures", exist_ok=True)
# out_path = os.path.join("figures", "Radboud dataset- Retriever and RAG comparison.pdf")
# fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
#
# print(f"Saved: {out_path}")
# plt.close(fig)
#
# #  error analysis
#
# total = len(df_radboud)
# # case 1: top1 = true_label AND generator = top1
# case_A = ((df_radboud["top_label_1"] == df_radboud["true_label"]) &
#           (df_radboud["prediction"] == df_radboud["top_label_1"])).sum()
#
# # Case 2: top1 = true_label AND generator != top1
# case_B = ((df_radboud["top_label_1"] == df_radboud["true_label"]) &
#           (df_radboud["prediction"] != df_radboud["top_label_1"])).sum()
#
# # C1: top1 wrong, top2 correct, generator correct
# case_C1 = ((df_radboud["top_label_1"] != df_radboud["true_label"]) &
#            (df_radboud["top_label_2"] == df_radboud["true_label"]) &
#            (df_radboud["prediction"] == df_radboud["true_label"])).sum()
#
# # C2: top1 wrong, top2 wrong, generator correct
# case_C2 = ((df_radboud["top_label_1"] != df_radboud["true_label"]) &
#            (df_radboud["top_label_2"] != df_radboud["true_label"]) &
#            (df_radboud["prediction"] == df_radboud["true_label"])).sum()
#
# # D1: top1 wrong, top2 correct, generator wrong
# case_D1 = ((df_radboud["top_label_1"] != df_radboud["true_label"]) &
#            (df_radboud["top_label_2"] == df_radboud["true_label"]) &
#            (df_radboud["prediction"] != df_radboud["true_label"])).sum()
#
# # D2: top1 wrong, top2 wrong, generator wrong
# case_D2 = ((df_radboud["top_label_1"] != df_radboud["true_label"]) &
#            (df_radboud["top_label_2"] != df_radboud["true_label"]) &
#            (df_radboud["prediction"] != df_radboud["true_label"])).sum()
#
#
# # print percentages
# print("Case A (Top-1 correct, generator follows Top-1):", round(case_A / total * 100, 2), "%")
# print("Case B (Top-1 correct, generator does not follow Top-1):", round(case_B / total * 100, 2), "%")
# print("Case C1 (Top-1 wrong, Top-2 correct, generator correct):", round(case_C1 / total * 100, 2), "%")
# print("Case C2 (Top-1 wrong, Top-2 wrong, generator correct):", round(case_C2 / total * 100, 2), "%")
# print("Case D1 (Top-1 wrong, Top-2 correct, generator wrong):", round(case_D1 / total * 100, 2), "%")
# print("Case D2 (Top-1 wrong, Top-2 wrong, generator wrong):", round(case_D2 / total * 100, 2), "%")
#
# # visualize table
# rows = [
#     ["A",
#      "Top-1 correct,\ngenerator follows Top-1",
#      case_A,
#      round(case_A / total * 100, 2),
#      "Retrieval helped"],
#     ["B",
#      "Top-1 correct,\ngenerator does not follow Top-1",
#      case_B,
#      round(case_B / total * 100, 2),
#      "Generator ignored correct retrieval"],
#
#     ["C1",
#      "Top-1 wrong, Top-2 correct,\ngenerator predicts true label",
#      case_C1,
#      round(case_C1 / total * 100, 2),
#      "Generator used Top-2"],
#
#     ["C2",
#      "Top-1 wrong, Top-2 wrong,\ngenerator predicts true label",
#      case_C2,
#      round(case_C2 / total * 100, 2),
#      "Generator correct without retrieval"],
#
#     ["D1",
#      "Top-1 wrong, Top-2 correct,\ngenerator also wrong",
#      case_D1,
#      round(case_D1 / total * 100, 2),
#      "Generator ignored correct Top-2"],
#
#     ["D2",
#      "Top-1 wrong, Top-2 wrong,\ngenerator also wrong",
#      case_D2,
#      round(case_D2 / total * 100, 2),
#      "Retrieval failed completely"]
# ]
#
# # dataframe
# table_df = pd.DataFrame(
#     rows,
#     columns=["Case", "Condition", "Count", "Percentage", "Interpretation"]
# )
#
# print(table_df)
#
# # visualize table and save as pdf
# fig, ax = plt.subplots(figsize=(18, 2.5 + 0.9 * len(table_df)))
# ax.axis("off")
#
# tbl = ax.table(
#     cellText=table_df.values,
#     colLabels=table_df.columns,
#     loc="center",
#     cellLoc="center"
# )
#
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(12)
# tbl.scale(1.2, 2.8)
#
# # make header bold
# for (row, col), cell in tbl.get_celld().items():
#     if row == 0:
#         cell.set_text_props(weight="bold")
#
# # save pdf
# os.makedirs("figures", exist_ok=True)
# out_path = os.path.join("figures", "Radboud dataset (error analysis) - Generator vs Retriever (Top1 and Top2).pdf")
# fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
#
# print(f"Saved: {out_path}")
# plt.close(fig)