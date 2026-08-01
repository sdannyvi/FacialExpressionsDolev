import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/kb_50%_private_test_50%.csv")

# df["prediction"] = df["prediction"].replace({"anger": "angry"})
# validate results
print(f"label list: {sorted(df['true_label'].unique().tolist())}")
print(f"prediction list: {sorted(df['prediction'].unique().tolist())}")
print(f"same unique labels? {set(df['true_label'].dropna().unique()) == set(df['prediction'].dropna().unique())}")
print(f"are there nulls in true label? {df['true_label'].isna().any()}")
print(f"are there nulls in prediction? {df['prediction'].isna().any()}")
print(f"columns: {df.columns.tolist()}")


# exact ties (no epsilon)
same_dist = df["top_cosine_1"] == df["top_cosine_2"]

same_dist_count = same_dist.sum()
same_dist_pct = same_dist_count / len(df)

print(f"same_dist_count: {same_dist_count}")
print(f"same_dist_pct: {same_dist_pct:.6f} ({same_dist_pct*100:.4f}%)")


# 1) exact cosine ties
same_cos = df["top_cosine_1"] == df["top_cosine_2"]

# 2) among those, different labels
diff_labels_in_ties = same_cos & (df["top_label_1"] != df["top_label_2"])

count_diff_labels_in_ties = diff_labels_in_ties.sum()
pct_diff_labels_in_ties = count_diff_labels_in_ties / len(df)

print(f"count_diff_labels_in_ties: {count_diff_labels_in_ties}")
print(f"pct_diff_labels_in_ties: {pct_diff_labels_in_ties:.6f} ({pct_diff_labels_in_ties*100:.4f}%)")

# calculate retriever accuracy and RAG accuracy
retriever_acc = accuracy_score(df["true_label"], df["top_label_1"])
rag_acc = accuracy_score(df["true_label"], df["prediction"])

print(f"retriever_acc: {retriever_acc:.6f} ({retriever_acc*100:.3f}%)")
print(f"rag_acc: {rag_acc:.6f} ({rag_acc*100:.3f}%)")

# calculate accuracy of retriever on leakage test
df_map = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/duplicate_map.csv")
test_map = df_map[df_map["split"] == "private_test"][["file_path", "is_leakage"]]
df = df.merge(test_map, on="file_path", how="inner")
# get only leakage test
leak_df = df[df["is_leakage"] == True]
retriever_acc_leak = accuracy_score(leak_df["true_label"], leak_df["top_label_1"])
gen_acc_leak = accuracy_score(leak_df["true_label"], leak_df["prediction"])
print(f"Retriever accuracy on leakage test: {retriever_acc_leak*100:.3f}")
print(f"RAG accuracy on leakage test: {gen_acc_leak*100:.3f}")

# retriever accuracy on non-leakage test samples
non_leak_df = df[df["is_leakage"] == False]
retriever_acc = accuracy_score(non_leak_df["true_label"], non_leak_df["top_label_1"])
gen_acc = accuracy_score(non_leak_df["true_label"], non_leak_df["prediction"])
print(f"Retriever accuracy on non-leakage test: {retriever_acc*100:.3f}")
print(f"RAG accuracy on non-leakage test: {gen_acc*100:.3f}")






