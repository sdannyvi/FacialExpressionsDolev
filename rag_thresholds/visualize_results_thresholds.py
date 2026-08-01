import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.LLaVa.llava_rag.vis_results import *
import pandas as pd
from pathlib import Path
import random


# folder path
base_path = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets")



# private test class distribution
private_test_order = ["private_test_0%.csv", "private_test_50%.csv","private_test_80%.csv"]

dfs_private = {}

for file_name in private_test_order:
    csv_path = base_path / file_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        dfs_private[file_name.replace(".csv", "")] = df
    else:
        print(f"Missing file: {csv_path}")

if dfs_private:
    class_distribution_table(dfs=dfs_private,
        label_col="true_label",
        file_name="Private test- class distribution",
        show_removed_percentage=True
    )

    label_map_private = {
        "private_test_0%": "Low agreement (0%)",
        "private_test_50%": "Medium agreement (50%)",
        "private_test_80%": "High agreement (80%)"
    }

    dfs_private_plot = {
        label_map_private[key]: df
        for key, df in dfs_private.items()
    }

    plot_label_distribution(
        dfs=dfs_private_plot,
        label_col="true_label",
        normalize="percent",
        plot_name="bar plot - Private test label distribution"
    )

# knowledge base class distribution
kb_order = ["kb_0%.csv","kb_50%.csv","kb_80%.csv","kb_50_match_size_kb_80%.csv"]

dfs_kb = {}

for file_name in kb_order:
    csv_path = base_path / file_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        dfs_kb[file_name.replace(".csv", "")] = df

        # validate
        print(f"\nDataset: {file_name}")
        counts = df["true_label"].value_counts(dropna=False)
        percentages = df["true_label"].value_counts(normalize=True, dropna=False) * 100
        for cls in counts.index:
            print(f"{cls}: {counts[cls]} ({percentages[cls]:.2f}%)")
    else:
        print(f"Missing file: {csv_path}")

if dfs_kb:
    class_distribution_table(
        dfs=dfs_kb,
        label_col="true_label",
        file_name="KB- class distribution",
        show_removed_percentage=True
    )


    label_map_kb = {
        "kb_0%": "Low agreement (0%)",
        "kb_50%": "Medium agreement (50%)",
        "kb_80%": "High agreement (80%)",
        "kb_50_match_size_kb_80%": "Medium agreement 50% (size-matched 80%)"
    }

    dfs_kb_plot = {
        label_map_kb[key]: df
        for key, df in dfs_kb.items()
    }

    plot_label_distribution(
        dfs=dfs_kb_plot,
        label_col="true_label",
        normalize="percent",
        plot_name="bar plot - knowledge base label distribution"
    )



# visualize ambiguity
kb0_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_0%.csv")
print(sorted(kb0_df["votes_percentage"].unique().tolist())[:10])
ambiguity_distribution(df=kb0_df, plot_name="vote percentage distribution")

plot_ambiguity_image(df=kb0_df, seed=43, file_name="ambiguity images")

# # path to results
# path_dir = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results"
#
#
# # loop over all .csv files
# for file_path in Path(path_dir).glob("*.csv"):
#     df = pd.read_csv(file_path)
#
#     # get file name, remove extension .csv
#     base_name = file_path.stem
#
#     # create plot title. exp: KB 0%, Public Test 0%
#     # if format is match size
#     if "[" in base_name:
#         parts = base_name.split("_")
#         kb_pct = parts[1]
#         public_pct = parts[-1]
#         match_kb = base_name.split("[match_size_kb_")[1].split("%]")[0]
#         title = f"KB {kb_pct}, Public Test {public_pct} (KB Sample Size Equal to {match_kb}%)"
#     else:
#         # Format: kb_0%_public_test_0%
#         # Goal: KB 0%, Public Test 0%
#         parts = base_name.split("_")
#         kb_pct = parts[1]
#         public_pct = parts[-1]
#         title = f"KB {kb_pct}, Public Test {public_pct}"
#     print(f"plot title for file: {base_name} is: {title}")
#     print(f"true label: {sorted(df['true_label'].unique().tolist())}")
#     print(f"prediction: {sorted(df['prediction'].unique().tolist())}")
#     print(f"is true label match prediction? {set(df['true_label'].unique()) == set(df['prediction'].unique())}")
#     print(f"number of samples in {base_name}: {len(df)}")
#     # Call your plotting function
#     plt = plot_classification_report(
#         true_classes_column=df["true_label"],
#         pred_classes_column=df["prediction"],
#         title_text=title
#     )
#     plt.show()


# path_train_test = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets"
# dict_count = {}
# # loop over all .csv files
# for file_path in Path(path_train_test).glob("*.csv"):
#     df = pd.read_csv(file_path)
#
#     # get file name, remove extension .csv
#     base_name = file_path.stem
#     print(f"number of samples in {base_name}: {len(df)}")
#     dict_count[base_name] = df
#
# print(f"dict count: {dict_count}")
# # calculate Jaccard
# def jaccard_distance(df1, df2, key="file_path"):
#     set1 = set(df1[key])
#     set2 = set(df2[key])
#     intersection = len(set1 & set2)
#     union = len(set1 | set2)
#     print(f"intersection: {intersection}")
#     print(f"union: {union}")
#     return (1 - intersection / union) * 100
#
#
#
# dist = jaccard_distance(df1=dict_count['kb_0%'], df2=dict_count['kb_50%'])
# print(f"Jaccard for KB 0, KB 50: {round(dist,4)}%")
# dist = jaccard_distance(df1=dict_count['kb_0%'], df2=dict_count['kb_80%'])
# print(f"Jaccard for KB 0, KB 80: {round(dist,4)}%")
# dist = jaccard_distance(df1=dict_count['kb_50%'], df2=dict_count['kb_80%'])
# print(f"Jaccard for KB 50, KB 80: {dist}%")
# dist = jaccard_distance(df1=dict_count['kb_50%_[match_size_kb_80%]'], df2=dict_count['kb_50%'])
# print(f"Jaccard for KB kb_50%_[match_size_kb_80%], KB 50: {dist}%")
# dist = jaccard_distance(df1=dict_count['kb_50%_[match_size_kb_80%]'], df2=dict_count['kb_80%'])
# print(f"Jaccard for KB kb_50%_[match_size_kb_80%], KB 80: {round(dist,4)}%")
#
#
# dist = jaccard_distance(df1=dict_count['public_test_0%'], df2=dict_count['public_test_50%'])
# print(f"Jaccard for test 0, test 50: {round(dist,4)}%")
# dist = jaccard_distance(df1=dict_count['public_test_0%'], df2=dict_count['public_test_80%'])
# print(f"Jaccard for test 0, test 80: {round(dist,4)}%")
# dist = jaccard_distance(df1=dict_count['public_test_50%'], df2=dict_count['public_test_80%'])
# print(f"Jaccard for test 50, test 80: {round(dist,4)}%")
#
#
#
# # total number of samples in fer before thresholding
# fer = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_data.csv")
# print(f"number of samples in fer: {len(fer)}")
#
#
#
#
#
# # estimating the retriever
# res6 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/kb_50%_public_test_50%.csv")
# print(f"cols: {res6.columns.tolist()}")
# retriever_eval1 = res6[['true_label', 'top_label_1', 'top_label_2', 'top_label_3']].copy()
#
#
# # for each row get the prediction
# def get_mode_with_random_tie(row):
#     """
#     get the mode as a prediction by the retriever. if there is no mode - select randomly among the three top K labels.
#     """
#     labels = [row['top_label_1'], row['top_label_2'], row['top_label_3']]
#     label_counts = pd.Series(labels).value_counts()
#     max_count = label_counts.max()
#     top_candidates = label_counts[label_counts == max_count].index.tolist()
#     return random.choice(top_candidates)
#
# # Apply to each row
# retriever_eval1['prediction'] = retriever_eval1.apply(get_mode_with_random_tie, axis=1)
#
# # Calculate accuracy
# accuracy = (retriever_eval1['prediction'] == retriever_eval1['true_label']).mean()
# print(f"Accuracy: {accuracy:.3%}")
#
# print("\nFirst 5 prediction examples:")
# print(retriever_eval1[['true_label', 'top_label_1', 'top_label_2', 'top_label_3', 'prediction']].head(20))

