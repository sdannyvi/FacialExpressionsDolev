import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
from pathlib import Path


# loop through csv files in path
base_path = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results")

# validate and visualize for each csv
for csv_path in base_path.glob("*.csv"):

    # get file name
    file_name = csv_path.stem.lower()

    print(f"Processing file: {csv_path.name}")

    # load csv
    df = pd.read_csv(csv_path)

    # validate and normalize prediction
    df = validate_results(df)

    # plot name
    parts = file_name.split("_")
    plot_title=""


    if file_name.startswith("kb_50%_[match_size_kb_80%]_private_test_50%"):
        plot_title = "KB 50% (match size with 80%)"

    elif file_name.startswith("kb"):
        # RAG case
        kb_percent = parts[1]
        test_percent = parts[-1]

        plot_title = f"KB {kb_percent} Private Test {test_percent}"

    elif file_name.startswith("zero_shot"):
        # Zero-shot case
        test_percent = parts[-1]

        plot_title = f"Zero-shot Private Test {test_percent}"


    print(f"parsed title: {plot_title}")

    # visualize results
    plot_metrics(y_true=df["true_label"], y_pred=df["prediction"], digits=2, formats=("pdf",),
                     plot_name=plot_title + " - Report" )
    plot_confusion_matrix(true_classes=df["true_label"],
                          pred_classes=df["prediction"], formats=("pdf",),
                          plot_name=plot_title + " - Confusion matrix")

# print zero shot and rag for 50%
df_zero = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/zero_shot_private_test_50%.csv")
df_rag = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/experiments_results/kb_50%_private_test_50%.csv")

df_zero = validate_results(df_zero)
df_rag = validate_results(df_rag)
plot_confusion_matrix(true_classes=df_zero["true_label"],pred_classes=df_zero["prediction"], formats=("pdf",),
                      plot_name="zero_shot_ferplus")

plot_confusion_matrix(true_classes=df_rag["true_label"],pred_classes=df_rag["prediction"], formats=("pdf",),
                      plot_name="rag_ferplus")

