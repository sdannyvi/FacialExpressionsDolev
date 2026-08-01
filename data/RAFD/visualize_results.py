import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
from pathlib import Path

# train_set = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/train_set_radboud.csv")
# test_set = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/test_set_radboud.csv")
#
# class_distribution_table(dfs={"Train": train_set, "Test": test_set},
#                          label_col="true_label", file_name="class distribution table - Radboud")
#
# # counts + percentages for each class in df["true_label"]
# counts = train_set["true_label"].value_counts(dropna=False)
# percentages = train_set["true_label"].value_counts(normalize=True, dropna=False) * 100
#
# for cls in counts.index:
#     print(f"{cls}: {counts[cls]} ({percentages[cls]:.2f}%)")

# folder with Radboud csv files
base_path = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/results")

for csv_path in base_path.glob("*.csv"):
    file_name = csv_path.stem.lower()
    print(f"\nProcessing: {csv_path.name}")

    run_type = "missing"
    run_label = "missing"
    # decide run type from file name
    if "zero" in file_name:
        run_type = "zero shot"
        run_label = "Zero shot"
    elif "rag" in file_name:
        run_type = "rag"
        run_label = "RAG"

    # load csv
    df = pd.read_csv(csv_path)

    # validate
    print(f"Validate {run_type}:")
    df = validate_results(df)

    # plot full dataset
    plot_metrics(
        y_true=df["true_label"],
        y_pred=df["prediction"],
        digits=2,
        formats=("pdf",),
        plot_name=f"Radboud ({run_label}) - Report"
    )

    plot_confusion_matrix(
        true_classes=df["true_label"],
        pred_classes=df["prediction"],
        formats=("pdf",),
        plot_name=f"Radboud ({run_label}) - Confusion matrix"
    )

    # split by camera angle
    df["camera_angle"] = df["camera_angle"].astype(str).str.strip()

    df_profile = df[df["camera_angle"].isin(["Rafd000", "Rafd180"])]
    df_semi_profile = df[df["camera_angle"].isin(["Rafd045", "Rafd135"])]
    df_frontal = df[df["camera_angle"] == "Rafd090"]

    subsets = {
        "Profile": df_profile,
        "Semi profile": df_semi_profile,
        "Frontal": df_frontal
    }

    for subset_name, df_subset in subsets.items():
        if df_subset.empty:
            print(f"Skipped {subset_name}: no rows found")
            continue

        print(f"{subset_name}: {len(df_subset)} rows")

        plot_metrics(
            y_true=df_subset["true_label"],
            y_pred=df_subset["prediction"],
            digits=2,
            formats=("pdf",),
            plot_name=f"Radboud ({subset_name}) ({run_label}) - Report"
        )

        plot_confusion_matrix(
            true_classes=df_subset["true_label"],
            pred_classes=df_subset["prediction"],
            formats=("pdf",),
            plot_name=f"Radboud ({subset_name}) ({run_label}) - Confusion matrix"
        )



df_train = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/train_set_radboud.csv")
df_test = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/test_set_radboud.csv")
class_distribution_table(dfs={"train set": df_train, "test set": df_test},label_col="true_label",
                         file_name="Radboud - class distribution")









#
#
#
#
# df_zero = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/results/zero_shot_radboud.csv")
#
# # zero shot
# df_zero = validate_results(df_zero)
# print(f"predictions: {df_zero['prediction'].unique().tolist()}")
# plot_metrics(y_true=df_zero["true_label"],y_pred=df_zero["prediction"],
#              digits=2,plot_name="Radboud (zero shot)- report")
#
# # split by camera angle
# # frontal views
# df_zero["camera_angle"] = df_zero["camera_angle"].astype(str).str.strip()
# df_profile = df_zero[df_zero["camera_angle"].isin(["Rafd000", "Rafd180"])]
#
# # semi-profile views
# df_semi_profile = df_zero[df_zero["camera_angle"].isin(["Rafd045", "Rafd135"])]
#
# # profile views
# df_frontal = df_zero[df_zero["camera_angle"] == "Rafd090"]
#
# plot_metrics(y_true=df_profile["true_label"],y_pred=df_profile["prediction"],
#              digits=2,plot_name="Radboud (zero shot-profile)- report")
# plot_metrics(y_true=df_semi_profile["true_label"],y_pred=df_semi_profile["prediction"],
#              digits=2,plot_name="Radboud (zero shot-semi profile)- report")
# plot_metrics(y_true=df_frontal["true_label"],y_pred=df_frontal["prediction"],
#              digits=2,plot_name="Radboud (zero shot- frontal)- report")
#
# # rag
# df_rag = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/results/rag_radboud.csv")
# df_rag = validate_results(df_rag)
# print(f"predictions: {df_rag['prediction'].unique().tolist()}")
# plot_metrics(y_true=df_rag["true_label"],y_pred=df_rag["prediction"],
#              digits=2,plot_name="Radboud (RAG)- report")