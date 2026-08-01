import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
from pathlib import Path


# loop through csv files in path
base_path = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/generator_selection")

# validate and visualize for each csv
for csv_path in base_path.glob("*.csv"):
    # get file name
    file_name = csv_path.stem.lower()

    print(f"Processing file: {csv_path.name}")

    # load csv
    df = pd.read_csv(csv_path)

    # validate and normalize prediction
    df = validate_results(df)
    print(f"true_label: {sorted(df['true_label'].unique().tolist())}")
    print(f"predictions: {sorted(df['prediction'].unique().tolist())}")

    # special normalization for OneVision Zero-shot
    is_special = ("onevision" in file_name) and ("zero" in file_name)
    classes_list = None
    if is_special:
        print(f"the number and percentage of occurrences for each unique prediction value:")
        counts = df["prediction"].value_counts()
        percentages = df["prediction"].value_counts(normalize=True) * 100
        result = pd.DataFrame({
            "count": counts,
            "percentage": percentages
        })
        print(result)

        print(f"convert invalid predictions to 'invalid' class.")
        true_label_original = df["true_label"].unique().tolist()
        true_label_new = true_label_original.copy()
        true_label_new.append("invalid")
        print(f"true label original: {sorted(true_label_original)}")
        print(f"true label new: {sorted(true_label_new)}")

        # change invalid predictions to "invalid" class
        mask_invalid = ~df["prediction"].isin(true_label_original)
        df.loc[mask_invalid, "prediction"] = "invalid"
        print("Total invalid predictions converted:", mask_invalid.sum())
        classes_list = true_label_new

    # plot name
    parts = file_name.split("_")
    model_part = " ".join(parts[:3])
    run_type = " ".join(parts[3:]).replace("rag", "RAG").replace("zero shot", "Zero-shot")
    plot_title = f"{model_part} ({run_type})"
    # visualize results
    plot_metrics(y_true=df["true_label"], y_pred=df["prediction"],
                 classes_list=classes_list, digits=2,
                 plot_name=plot_title + " - Report" )
    plot_confusion_matrix(true_classes=df["true_label"],
                          pred_classes=df["prediction"],
                          classes_list=classes_list,
                          plot_name=plot_title + " - Confusion matrix")


# class distribution
df_train = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_50%.csv")
df_test = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/public_test_50%.csv")
class_distribution_table(dfs={"train set (50%)": df_train, "public test (50%)": df_test},label_col="true_label",
                         file_name="generator selection - class distribution")