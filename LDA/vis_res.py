import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *

# load dfs
df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LDA/rag_results_lda/ferplus10%_llava34b_cliplarge14L_lda6components.csv")

df = df.rename(columns={"predictions": "prediction"})

# True only if every row matches (including the same order)
same_order = df["file_path"].astype(str).reset_index(drop=True).equals(
    df["query_file_path"].astype(str).reset_index(drop=True)
)

print("yes, same order" if same_order else "not in the same order")


# check that all predictions are in the same format as true labels
print(f"the true labels: {sorted(df['true_label'].unique().tolist())}")
print(f"the true labels: {sorted(df['prediction'].unique().tolist())}")

print(f"number of nulls ZS: {df['prediction'].isnull().sum()}")

# classification report
plot_classification_report(true_classes_column=df['true_label'],
                           pred_classes_column=df['prediction'],
                           plot_name="CLIP + LDA - Classification Report")


# report
plot_metrics(y_true=df['true_label'], y_pred=df['prediction'], digits=2,
             plot_name="CLIP + LDA - Report", formats=("pdf", "png"))


# confusion matrix

plot_confusion_matrix(true_classes=df['true_label'],pred_classes=df['prediction'],
                      plot_name="CLIP + LDA - Confusion Matrix", formats=("pdf","png"))
