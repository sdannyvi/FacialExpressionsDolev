import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *

# load dfs
df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/rag_results/rag_llava34b_emonet_conv1x1output_0.10.csv")

df = df.rename(columns={"predictions": "prediction"})



# check that all predictions are in the same format as true labels
print(f"the true labels: {sorted(df['true_label'].unique().tolist())}")
print(f"the true labels: {sorted(df['prediction'].unique().tolist())}")

print(f"number of nulls ZS: {df['prediction'].isnull().sum()}")

# classification report
plot_classification_report(true_classes_column=df['true_label'],
                           pred_classes_column=df['prediction'],
                           plot_name="Emonet (CONV1X1) - Classification Report")


# report
plot_metrics(y_true=df['true_label'], y_pred=df['prediction'], digits=2,
             plot_name="Emonet (CONV1X1) - Report", formats=("pdf", "png"))


# confusion matrix

plot_confusion_matrix(true_classes=df['true_label'],pred_classes=df['prediction'],
                      plot_name="Emonet (CONV1X1) - Confusion Matrix", formats=("pdf","png"))


