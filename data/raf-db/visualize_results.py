import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *


df_rag = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/rag_rafd.csv")
df_zero_shot = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/zero_shot_raf_db.csv")


print("validate rag:")
df_rag = validate_results(df_rag)


# visualize results
plot_metrics(y_true=df_rag["true_label"], y_pred=df_rag["prediction"], digits=2, formats=("pdf",),
             plot_name= "RAF-DB (RAG) - Report")
plot_confusion_matrix(true_classes=df_rag["true_label"],
                      pred_classes=df_rag["prediction"], formats=("pdf",),
                      plot_name="RAF-DB (RAG) - Confusion matrix")


print(f"validate zero shot:")
df_zero_shot = validate_results(df_zero_shot)


# visualize results
plot_metrics(y_true=df_zero_shot["true_label"], y_pred=df_zero_shot["prediction"], digits=2, formats=("pdf",),
             plot_name= "RAF-DB (Zero-shot) - Report")
plot_confusion_matrix(true_classes=df_zero_shot["true_label"],
                      pred_classes=df_zero_shot["prediction"], formats=("pdf",),
                      plot_name="RAF-DB (Zero-shot) - Confusion matrix")

# class distribution
df_train = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/train_set.csv")
df_test = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/test_set.csv")
class_distribution_table(dfs={"train set": df_train, "test set": df_test},label_col="true_label",
                         file_name="raf-db - class distribution")



