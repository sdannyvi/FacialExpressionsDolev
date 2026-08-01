"""
Visualize Zero shot and RAG results using the fer plus dataset, and Clip-l14 as a retriever, using LDA
for reducing dimensionality.
10% RAG examples.
llava34B as a generator.
"""
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
import wandb

try: 
    wandb.login()
except Exception: 
    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

wandb.init(project="fer_plus_0.03_LDA", name="test",
            notes="RAG using LLaVa and CLIP-L14 with LDA and 5% KB",
            tags=["RAG 5%", "llava", "CLIP-L14", "LDA"])

# load dfs
df_res = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LDA/rag_results_lda/ferplus5%_llava34b_cliplarge14L_lda6components.csv")

zs_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_zero_shot_34b.csv")

# check that all predictions are in the same format as true labels
print(f"the true labels: {sorted(df_res['true_label'].unique().tolist())}")
print(f"Predictions in RAG 5%: {sorted(df_res['predictions'].unique().tolist())}")

# is all file paths are compatible
false_count = (df_res['file_path'] != df_res['query_file_path']).sum()
print(f"Number of mismatches: {false_count}")



# RAG 5%
plot_class_report = plot_classification_report(true_classes_column=df_res['true_label'],
                                          pred_classes_column=df_res['predictions'],
                                          title_text="Classification Report: RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA")
wandb.log({"Classification Report: RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA": plot_class_report})
plot_class_report.show()


plot_cm = plot_confusion_matrix(true_classes_column=df_res['true_label'],
                                     pred_classes_column=df_res['predictions'],
                                     title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA",
                                     is_normalize=True)

wandb.log({"Confusion Matrix (Normalized): RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA": plot_cm})
plot_cm.show()

plot_cm_not_norm = plot_confusion_matrix(true_classes_column=df_res['true_label'],
                                     pred_classes_column=df_res['predictions'],
                                     title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA",
                                     is_normalize=False)

wandb.log({"Confusion Matrix: RAG (5%) on FER+ using LLaVa34b and CLIP-L14 with LDA": plot_cm_not_norm})
plot_cm_not_norm.show()









# load kb 5%
kb10_df  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
retriever_plot_rag10 = f1_at_k(res_df=df_res, kb_df=kb10_df, k=3,
                              title_text="RAG 5%: Retrieval Evaluation as K=3 (CLIP and LDA)")
retriever_plot_rag10.show()

# plot images where RAG misclassified
filtered_df_rag5 = get_misclassified_vs_correct(df_rag=df_res, df_zs=zs_df, sample_size=10, is_rag_wrong=True)
figures_miss_5 = plot_images_from_classes(df=filtered_df_rag5,
                                          title="Misclassified Images by RAG(5%), but not by Zero-Shot",
                                          is_rag_wrong=True)
for key in figures_miss_5:
    title_for_wandb = f"Misclassified Images by RAG (5%), but not by Zero-Shot ({key})"
    wandb.log({title_for_wandb: figures_miss_5[key]})
    figures_miss_5[key].show()

# plot images where RAG correctly classified and zeroshot misclassified
filtered_df_zs_5 = get_misclassified_vs_correct(df_rag=df_res, df_zs=zs_df, sample_size=10, is_rag_wrong=False)
figures_correct_5 = plot_images_from_classes(df=filtered_df_zs_5,
                                             title="Misclassified Images by Zero-Shot, but not by RAG(5%)",
                                             is_rag_wrong=False)
for key in figures_correct_5:
    title_for_wandb = f"Misclassified Images by Zero-Shot, but not by RAG(5%) ({key})"
    wandb.log({title_for_wandb: figures_correct_5[key]})
    figures_correct_5[key].show()

wandb.finish()



