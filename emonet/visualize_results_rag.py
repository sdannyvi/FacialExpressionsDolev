"""
Visualize Zero shot and RAG results using the fer plus dataset, while during preprocessing phase,
duplicates removal was done using DBSCAN with epsilon value 0.03 as a distance threshold.
Using Emonet as a retriever and LLaVA as a generator.
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

wandb.init(project="fer_plus_0.03_emonet", name="llava-emonet: rag 5%",
            notes="RAG using LLaVa and Emonet with 5% KB",
            tags=["RAG 5%", "llava", "Emonet"])

# load dfs
rag_5_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/rag_results/rag_llava34b_emonet_conv1x1output_0.05.csv")
rag_10_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/rag_results/rag_llava34b_emonet_conv1x1output_0.10.csv")
zs_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_zero_shot_34b.csv")

# check that all predictions are in the same format as true labels
print(f"the true labels: {sorted(rag_5_df['true_label'].unique().tolist())}")
print(f"Predictions in RAG 5%: {rag_5_df['predictions'].unique().tolist()}")
print(f"Predictions in RAG 10%: {rag_10_df['predictions'].unique().tolist()}")

# filter out nan rows
print(f"number of nulls in results for 5%: {rag_5_df['predictions'].isnull().sum()}")
print(f"number of nulls in results for 10%: {rag_10_df['predictions'].isnull().sum()}")



# RAG 5%
cp_plot_rag5 = plot_classification_report(true_classes_column=rag_5_df['true_label'],
                                          pred_classes_column=rag_5_df['predictions'],
                                          title_text="Classification Report: RAG (5%) on FER+ using LLaVa34b and Emonet")
wandb.log({"Classification Report: RAG (5%) on FER+ using LLaVa34b and Emonet": cp_plot_rag5})
cp_plot_rag5.show()


cm_plot_rag5 = plot_confusion_matrix(true_classes_column=rag_5_df['true_label'],
                                     pred_classes_column=rag_5_df['predictions'],
                                     title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and Emonet",
                                     is_normalize=True)

wandb.log({"Confusion Matrix (Normalized): RAG (5%) on FER+ using LLaVA34b and Emonet": cm_plot_rag5})
cm_plot_rag5.show()

cm_plot_rag5 = plot_confusion_matrix(true_classes_column=rag_5_df['true_label'],
                                     pred_classes_column=rag_5_df['predictions'],
                                     title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and Emonet",
                                     is_normalize=False)

wandb.log({"Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and Emonet": cm_plot_rag5})
cm_plot_rag5.show()


# load kb 5%
kb5_df  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.05.csv")
retriever_plot_rag5 = f1_at_k(res_df=rag_5_df, kb_df=kb5_df, k=3,
                              title_text="RAG 5%: Retrieval Evaluation as K=3 (Emonet)")
retriever_plot_rag5.show()

# plot images where RAG misclassified
filtered_df_rag5 = get_misclassified_vs_correct(df_rag=rag_5_df, df_zs=zs_df, sample_size=10, is_rag_wrong=True)
figures_miss_5 = plot_images_from_classes(df=filtered_df_rag5,
                                          title="Misclassified Images by RAG(5%), but not by Zero-Shot",
                                          is_rag_wrong=True)
for key in figures_miss_5:
    title_for_wandb = f"Misclassified Images by RAG (5%), but not by Zero-Shot ({key})"
    wandb.log({title_for_wandb: figures_miss_5[key]})
    figures_miss_5[key].show()

# plot images where RAG correctly classified and zeroshot misclassified
filtered_df_zs_5 = get_misclassified_vs_correct(df_rag=rag_5_df, df_zs=zs_df, sample_size=10, is_rag_wrong=False)
figures_correct_5 = plot_images_from_classes(df=filtered_df_zs_5,
                                             title="Misclassified Images by Zero-Shot, but not by RAG(5%)",
                                             is_rag_wrong=False)
for key in figures_correct_5:
    title_for_wandb = f"Misclassified Images by Zero-Shot, but not by RAG(5%) ({key})"
    wandb.log({title_for_wandb: figures_correct_5[key]})
    figures_correct_5[key].show()

wandb.finish()




# RAG 10%
try: 
    wandb.login()
except Exception: 
    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

wandb.init(project="fer_plus_0.03_emonet", name="llava-emonet: rag 10%",
            notes="RAG using LLaVa and Emonet with 10% KB",
            tags=["RAG 10%", "llava", "emonet"])


cp_plot_rag10 = plot_classification_report(true_classes_column=rag_10_df['true_label'],
                                           pred_classes_column=rag_10_df['predictions'],
                                           title_text="Classification Report: RAG (10%) on FER+ using LLaVa34b and Emonet")
wandb.log({"Classification Report: RAG (10%) on FER+ using LLaVa34b and Emonet": cp_plot_rag10})
cp_plot_rag10.show()


cm_plot_rag10 = plot_confusion_matrix(true_classes_column=rag_10_df['true_label'],
                                      pred_classes_column=rag_10_df['predictions'],
                                      title_text="Confusion Matrix: RAG (10%) on FER+ using LLaVA34b and Emonet",
                                      is_normalize=True)

wandb.log({"Confusion Matrix (Normalized): RAG (10%) on FER+ using LLaVA34b and Emonet": cm_plot_rag10})
cm_plot_rag10.show()

cm_plot_rag10 = plot_confusion_matrix(true_classes_column=rag_10_df['true_label'],
                                      pred_classes_column=rag_10_df['predictions'],
                                      title_text="Confusion Matrix: RAG (10%) on FER+ using LLaVA34b and Emonet",
                                      is_normalize=False)

wandb.log({"Confusion Matrix: RAG (10%) on FER+ using LLaVA34b and Emonet": cm_plot_rag10})
cm_plot_rag10.show()


# results kb 10%
kb10_df  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
retriever_plot_rag10 = f1_at_k(res_df=rag_10_df, kb_df=kb10_df, k=3,
                               title_text="RAG (10%): Retrieval Evaluation as K=3 (Emonet)")
wandb.log({"RAG (10%): Retrieval Evaluation as K=3 (Emonet)": retriever_plot_rag10})
retriever_plot_rag10.show()


wandb.finish()