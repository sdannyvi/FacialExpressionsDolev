"""
Visualize Zero shot and RAG results using the fer plus dataset, and ResEmoteNet as a retriever,
llava34B as a generator.
"""
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
# import wandb

# try: 
#     wandb.login()
# except Exception: 
#     raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

# wandb.init(project="fer_plus_0.03_resemotenet", name="llava-resemotenet: fc1 layer",
#             notes="RAG using LLaVa and resemotenet with 5% KB",
#             tags=["RAG 5%", "llava", "resemotenet"])

# load dfs
rag_fc1 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ResEmoteNet/rag_results_ResEmoteNet/ferplus_10%_ResEmoteNet_LLava34b_fc1Layer.csv")
zs_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_zero_shot_34b.csv")

rag_fc1 = rag_fc1.rename(columns={"predictions": "prediction"})
zs_df = zs_df.rename(columns={"predictions": "prediction"})
# check that all predictions are in the same format as true labels
print(f"the true labels: {sorted(rag_fc1['true_label'].unique().tolist())}")
print(f"Predictions in RAG 5%: {rag_fc1['prediction'].unique().tolist()}")


# filter out nan rows
print(f"number of nulls in results for 10%: {rag_fc1['prediction'].isnull().sum()}")



# RAG 5%
cp_plot_rag5 = plot_classification_report(true_classes_column=rag_fc1['true_label'],
                                          pred_classes_column=rag_fc1['prediction'],
                                          plot_name="ResEmoteNet (FC1)- Classification Report")
# wandb.log({"Classification Report: RAG (5%) on FER+ using LLaVa34b and ResEmoteNet (FC1)": cp_plot_rag5})
cp_plot_rag5.show()
cp_plot_rag5.write_html("cp_plot_rag5.html")


cm_plot_rag5 = plot_confusion_matrix(true_classes=rag_fc1['true_label'],pred_classes=rag_fc1['prediction'],
                                     plot_name="ResEmoteNet (FC1) - Confusion Matrix", formats=("pdf","png"))


plot_metrics(y_true=rag_fc1['true_label'], y_pred=rag_fc1['prediction'], digits=2, plot_name="ResEmoteNet (FC1) - Report", formats=("pdf", "png"))


# # load kb 5%
# kb5_df  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.05.csv")
# retriever_plot_rag5 = f1_at_k(res_df=rag_fc1, kb_df=kb5_df, k=3,
#                               title_text="RAG 5%: Retrieval Evaluation as K=3 (ResEmoteNet (FC1))")
# retriever_plot_rag5.show()
#
# # plot images where RAG misclassified
# filtered_df_rag5 = get_misclassified_vs_correct(df_rag=rag_fc1, df_zs=zs_df, sample_size=10, is_rag_wrong=True)
# figures_miss_5 = plot_images_from_classes(df=filtered_df_rag5,
#                                           title="Misclassified Images by RAG(5%), but not by Zero-Shot",
#                                           is_rag_wrong=True)
# for key in figures_miss_5:
#     # title_for_wandb = f"Misclassified Images by RAG (5%), but not by Zero-Shot ({key})"
#     # wandb.log({title_for_wandb: figures_miss_5[key]})
#     figures_miss_5[key].show()
#
# # plot images where RAG correctly classified and zeroshot misclassified
# filtered_df_zs_5 = get_misclassified_vs_correct(df_rag=rag_fc1, df_zs=zs_df, sample_size=10, is_rag_wrong=False)
# figures_correct_5 = plot_images_from_classes(df=filtered_df_zs_5,
#                                              title="Misclassified Images by Zero-Shot, but not by RAG(5%)",
#                                              is_rag_wrong=False)
# for key in figures_correct_5:
#     # title_for_wandb = f"Misclassified Images by Zero-Shot, but not by RAG(5%) ({key})"
#     # wandb.log({title_for_wandb: figures_correct_5[key]})
#     figures_correct_5[key].show()
#
# # wandb.finish()


#
#
# # RAG 10%
# # try: 
#      wandb.login()
# # except Exception: 
#      raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

# # wandb.init(project="fer_plus_0.03_resemotenet", name="llava-resemotenet: fc3 layer",
# #             notes="RAG using LLaVa and Emonet with 10% KB",
# #             tags=["RAG 10%", "llava", "emonet"])
#
#
# cp_plot_rag10 = plot_classification_report(true_classes_column=rag_fc3['true_label'],
#                                            pred_classes_column=rag_fc3['predictions'],
#                                            title_text="Classification Report: RAG (5%) on FER+ using LLaVa34b and ResEmoteNet (FC3)")
# # wandb.log({"Classification Report: RAG (5%) on FER+ using LLaVa34b and ResEmoteNet (FC3)": cp_plot_rag10})
# cp_plot_rag10.show()
#
#
# cm_plot_rag10 = plot_confusion_matrix(true_classes_column=rag_fc3['true_label'],
#                                       pred_classes_column=rag_fc3['predictions'],
#                                       title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and ResEmoteNet (FC3)",
#                                       is_normalize=True)
#
# # wandb.log({"Confusion Matrix (Normalized): RAG (5%) on FER+ using LLaVA34b and ResEmoteNet (FC3)": cm_plot_rag10})
# cm_plot_rag10.show()
#
# cm_plot_rag10 = plot_confusion_matrix(true_classes_column=rag_fc3['true_label'],
#                                       pred_classes_column=rag_fc3['predictions'],
#                                       title_text="Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and ResEmoteNet (FC3)",
#                                       is_normalize=False)
#
# # wandb.log({"Confusion Matrix: RAG (5%) on FER+ using LLaVA34b and ResEmoteNet (FC3)": cm_plot_rag10})
# cm_plot_rag10.show()
#
#
# # results kb 10%
# kb10_df  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
# retriever_plot_rag10 = f1_at_k(res_df=rag_fc3, kb_df=kb10_df, k=3,
#                                title_text="RAG (5%): Retrieval Evaluation as K=3 (ResEmoteNet (FC3))")
# # wandb.log({"RAG (10%): Retrieval Evaluation as K=3 (ResEmoteNet (FC3))": retriever_plot_rag10})
# retriever_plot_rag10.show()
#
#
# # wandb.finish()