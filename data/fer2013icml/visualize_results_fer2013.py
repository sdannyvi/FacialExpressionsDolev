"""
Visualize results to all frameworks on FER plus:
llava zero shot, clip zero shot, llava rag with 5%, llava with rag 10% and upload it to wandb.
"""
# ...LLaVa.llava_rag.vis_results import *
from Thesis.VLMs.LLaVa.llava_rag import vis_results as vis
import pandas as pd
import wandb
try: 
      wandb.login()
except Exception: 
      raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")


#### LLaVa - Zero-Shot ####

# wandb.init(project="fer_2013", name="llava: zero-shot",
#             notes="Zero-Shot using LLaVa",
#             tags=["zero shot", "llava"])

# import results df
zero_llava_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LLaVa/llava_rag/results/llava_zero_shot_34b.csv"
zero_llava_df = pd.read_csv(zero_llava_path)
print(f"zero shot llava columns: {zero_llava_df.columns.tolist()}")
print(f"unique values in predictions:\n{zero_llava_df['predictions'].unique().tolist()}"
      f"\nand in true labels:\n{zero_llava_df['true_label'].unique().tolist()}")
print(zero_llava_df["true_label"].value_counts(normalize=True))
class_report_plot = vis.plot_classification_report(true_classes_column=zero_llava_df["true_label"],
                               pred_classes_column=zero_llava_df["predictions"], title_text="Report: FER 2013 in Zero-Shot using LLaVa 34B")
# wandb.log({"Classification Report: FER 2013, zero-shot using LLaVa 34B": class_report_plot})
class_report_plot.show()

hist_plot = vis.hist_visualization(df=zero_llava_df, plot_title="Test Set Distribution - FER 2013")
# wandb.log({"Test Set Distribution: FER 2013": hist_plot})
hist_plot.show()

cm_plot = vis.plot_confusion_matrix(true_classes_column=zero_llava_df["true_label"], pred_classes_column=zero_llava_df["predictions"],
                                title_text="Confusion Matrix: FER 2013 in Zero-Shot using LLaVa 34B",
                                is_normalize=False)
# wandb.log({"Confusion Matrix: FER 2013 in Zero-Shot using LLaVa 34B": cm_plot})
cm_plot.show()

cm_plot_norm = vis.plot_confusion_matrix(true_classes_column=zero_llava_df["true_label"], pred_classes_column=zero_llava_df["predictions"],
                                title_text="Confusion Matrix: FER 2013 in Zero-Shot using LLaVa 34B",
                                is_normalize=True)
# wandb.log({"Normalized Confusion Matrix: FER 2013 in Zero-Shot using LLaVa 34B": cm_plot_norm})
cm_plot_norm.show()
# wandb.finish()


#### LLaVa - RAG (KB=5%) ####

# # wndb run
# wandb.init(project="fer_2013", name="llava: RAG(5%)",
#             notes="RAG using LLaVa-Next (34B) as a generator and CLIP base as a retriever, "
#                   "with KB size 5% out of the data.",
#             tags=["RAG", "llava"])


# import df
path_rag_5 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LLaVa/llava_rag/results/llava_rag_0.05_34b.csv"
llava_rag_5_df = pd.read_csv(path_rag_5)
print(f"LLaVa RAG 5% columns: {llava_rag_5_df.columns.tolist()}")
print(f"Unique true labels:\n{llava_rag_5_df['true_label'].unique().tolist()}"
      f"\nUnique predictions:\n{llava_rag_5_df['predictions'].unique().tolist()}")
# replace each "anger" with "angry" to match the true labels
llava_rag_5_df["predictions"] = llava_rag_5_df["predictions"].replace("anger", "angry")
print(f"Unique true labels:\n{llava_rag_5_df['true_label'].unique().tolist()}"
      f"\nUnique predictions:\n{llava_rag_5_df['predictions'].unique().tolist()}")

class_reports_rag_5 = vis.plot_classification_report(true_classes_column=llava_rag_5_df['true_label'],
                                                     pred_classes_column=llava_rag_5_df['predictions'],
                                                     title_text="Report: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever")
# wandb.log({"Report: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever": class_reports_rag_5})
class_reports_rag_5.show()

# confusion matrix
cm_plot = vis.plot_confusion_matrix(true_classes_column=llava_rag_5_df["true_label"],
                                    pred_classes_column=llava_rag_5_df["predictions"],
                                    title_text="Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B",
                                    is_normalize=False)
# wandb.log({"Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever": cm_plot})
cm_plot.show()

cm_plot_norm = vis.plot_confusion_matrix(true_classes_column=llava_rag_5_df["true_label"],
                                         pred_classes_column=llava_rag_5_df["predictions"],
                                         title_text="Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B",
                                         is_normalize=True)
# wandb.log({"Normalized Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever": cm_plot_norm})
cm_plot_norm.show()


# retriever estimation
# knowledge base df
kb_df_5 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LLaVa/llava_rag/kb_sets/kb_0.05_rag.csv")
table_retriever = vis.f1_at_k(res_df=llava_rag_5_df, kb_df=kb_df_5,
                              k=3, title_text=f"Retrieval Evaluation <br> at K=3 (CLIP Base)")
# wandb.log({"Retrieval Evaluation <br> at K=3 (CLIP Base)": table_retriever})
table_retriever.show()
# wandb.finish()






#### LLaVa - RAG (KB=10%) ####

# # wndb run
# wandb.init(project="fer_2013", name="llava: RAG(10%)",
#             notes="RAG using LLaVa-Next (34B) as a generator and CLIP base as a retriever, "
#                   "with KB size 10% out of the data.",
#             tags=["RAG", "llava"])


# import df
path_rag_10 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LLaVa/llava_rag/results/llava_rag_0.1_34b.csv"
llava_rag_10_df = pd.read_csv(path_rag_10)
print(f"LLaVa RAG 10% columns: {llava_rag_10_df.columns.tolist()}")
print(f"Unique true labels:\n{llava_rag_10_df['true_label'].unique().tolist()}"
      f"\nUnique predictions:\n{llava_rag_10_df['predictions'].unique().tolist()}")
print(f"number of rows in each value: {llava_rag_10_df['predictions'].value_counts()}")
llava_rag_10_df["predictions"] = llava_rag_10_df["predictions"].replace("anger", "angry")
print(f"Unique true labels:\n{sorted(llava_rag_10_df['true_label'].unique().tolist())}"
      f"\nUnique predictions:\n{sorted(llava_rag_10_df['predictions'].unique().tolist())}")

# classification report
class_reports_rag_10 = vis.plot_classification_report(true_classes_column=llava_rag_10_df['true_label'],
                                                     pred_classes_column=llava_rag_10_df['predictions'],
                                                     title_text="Report: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as Retriever")
# wandb.log({"Report: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever (KB=10%)": class_reports_rag_10})
class_reports_rag_10.show()

# confusion matrix
cm_plot = vis.plot_confusion_matrix(true_classes_column=llava_rag_10_df["true_label"],
                                    pred_classes_column=llava_rag_10_df["predictions"],
                                    title_text="Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B (KB=10%)",
                                    is_normalize=False)
# wandb.log({"Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever (KB=10%)": cm_plot})
cm_plot.show()

cm_plot_norm = vis.plot_confusion_matrix(true_classes_column=llava_rag_10_df["true_label"],
                                         pred_classes_column=llava_rag_10_df["predictions"],
                                         title_text="Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B (KB=10%)",
                                         is_normalize=True)
# wandb.log({"Normalized Confusion Matrix: FER 2013 in RAG Framework using LLaVa 34B and CLIP base as a Retriever (KB=10%)": cm_plot_norm})
cm_plot_norm.show()


# retriever estimation
# knowledge base df
kb_df_10 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LLaVa/llava_rag/kb_sets/kb_0.1_rag.csv")
table_retriever = vis.f1_at_k(res_df=llava_rag_10_df, kb_df=kb_df_10,
                              k=3, title_text=f"Retrieval Evaluation <br> at K=3 (CLIP Base), KB=10%")
# wandb.log({"Retrieval Evaluation <br> at K=3 (CLIP Base), KB=10%": table_retriever})
table_retriever.show()
# wandb.finish()


### Zero-Shot CLIP base ###


# wandb.init(project="fer_2013", name="clip base: zero-shot",
#             notes="Zero-Shot using CLIP base",
#             tags=["zero shot", "CLIP"])

# import results
zero_shot_clip_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/clip/results/clip_base34_zero_shot.csv"
clip_res_df = pd.read_csv(zero_shot_clip_path)
print(f"zero shot llava columns: {clip_res_df.columns.tolist()}")
print(f"unique values in predictions:\n{sorted(clip_res_df['predictions'].unique().tolist())}"
      f"\nand in true labels:\n{sorted(clip_res_df['true_label'].unique().tolist())}")
print(clip_res_df["true_label"].value_counts(normalize=True))

# classification report
class_report_plot = vis.plot_classification_report(true_classes_column=clip_res_df["true_label"],
                                                   pred_classes_column=clip_res_df["predictions"],
                                                   title_text="Report: FER 2013 in Zero-Shot using CLIP base")
# wandb.log({"Classification Report: FER 2013 in Zero-Shot using CLIP base": class_report_plot})
class_report_plot.show()

cm_plot = vis.plot_confusion_matrix(true_classes_column=clip_res_df["true_label"],
                                    pred_classes_column=clip_res_df["predictions"],
                                    title_text="Confusion Matrix: FER 2013 in Zero-Shot using CLIP base",
                                    is_normalize=False)
# wandb.log({"Confusion Matrix: FER 2013 in Zero-Shot using CLIP base": cm_plot})
cm_plot.show()

cm_plot_norm = vis.plot_confusion_matrix(true_classes_column=clip_res_df["true_label"],
                                         pred_classes_column=clip_res_df["predictions"],
                                         title_text="Confusion Matrix: FER 2013 in Zero-Shot using CLIP base",
                                         is_normalize=True)
# wandb.log({"Normalized Confusion Matrix: FER 2013 in Zero-Shot using CLIP base": cm_plot_norm})
cm_plot_norm.show()
# wandb.finish()