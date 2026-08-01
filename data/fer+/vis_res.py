import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
#
# # load dfs
# df_zs = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_zero_shot_34b.csv")
# df_clip_b = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_rag_0.10_llava34b_clipbase32.csv")
# df_clip_l = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_rag_0.10_llava34b_cliplarge14.csv")
#
#
# df_zs =df_zs.rename(columns={"predictions": "prediction"})
# df_clip_b =df_clip_b.rename(columns={"predictions": "prediction"})
# df_clip_l = df_clip_l.rename(columns={"predictions": "prediction"})
#
#
# # check that all predictions are in the same format as true labels
# print(f"the true labels: {sorted(df_zs['true_label'].unique().tolist())}")
# print(f"Predictions ZS: {sorted(df_zs['prediction'].unique().tolist())}")
# print(f"Predictions RAG CLIP-BASE: {sorted(df_clip_b['prediction'].unique().tolist())}")
# print(f"Predictions CLIP-LARGE: {sorted(df_clip_l['prediction'].unique().tolist())}")
#
# print(f"number of nulls ZS: {df_zs['prediction'].isnull().sum()}")
# print(f"number of nulls CLIP-BASE: {df_clip_b['prediction'].isnull().sum()}")
# print(f"number of nulls CLIP-LARGE%: {df_clip_l['prediction'].isnull().sum()}")
#
# # classification report
# plot_classification_report(true_classes_column=df_zs['true_label'],
#                            pred_classes_column=df_zs['prediction'],
#                            plot_name="ZERO SHOT - Classification Report")
#
# plot_classification_report(true_classes_column=df_clip_b['true_label'],
#                            pred_classes_column=df_clip_b['prediction'],
#                            plot_name="RAG (CLIP-BASE) - Classification Report")
#
# plot_classification_report(true_classes_column=df_clip_l['true_label'],
#                            pred_classes_column=df_clip_l['prediction'],
#                            plot_name="RAG (CLIP-LARGE) - Classification Report")
#
# # report
# plot_metrics(y_true=df_zs['true_label'], y_pred=df_zs['prediction'], digits=2,
#              plot_name="ZERO SHOT - Report", formats=("pdf", "png"))
#
# plot_metrics(y_true=df_clip_b['true_label'], y_pred=df_clip_b['prediction'], digits=2,
#              plot_name="RAG (CLIP-BASE) - Report", formats=("pdf", "png"))
#
# plot_metrics(y_true=df_clip_l['true_label'], y_pred=df_clip_l['prediction'], digits=2,
#              plot_name="RAG (CLIP-LARGE) - Report", formats=("pdf", "png"))
# # confusion matrix
#
# plot_confusion_matrix(true_classes=df_zs['true_label'],pred_classes=df_zs['prediction'],
#                       plot_name="ZERO SHOT - Confusion Matrix", formats=("pdf","png"))
#
#
# plot_confusion_matrix(true_classes=df_clip_b['true_label'],pred_classes=df_clip_b['prediction'],
#                       plot_name="RAG (CLIP-BASE) - Confusion Matrix", formats=("pdf","png"))
#
#
# plot_confusion_matrix(true_classes=df_clip_l['true_label'],pred_classes=df_clip_l['prediction'],
#                       plot_name="RAG (CLIP-LARGE) - Confusion Matrix", formats=("pdf","png"))
#
#
#
# # retriever evaluation
# clip_b = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_llava_rag_0.10_llava34b_clipbase32.csv"
# clip_l =  "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/llava_results/fer_plus_rag_0.10_llava34b_cliplarge14.csv"
# emonet = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/rag_results/rag_llava34b_emonet_conv1x1output_0.10.csv"
# resemotenet = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ResEmoteNet/rag_results_ResEmoteNet/ferplus_10%_ResEmoteNet_LLava34b_fc1Layer.csv"
# clip_lda = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/LDA/rag_results_lda/ferplus10%_llava34b_cliplarge14L_lda6components.csv"
# paths_list = [clip_b, clip_l, emonet, resemotenet, clip_lda]
# df = retrieval_quality_eval(csv_paths=paths_list, plot_name="Retrieval Quality Evaluation")
# parts = []
# for _, row in df.iterrows():
#     for col in df.columns:
#         parts.append(f"{col}: {row[col]}")
# print(", ".join(parts))
#
#
# # class distribution of FER+ for all experiment
# df_kb_50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_50%.csv")
# df_public_test_50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/public_test_50%.csv")
# df_private_test_50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/private_test_50%.csv")
# plot_label_distribution({"train": df_kb_50, "val": df_public_test_50, "test": df_private_test_50},
#                         normalize="percent", plot_name="Class Distribution - FER+")
#
#
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# visualize_sample_images(df_kb_50, seed=42, n_images=2, plot_name="Example images - FER+")
#
# # class distribution of FER+ for first experiment
# df_kb_processed = pd.read_csv( "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
# df_test_processed = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_test.csv")
# plot_label_distribution({"train": df_kb_processed, "test": df_test_processed},
#                         normalize="percent", plot_name="Class Distribution - FER+ (Experiment 1)")
#

