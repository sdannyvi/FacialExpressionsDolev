import pandas as pd
import sys
import os

from Thesis.VLMs.LLaVa.llava_rag.vis_results import hist_visualization

sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
# from Thesis.VLMs.data_cleaning.retinaface_cleaning_tuning import tuning_confidence_thresh
# from Thesis.VLMs.data_cleaning.data_cleaning_tuning_visual import plot_confidence_thresh_curve
from Thesis.VLMs.data_cleaning.visualize_images import random_samples_plot
import wandb


# # run retinaface over different levels of confidence and find out how many images kept
# df_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_after_duplicates.csv"
# fer_plus_set = pd.read_csv(df_path)
# print(f"number of sampled before cleaning with retinaface: {len(fer_plus_set)}")
# save_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_retinaface_tuning.csv"
# res_tuning = tuning_confidence_thresh(df=fer_plus_set, save_to_path=save_path)
#
# visualize tuning - line plot
# save_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_retinaface_tuning.csv"
# res_tuning = pd.read_csv(save_path)
#
# try: 
#    wandb.login()
# except Exception: 
#    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")
# wandb.init(project="fer_plus_0.03", name="Tuning confidence threshold",
#              notes="using RetinaFace model to filter out non-face images, and defining the confidence threshold for Retinaface",
#              tags=["data cleaning", "retinaface", "fer plus"])
#
# fig_keep = plot_confidence_thresh_curve(res_tuning_df=res_tuning, keep=True)
# fig_drop = plot_confidence_thresh_curve(res_tuning_df=res_tuning, keep=False)
# fig_keep.write_image("RetinafaceTuning_keep.png", height=600, width=1200)
# fig_drop.write_image("RetinafaceTuning_drop.png", height=600, width=1200)
# wandb.log({"Data to Keep by Confidence Threshold": fig_keep})
# wandb.log({"Data to Drop by Confidence Threshold": fig_drop})

# run retinaface using 3 confidence levels and save the df under files_preprocess



# 1. visualize 100 random images from the original data (after duplicates)
plt_original = random_samples_plot(df=pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/fil"
                                                   "es_preprocess/ferplus_after_duplicates.csv"),
                                                    n=100, title='Original Data')
plt_original.show()

# 2. visualize 100 random images from confidence 0.5
# sample from the cleaned data - predicted as a face
plt_conf5 = random_samples_plot(df=pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/"
                                               "fer+/files_preprocess/ferplus_cleaned_conf5.csv"),
                                                n=100, title='Confidence 0.5: Images classified as a Face')
plt_conf5.show()


# 3. visualize 100 random images from confidence 0.6
plt_conf6 = random_samples_plot(df=pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/"
                                               "files_preprocess/ferplus_cleaned_conf6.csv"),
                                                n=100, title='Confidence 0.6: Images classified as a Face')
plt_conf6.show()

# 4. visualize 100 random images from confidence 0.7
plt_conf7 = random_samples_plot(df=pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/"
                                               "files_preprocess/ferplus_cleaned_conf7.csv"),
                                                n=100, title='Confidence 0.7: Images classified as a Face')
plt_conf7.show()

# # for each confidence plot the histogram to see how many images are in each class
# plt_hist = hist_visualization(df=pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/"
#                     "files_preprocess/ferplus_cleaned_conf6.csv"),
#                     plot_title='Class Distribution After using RetinaFace')
# plt_hist.show()
# visualize images from both results



# run retinaface using the selected confidence threshold