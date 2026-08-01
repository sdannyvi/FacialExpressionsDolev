import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os
import plotly.express as px
import torch
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import random
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import wandb
from Thesis.VLMs.data_cleaning.remove_duplicates import find_duplicates_tuning, vis_res_eps, remove_duplicates, plot_single_cluster
from Thesis.VLMs.LLaVa.llava_rag.vis_results import hist_visualization

# login to wandb
#try: 
#    wandb.login()
#except Exception: 
#    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")


# # create a project for fer plus
# wandb.init(
#     project="fer_plus",
#     entity= "FER-VLM",
#     name="preprocess_visualization",
#     tags=['preprocess','data distribution'],
#     notes="preprocess fer plus",  # optional notes
# )



# read data files
icml_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013/icml_data.csv")
fer_plus_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/fer2013new.csv")

# change the paths of the icml data
icml_data['file_path'] = icml_data['file_path'].str.replace("Vision-Language-Models", "VLMs")
print(icml_data['file_path'].head())


# print columns for each data
print(f"icml data columns: \n {icml_data.columns.tolist()}")
print(f"fer plus columns: \n {fer_plus_data.columns.tolist()}")

# print data types of columns
print(f"icml data types: \n {icml_data.dtypes}")
print(f"fer plus types: \n {fer_plus_data.dtypes}")

# reset indices before concatenation
icml_data.reset_index(drop=True, inplace=True)
fer_plus_data.reset_index(drop=True, inplace=True)

# concatenate dataframes
combined_df = pd.concat([icml_data, fer_plus_data], axis=1)

# drop unnecessary columns
combined_df.drop(columns=['Usage', 'Image name', 'true_label', 'coded_true_label'], inplace=True)
print(f"combined data columns: \n{combined_df.columns.tolist()}")
print(f"combined data dtypes: \n{combined_df.dtypes}")
# save raw csv in fer+ directory
# combined_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_raw_data.csv", index=False)

# nulls
for col in combined_df.columns:
    if combined_df[col].isnull().sum() != 0:
        print(f"NULLS in column '{col}': {combined_df[col].isnull().sum()}")

# delete rows with NF != 0
print(f"Number of samples: {len(combined_df)}, number of samples to drop (NF!=0): {len(combined_df[combined_df['NF']!=0])}")
combined_df = combined_df[combined_df['NF'] == 0]
print(f"Number of samples after drop NF: {len(combined_df)}")

# convert votes to percentage
emotion_cols = ['neutral','happiness', 'surprise', 'sadness', 'anger',
                'disgust', 'fear', 'contempt', 'unknown']
combined_df[emotion_cols] = combined_df[emotion_cols].div(10).round(3)
pd.set_option('display.max_columns', None)
print(combined_df[emotion_cols].head(2))

# save the maximum label and its value percentage
combined_df['true_label'] = combined_df[emotion_cols].idxmax(axis=1)
combined_df['votes_percentage'] = combined_df[emotion_cols].max(axis=1)
print(f"{combined_df[['true_label', 'votes_percentage']].head(2)}")

# change the name of the classes to be written exactly the same as the labels in fer 2013
print(f"original labels: \n{icml_data['true_label'].unique().tolist()}")
label_map_match_2013 = {
    'neutral': 'neutral',
    'happiness': 'happy',
    'surprise': 'surprise',
    'sadness': 'sad',
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'unknown': 'unknown',
    'contempt': 'contempt'
}
print(f"Before changing label values: \n{combined_df['true_label'].unique().tolist()}")
combined_df['true_label'] = combined_df['true_label'].map(label_map_match_2013)
print(f"After changing label values: \n{combined_df['true_label'].unique().tolist()}")


# define aa function for histogram distribution visualization
def hist_visualization(df, plot_title):
    """
    df (DataFrame): dataframe to plot, must contain a column name "true_label".
    plot_title (String): the title of the plot.
    """
    # calculate the number and percentage of each label in the data
    label_count = df['true_label'].value_counts()
    label_percentage = (label_count / label_count.sum()) * 100

    # create a text to be shown above each bar
    text = label_percentage.round(2).astype(str) + '%<br>(' + label_count.astype(str) + ')'

    fig = px.bar(x=label_percentage.index, y=label_percentage.values, text=text)
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(width=900, height=650,
                      title=dict(text=f"<b>{plot_title}<b>", x=0.5, font=dict(size=24)),
                      xaxis_title=dict(text="Class", font=dict(size=18)),
                      yaxis_title=dict(text="Percentage", font=dict(size=18)),
                      yaxis=dict(range=[0, 100]),
                      # yaxis=dict(range=[0, label_percentage.max() + 5]),
                      xaxis=dict(tickfont=dict(size=15))
                      )
    return fig


# plot histogram before removing rows by a threshold of 0.8
fig = hist_visualization(df=combined_df, plot_title="Data Distribution Before Filtering by a Threshold")
fig.show()

# keep only the rows where maximum vote is 0.8 or higher
filtered_combined_df = combined_df[combined_df['votes_percentage'] >= 0.8]
filtered_combined_df = filtered_combined_df.reset_index(drop=True)
fig = hist_visualization(df=filtered_combined_df, plot_title="Data Distribution After Filtering by a Threshold of 0.8")
fig.show()
print(f"Total samples before filtering: {len(combined_df)}")
print(f"Total samples after filtering by threshold of 0.8: {len(filtered_combined_df)}")

# delete rows with labels: unknown, contempt, disgust
filtered_combined_df = filtered_combined_df[~filtered_combined_df['true_label'].isin(['unknown', 'disgust', 'contempt'])]
filtered_combined_df = filtered_combined_df.reset_index(drop=True)
fig = hist_visualization(df=filtered_combined_df, plot_title="Data Distribution with Columns to use")
fig.show()

# adding a coded label column
label_code_map = {
    'neutral': 6,
    'sad': 4,
    'happy': 3,
    'surprise': 5,
    'angry': 0,
    'fear': 2,
    'disgust': 1
}
filtered_combined_df['coded_true_label'] = filtered_combined_df['true_label'].map(label_code_map)

# print the columns and remove redundant
print(f"columns: \n{filtered_combined_df.columns.tolist()}")
print(filtered_combined_df['NF'].unique().tolist())
cols_to_drop = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
filtered_combined_df.drop(columns=cols_to_drop, inplace=True)
print(f"Filtered combined df: {filtered_combined_df.columns.tolist()}")
filtered_combined_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_after_threshold.csv", index=False)


# removing duplicates
dup_df = filtered_combined_df.copy(deep=True).reset_index(drop=True)

# eps list
eps_list = [0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
npz_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_clustering_embeddings.npz"
npz_file = find_duplicates_tuning(df=dup_df,eps_list=eps_list, save_to_path=npz_path)

plots_list = vis_res_eps(res_npz=npz_file,eps_list=eps_list, n_clusters_to_sample=5,seed=1)
for plot in plots_list:
    plot.show()

# print the size of the clusters in descending order, all clusters with more than 10 samples
npz_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_clustering_embeddings.npz"
data = np.load(npz_path, allow_pickle=True)
cluster_labels = data['clusters_eps_003']
file_paths = data['file_path']
# create a df
df_clusters = pd.DataFrame({
    'file_path': file_paths,
    'cluster_id': cluster_labels
})
# group by cluster id and count samples in each group
cluster_counts = df_clusters.groupby('cluster_id').size().reset_index(name='count')
# only clusters with more than 10 samples
filtered_clusters = cluster_counts[cluster_counts['count'] > 10]
filtered_clusters = filtered_clusters.sort_values(by='count', ascending=False)
print(filtered_clusters)


## delete after
npz_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_clustering_embeddings.npz"
data = np.load(npz_path, allow_pickle=True)
dup_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_after_threshold.csv")
## delete after
# plot the images in a specific cluster
plot = plot_single_cluster(df=df_clusters,cluster_id=76)
plot.show()

# choose which clusters to ignore when removing duplicates - meaning, don't remove duplicates from these clusters
excluded_clusters = [7,1]


# remove duplicates
after_duplicates_df = remove_duplicates(res_npz=data, eps_val=0.03, df_to_clean=dup_df, excluded_cluster_ids=excluded_clusters)
after_duplicates_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_after_duplicates.csv",
                           index=False)

# import the after duplicated df
after_duplicates_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_after_duplicates.csv")
fig = hist_visualization(df=after_duplicates_df, plot_title="Data Distribution After Removing Duplicates")
fig.show()