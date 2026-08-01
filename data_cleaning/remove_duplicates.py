"""
function to remove duplicates:
1. find duplicates tuning: get the dataframe and eps list, and conduct clustering with different eps values
2. vis res eps: gets the results from the prev function and plot for each es value, returns a list of plots
3. remove duplicates: gets the df to clean, the clustering npz results and the selected eps values, and a list of clusters
                    ids to exclude from duplicate removal (for  example too large clusters that are not contain duplicates).
                    and returns the cleaned df.
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os
import torch
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import random
import numpy as np
import seaborn as sns


# function to extract embeddings using CLIP model
def get_clip_embedding(image_path, model, processor):
    """
    Gets the path of an image, the loaded model and its processor, and extract embeddings from the image.
    returns: embeddings
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze().cpu().numpy()




def plot_clusters_grid(file_paths_list, clusters_list, eps_val):
    """
    file_paths_list: List of image paths
    clusters_list: List of cluster IDs (same length as file_paths_list)
    """

    # Group images by cluster
    cluster_to_images = {}
    for img_path, cluster_id in zip(file_paths_list, clusters_list):
        if cluster_id not in cluster_to_images:
            cluster_to_images[cluster_id] = []
        cluster_to_images[cluster_id].append(img_path)

    # create a color map
    colors = sns.color_palette("tab10", len(cluster_to_images.keys()))
    color_map = {}
    for i, cluster_id in enumerate(cluster_to_images.keys()):
        color_map[cluster_id] = colors[i]


    # Calculate grid size
    n_clusters = len(cluster_to_images.keys())
    max_images_per_cluster = max(len(images) for images in cluster_to_images.values())
    width = max(max_images_per_cluster * 1.6, 6)  # minimum width = 6
    height = max(n_clusters * 1.6, 10)  # maximum height = 10

    fig, ax = plt.subplots(figsize=(width, height))

    # Loop and plot images
    for row_idx, cluster_id in enumerate(cluster_to_images.keys()):
        images_list = cluster_to_images[cluster_id]

        if cluster_id == -1:
            bbx_color = 'black'
            cluster_label = "No Cluster"
            ax.text(0, row_idx + 0.4, cluster_label, ha='center', va='bottom', fontsize=13,
                    color='black', fontweight='bold')
        else:
            bbx_color = color_map[cluster_id]
            cluster_label = f"Cluster: {n_clusters-row_idx}"
            #  add text to the whole cluster
            ax.text(0,row_idx+0.4, cluster_label, ha='center', va='bottom', fontsize=13,
                color='black',fontweight='bold')
        for col_idx, img_path in enumerate(images_list):
            x, y = col_idx, row_idx

            # Load and resize image
            image = Image.open(img_path).convert('L')
            image = image.resize((52,52))

            image_box = OffsetImage(image, zoom=1, cmap='gray')
            image_location = AnnotationBbox(image_box, (x, y), frameon=True,
                                            bboxprops=dict(edgecolor=bbx_color, linewidth=2))
            ax.add_artist(image_location)


    # Formatting
    ax.set_xlim(-1, max_images_per_cluster+1)
    ax.set_ylim(-1,n_clusters)
    ax.axis('off')
    ax.set_title(f"Distance Threshold = {eps_val}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close()
    return fig



def find_duplicates_tuning(df, eps_list=None, save_to_path=None):
    """
    Cluster the data using different rps thresholds in a DBSCAN clustering method, using the embeddings extracted using CLIP model.
    Saves the embeddings and eps values in a .npz file using the name in save_path and name of data.
    params:
    df (DataFrame): dataset, must contain a column name "file_path" that contains the paths to each image.
    eps_list (list): list of distance thresholds for DBSCAN method.
    save_to_path (String): the path where the npz file will be saved including the name of the namd ot the file.
                            name of the file must be: "data name_clustering_embeddings.npz"
    returns: .npz file contain the DBSCAN results.
    """
    # if eps list is None, create a default list
    if eps_list is None:
        eps_list = [0.11, 0.09, 0.07, 0.05, 0.03, 0.01]

    # set up accelerator for multi-GPU support
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = Accelerator(device_placement=True)
    # check if CUDA is being used
    print(f"Device: {accelerator.device}")
    if accelerator.device.type == 'cuda':
        print("CUDA is available")


    # load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", force_download=True)
    clip_model.to(device=accelerator.device)
    clip_model = accelerator.prepare(clip_model)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", force_download=True)
    clip_model.eval()

    # loop through images in the dataset and extract  embeddings
    clip_embeddings = []
    for path in df['file_path']:
        embedding = get_clip_embedding(path, clip_model, clip_processor)
        clip_embeddings.append(embedding)

    #convert list to numpy
    clip_embeddings = np.array(clip_embeddings)

    # save into npz file
    res_data = {
        'clip_embeddings': clip_embeddings,
        'file_path': np.array(df['file_path'])
    }

    # looping through eps values and conduct DBSCAN clustering
    for eps in eps_list:
        labels = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit_predict(clip_embeddings)
        key_name = f"clusters_eps_{str(eps).replace('.','')}"
        res_data[key_name] = labels

    # save the res ad npz file
    np.savez(save_to_path, **res_data)

    # return the file also
    res_npz = np.load(save_to_path, allow_pickle=True)
    return res_npz



# a function to visualize results
def vis_res_eps(res_npz, eps_list, n_clusters_to_sample, seed=1):
    """"
    Gets npz file with columns "clusters_eps_{value}", eps list, number of cluster I want to visualize and seed value.
    for each eps value, it visualizes the images sampled.
    following the same images to track from the first iteration (eps) until the end (last eps).
    params:
    ers_npz (npz file): contain clusters_eps_{value} columns, and file_path.
    eps_list (list): a list of eps values tested and in the npz file.
    n_clusters_to_sample (int): number of clusters to sample randomly.
    seed (int): for reproducibility in randomization.
    returns
    """
    # convert res_npz to dataframe except of embeddings column
    clustering_df = pd.DataFrame({'file_path': res_npz['file_path'].tolist()})
    for eps in eps_list:
        # get the key on npz file
        key = f"clusters_eps_{str(eps).replace('.','')}"
        # add the new col to df
        clustering_df[f"clusters_eps_{eps}"] = res_npz[key]

    # sort eps list from highest to lowest
    eps_list = sorted(eps_list, reverse=True)

    # take the highest eps value and sample randomly clusters
    random.seed(seed)
    highest_eps = eps_list[0]
    first_col_name = f"clusters_eps_{highest_eps}"
    # to not sample cluster -1 and get all unique ids
    valid_clusters = [c for c in clustering_df[first_col_name].unique() if c != -1]
    # sample n clusters randomly
    sampled_clusters = random.sample(valid_clusters, k=n_clusters_to_sample)

    # save tracked images
    df_to_plot = clustering_df[clustering_df[first_col_name].isin(sampled_clusters)].copy(deep=True).reset_index(drop=True)
    # save file paths and cluster ids as a list
    tracked_images = df_to_plot['file_path'].tolist()
    clusters_list = df_to_plot[first_col_name].tolist()
    print(f"Tracked {len(tracked_images)} images from clusters: {set(clusters_list)}")

    # loop through eps values in order to make a plot for each
    plots = []
    for i, eps in enumerate(eps_list):
        cluster_col_name = f"clusters_eps_{eps}"
        if i == 0:
            plot_fig = plot_clusters_grid(file_paths_list=tracked_images, clusters_list=clusters_list, eps_val=eps)
            plots.append(plot_fig)
            continue
        # if this is not the highest distance value in eps list (the first value in the list) then use tracked images

        # get tracked images that are in cluster -1
        no_cluster_images = clustering_df[
            (clustering_df['file_path'].isin(tracked_images)) &
            (clustering_df[cluster_col_name] == -1)
        ].copy(deep=True).reset_index(drop=True)

        # get tracked images that are in actual clusters (not -1)
        clusters_to_plot  = clustering_df[
            (clustering_df['file_path'].isin(tracked_images)) &
            (clustering_df[cluster_col_name] != -1)
        ][cluster_col_name].unique()

        # get all images from those clusters
        clusters_df = clustering_df[clustering_df[cluster_col_name].isin(clusters_to_plot)].copy(deep=True).reset_index(drop=True)

        # combine images from those clusters, with the tracked images from cluster -1
        plot_df = pd.concat([clusters_df, no_cluster_images], ignore_index=True)

        # extract relevant lists
        file_paths_list = plot_df['file_path'].tolist()
        clusters_list = plot_df[cluster_col_name].tolist()
        print(f"Plotting for eps={eps} with {len(file_paths_list)} images and clusters {set(clusters_list)}")

        # Plot
        plot_fig = plot_clusters_grid(file_paths_list=file_paths_list, clusters_list=clusters_list, eps_val=eps)
        plots.append(plot_fig)

    
    return plots


# function for removing the duplicates after choosing the eps value, and save the new dataframe
def remove_duplicates(res_npz, eps_val, df_to_clean, excluded_cluster_ids=None):
    """
    Gets an eps value to use for removing duplicates, and remove duplicates from df to clea using the correct column
    (clustering results) from res_npz.

    params:
    res_npz (npz file): the results from the clustering
    eps_val (int): value of the eps I want to use in order to remove duplicates.
    df_to_clean (DataFrame): the dataframe to clean and remove duplicates from.
    excluded_cluster_ids (list): a list that includes all the clusters I don't want to remove duplicates from,
                                since they are too big to be considered as containing duplicates. no need to write -1 in it.


    returns: cleaned DatFrame.
    """

    # the column I want to use for removing duplicates (the selected eps value)
    npz_key = f"clusters_eps_{str(eps_val).replace('.','')}"
    cluster_col = f"clusters_eps_{eps_val}"

    # add the clustering col to df
    df = df_to_clean.copy(deep=True).reset_index(drop=True)
    df[cluster_col] = res_npz[npz_key]

    # exclude clusters from removal pipeline
    if excluded_cluster_ids is None:
        excluded_cluster_ids = [-1]
    else:
        excluded_cluster_ids = excluded_cluster_ids.copy()
        excluded_cluster_ids.append(-1)

    # filter out images that are not belong to any cluster of duplicates, leaving only the clusters of duplicates
    duplicate_clusters_df =df[~df[cluster_col].isin(excluded_cluster_ids)].copy(deep=True).reset_index(drop=True)
    print(f"Number of samples duplicated: {len(duplicate_clusters_df)}")

    # group by custer id
    grouped = duplicate_clusters_df.groupby(cluster_col)
    print(f"Number of duplicate groups: {len(grouped)}")

    # select the best image to keep, in each cluster
    best_images_df = grouped.apply(lambda x: x.loc[x['votes_percentage'].idxmax()]).copy(deep=True).reset_index(drop=True)

    #now add to the cleaned data, the images we didn't find duplicates to
    non_duplicate_images_df = df[df[cluster_col].isin(excluded_cluster_ids)].copy().reset_index(drop=True)

    # concatenate ti get the final cleaned from duplicates df
    final_df = pd.concat([best_images_df, non_duplicate_images_df]).reset_index(drop=True).reset_index(drop=True)
    print(f"Number of original samples: {len(df)}")
    print(f"Number of samples after removing duplicates: {len(final_df)}")

    return final_df



def plot_single_cluster(df, cluster_id):
    """
    plots all images in a given cluster id.
    df (DataFrame): DataFrame with 'file_path' and 'cluster_id' columns.
    cluster_id (int): ID of the cluster to visualize.
    """


    cluster_images = df[df['cluster_id'] == cluster_id]['file_path'].tolist()
    n_images = len(cluster_images)

    if n_images == 0:
        print(f"No images found for cluster {cluster_id}.")
        return

    # set figure width based on number of images
    width = max(n_images * 1.5, 6)
    height = 2.5

    fig, ax = plt.subplots(figsize=(width, height))

    for idx, img_path in enumerate(cluster_images):
        try:
            image = Image.open(img_path).convert('L').resize((52, 52))
            image_box = OffsetImage(image, zoom=1, cmap='gray')
            image_location = AnnotationBbox(image_box, (idx, 0), frameon=True,
                                            bboxprops=dict(edgecolor='blue', linewidth=1.5))
            ax.add_artist(image_location)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

    ax.set_xlim(-1, n_images)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title(f"Cluster {cluster_id} with {n_images} samples", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.close()
    return fig
