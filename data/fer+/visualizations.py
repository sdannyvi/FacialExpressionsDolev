import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, no_update, callback
import random
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import wandb
#
#
# try: 
#    wandb.login()
# except Exception: 
#    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

#
# wandb.init(
#     project="fer_plus",
#     entity= "FER-VLM",
#     name="tuning_duplicate_removal",
#     tags=['eps', 'k-distance line plot'],
#     notes="tuning duplicate removal process",
# )
#
# load data
data = np.load("embeddings_clustering_data.npz", allow_pickle=True)
clip_embeddings = data['clip_embeddings']
clustering_df = pd.DataFrame({'file_path': data['file_paths'].tolist(),
                                     'clusters_eps_0.11': data['clusters_eps_11'],
                                     'clusters_eps_0.09': data['clusters_eps_09'],
                                     'clusters_eps_0.07': data['clusters_eps_07'],
                                     'clusters_eps_0.05': data['clusters_eps_05'],
                                     'clusters_eps_0.03': data['clusters_eps_03'],
                                     'clusters_eps_0.01': data['clusters_eps_01']})

# sample 200 clusters randomly
random.seed(1)
cluster_col = 'clusters_eps_0.05'
is_random = False
# count the number of samples in the largest clusters and print it
counts = clustering_df[cluster_col].value_counts()
filtered = counts[counts > 5]
sorted_filtered = filtered.sort_values(ascending=False)
print(f"clusters contains more than 5 samples: \n {sorted_filtered}")
if is_random:
    # sample clusters to plot
    unique_clusters = clustering_df[cluster_col].unique().tolist()
    valid_clusters = [c for c in unique_clusters if c != -1 and c != 1]
    print(f"Number of clusters: {len(valid_clusters)}")
    sampled_clusters = random.sample(valid_clusters, k=150)
else:
    counts = clustering_df[cluster_col].value_counts()
    counts_sorted = counts.sort_values(ascending=False)
    unique_clusters = counts_sorted.index.tolist()
    valid_clusters = [c for c in unique_clusters if c != -1 and c != 1]
    sampled_clusters = valid_clusters[:25]
    print(f"Number of clusters: {len(sampled_clusters)}")


# get the images and clusters list for these clusters
# save tracked images in a list
df_to_plot = clustering_df[clustering_df[cluster_col].isin(sampled_clusters)].copy(deep=True).reset_index(drop=True)

# get sampled embeddings
sampled_indices = clustering_df[clustering_df[cluster_col].isin(sampled_clusters)].index.values
clip_emb_sampled = clip_embeddings[sampled_indices]

# cluster all the embeddings
emb_2d = TSNE(n_components=2,perplexity=10.0).fit_transform(clip_emb_sampled)

# add components embeddings to dataframe to plot
df_to_plot['x_component'] = emb_2d[:,0].tolist()
df_to_plot['y_component'] = emb_2d[:,1].tolist()

# convert PIL image to base64
def pil_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/jpeg;base64," + encoded_image

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_to_plot["x_component"],
    y=df_to_plot["y_component"],
    mode="markers",
    marker=dict(
        size=8,
        color=df_to_plot[cluster_col],
        colorscale="Viridis",  # Optional color scale
    ),
    customdata=df_to_plot[["file_path", cluster_col]].values,  # For hover callback
))


eps = cluster_col
title = f"Duplicates - eps Threshold {eps}"
fig.update_layout(template='plotly_white')
fig.update_layout(
    title={'text': title, 'xanchor': 'center', 'x': 0.5, 'y': 0.90},
    xaxis_title="x",
    yaxis_title="y",
    width=700,
    height=700,
)

# Disable default hover information
fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)

# Initialize Dash app
app = Dash(__name__)

# Layout of the Dash app
app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction='bottom'),
    ],
)

# Callback function to display hover information
@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    image_path = hover_data["customdata"][0]
    cluster = hover_data["customdata"][1]
    pil_image = Image.open(image_path)
    im_url = pil_image_to_base64(pil_image)
    children = [
        html.Div([
            html.Img(src=im_url, style={"width": "100px", 'display': 'block', 'margin': '0 auto'}),
            html.P("Cluster: " + str(cluster), style={'font-weight': 'bold'}),
        ])
    ]

    return True, bbox, children



# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
    print("Dash server started.")
    app.run(jupyter_mode="external")



#
#
# def plot_clusters_grid(file_paths_list, clusters_list, eps_val):
#     """
#     file_paths_list: List of image paths
#     clusters_list: List of cluster IDs (same length as file_paths_list)
#     """
#
#     # Group images by cluster
#     cluster_to_images = {}
#     for img_path, cluster_id in zip(file_paths_list, clusters_list):
#         if cluster_id not in cluster_to_images:
#             cluster_to_images[cluster_id] = []
#         cluster_to_images[cluster_id].append(img_path)
#
#     # create a color map
#     colors = sns.color_palette("tab10", len(cluster_to_images.keys()))
#     color_map = {}
#     for i, cluster_id in enumerate(cluster_to_images.keys()):
#         color_map[cluster_id] = colors[i]
#
#
#     # Calculate grid size
#     n_clusters = len(cluster_to_images.keys())
#     max_images_per_cluster = max(len(images) for images in cluster_to_images.values())
#     width = max(max_images_per_cluster * 1.6, 6)  # minimum width = 6
#     height = max(n_clusters * 1.6, 10)  # maximum height = 10
#
#     fig, ax = plt.subplots(figsize=(width, height))
#
#     # fig, ax = plt.subplots(figsize=(max_images_per_cluster * 1.6, n_clusters * 1.6))
#
#     # Loop and plot images
#     for row_idx, cluster_id in enumerate(cluster_to_images.keys()):
#         images_list = cluster_to_images[cluster_id]
#
#         if cluster_id == -1:
#             bbx_color = 'black'
#             cluster_label = "No Cluster"
#             ax.text(0, row_idx + 0.4, cluster_label, ha='center', va='bottom', fontsize=13,
#                     color='black', fontweight='bold')
#         else:
#             bbx_color = color_map[cluster_id]
#             cluster_label = f"Cluster: {n_clusters-row_idx}"
#             #  add text to the whole cluster
#             ax.text(0,row_idx+0.4, cluster_label, ha='center', va='bottom', fontsize=13,
#                 color='black',fontweight='bold')
#         for col_idx, img_path in enumerate(images_list):
#             x, y = col_idx, row_idx
#
#             # Load and resize image
#             image = Image.open(img_path).convert('L')
#             image = image.resize((52,52))
#
#             image_box = OffsetImage(image, zoom=1, cmap='gray')
#             image_location = AnnotationBbox(image_box, (x, y), frameon=True,
#                                             bboxprops=dict(edgecolor=bbx_color, linewidth=2))
#             ax.add_artist(image_location)
#
#
#     # Formatting
#     ax.set_xlim(-1, max_images_per_cluster+1)
#     ax.set_ylim(-1,n_clusters)
#     ax.axis('off')
#     ax.set_title(f"Distance Threshold = {eps_val}", fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     # wandb.log({f"DBSCAN Results for eps_threshold={eps_val}": wandb.Image(plt)})
#     plt.show()
#     plt.close()
#     return plt
#
# import random
#
#
#
#
# cluster_col = 'clusters_eps_0.11'
#
#
# # get the size of the clusters
# cluster_sizes = clustering_df.groupby(cluster_col).size()
#
# # get clusters that contain less than 8 images or cluster id -1
# valid_clusters = cluster_sizes[(cluster_sizes <= 10)].index.tolist()
# filtered_df = clustering_df[clustering_df[cluster_col].isin(valid_clusters)].copy(deep=True).reset_index(drop=True)
# # randomly select 3
# random.seed(20)
# tracked_images = random.sample(filtered_df['file_path'].tolist(), k=10)
#
#
# #
# # # select 3 random clusters to plot from eps 0.05
# # cluster_col= 'clusters_eps_0.11'
# # random.seed(100)
# eps_list = [0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
# # # sample images to track
# # tracked_images = random.sample(clustering_df['file_path'].tolist(), k=5)
#
# for i in range(len(eps_list)):
#     # cluster col name
#     cluster_col_name = f"clusters_eps_{eps_list[i]}"
#
#     # find the clusters of the samples to track
#     # 1. get tracked images with cluster -1
#     no_cluster_images = clustering_df[
#         (clustering_df['file_path'].isin(tracked_images)) &
#         (clustering_df[cluster_col_name] == -1)
#     ].copy(deep=True).reset_index(drop=True)
#
#     # get the other tracked images that are in a real cluster
#     clusters_to_plot  = clustering_df[
#         (clustering_df['file_path'].isin(tracked_images)) &
#         (clustering_df[cluster_col_name] != -1)
#     ][cluster_col_name].unique()
#
#     # get all images from those clusters
#     clusters_df = clustering_df[clustering_df[cluster_col_name].isin(clusters_to_plot)].copy(deep=True).reset_index(drop=True)
#
#     # combine the images from cluster df and the tracked images that without a cluster (-1)
#     plot_df = pd.concat([clusters_df, no_cluster_images], ignore_index=True)
#
#     # extract relevant lists
#     file_paths_list = plot_df['file_path'].tolist()
#     clusters_list = plot_df[cluster_col_name].tolist()
#     print(f"Plotting for eps={eps_list[i]} with {len(file_paths_list)} images and clusters {set(clusters_list)}")
#
#     # Plot
#     plot_clusters_grid(file_paths_list=file_paths_list, clusters_list=clusters_list, eps_val=eps_list[i])
#
#
#
# # plot k distance
# def plot_k_distance_graph(X, k):
#     neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
#     neigh.fit(X)
#     distances, _ = neigh.kneighbors(X)
#     # print the first neighbor distance
#     print(f"The first neighbor distance: {distances[:,0]}")
#     distances = np.sort(distances[:, k-1])
#     plt.figure(figsize=(10, 6))
#     plt.plot(distances)
#     plt.xlabel('Data Points')
#     plt.ylabel(f'{k}-th nearest neighbor distance')
#     plt.title('K-distance Graph')
#     plt.yticks(np.arange(0, 1, 0.04))
#     plt.show()
#
#     # plotly
#     df = pd.DataFrame({'Distance': distances})
#     fig = px.line(df, y='Distance', title='K-Distance')
#     fig.add_annotation(text="The plot shows distances  to the 2nd nearest neighbor for all samples."
#                             "The place where distance starts increasing sharply is where epsilon should be.", xref='paper', yref='paper', x=0, y=1.15,
#                        showarrow=False, align='left', font=dict(size=12))
#
#     wandb.log({"Distance from 2nd Closest Neighbor": wandb.Plotly(fig)})
#
#     # fig.show()
# # Plot k-distance graph
# plot_k_distance_graph(X=clip_embeddings, k=2)
# wandb.finish()

# try: 
#    wandb.login()
# except Exception: 
#    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")

#
# wandb.init(
#     project="fer_plus",
#     entity= "FER-VLM",
#     name="distribution after retinaface",
#     tags=['retinaface', 'dictribution'],
#     notes="non face images removal using RetinaFace.",
# )

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

# df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_clean_confidence_0.6.csv")
#
# fig = hist_visualization(df=df, plot_title="Distribution After Removing Non Face Images (RetinaFace)")
# wandb.log({"Distribution after RetinaFace": fig})
# wandb.finish()