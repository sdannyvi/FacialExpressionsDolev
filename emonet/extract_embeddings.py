import torch
from emonet_repo.emonet.models import EmoNet
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from torch import nn
import io
import base64
from dash import Dash, html, dcc, Input, Output, no_update, callback
import plotly.graph_objects as go
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # functions
# loading model
def load_model(number_classes):
    """
    number_classes: 5 or 8. Which Pretrained Emonet to load.
    """
    if number_classes == 5:
        pretrained_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/emonet_repo/pretrained/emonet_5.pth"
    # trained on 8 classes
    else:
        pretrained_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/emonet/emonet_repo/pretrained/emonet_8.pth"
    # load weights to device
    state_dict = torch.load(pretrained_path, map_location=torch.device(device))
    print(f"state dirct: {state_dict.keys()}")
    # initialize model instance and load to device
    pretrained_model = EmoNet(n_expression=number_classes).to(device)
    # load weights to model architecture
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model






# # start code
# load model
model = load_model(number_classes=8)
model.eval()

# load data
data_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.05.csv"
df = pd.read_csv(data_path)

# process data ans store in a dictionary
preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])


df['processed_image'] = None
for idx, row in df.iterrows():
    image_path = row['file_path']
    image = Image.open(image_path)
    processed_image = preprocess(image)
    processed_image = processed_image.repeat(3, 1, 1)
    df.at[idx, 'processed_image'] = processed_image

print(f"shape of an image after processing: {df['processed_image'][0].shape}")


# set from which part of the model to get embeddings
conv1x1_output = None
def hook(module,input,output):
    global conv1x1_output
    conv1x1_output = output

hook_handle = model.conv1x1_input_emo_2.register_forward_hook(hook)

# extract embeddings
df['logits_embeddings'] = None
df['avg_embeddings'] = None
df['max_embeddings'] = None
df['valence'] = None
df['arousal'] = None
model.eval()
with torch.no_grad():
    for idx in range(len(df)):
        # forward pass
        inputs = df['processed_image'][idx].unsqueeze(0).to(device)
        outputs = model(inputs)
        # save inputs from the last layer
        df.at[idx, 'logits_embeddings'] = outputs['expression'].cpu().numpy()
        df.at[idx, 'valence'] = outputs['valence'].cpu().numpy()
        df.at[idx, 'arousal'] = outputs['arousal'].cpu().numpy()
        # save embeddings from conv1x1 as average pool and max pool
        avg_pool = nn.functional.adaptive_avg_pool2d(conv1x1_output, (1, 1)).view(conv1x1_output.size(0), -1)
        max_pool = nn.functional.adaptive_max_pool2d(conv1x1_output, (1, 1)).view(conv1x1_output.size(0), -1)
        df.at[idx,'avg_embeddings'] = avg_pool.cpu().numpy()
        df.at[idx, 'max_embeddings'] = max_pool.cpu().numpy()

print(f"valence: {df['valence'][0]}")
print(f"arousal: {df['arousal'][0]}")
print(f"shape of average embeddings: {df['avg_embeddings'][0].shape}")
print(f"shape of max embeddings: {df['max_embeddings'][0].shape}")
print(f"shape of logit embeddings: {df['logits_embeddings'][0].shape}")



# PCA
def pca(embedding_col, data_df):
    """
    embedding_col (String): the col name for reducing dimensions (PCA).
    data_df (DataFrame): the dataframe with the embeddings.
    Returns: X_pca. shape [N, 2]
    """
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    # turn data to numpy
    X = np.vstack(data_df[embedding_col].tolist())
    # X = np.stack(data_df[embedding_col].values)
    # normalize data
    X_scaled = scaler.fit_transform(X)
    # pca
    X_pca = pca.fit_transform(X_scaled)

    # create result col names
    pca_1_col_name = f"{embedding_col}_pca_1"
    pca_2_col_name = f"{embedding_col}_pca_2"
    data_df[pca_1_col_name] = X_pca[:, 0]
    data_df[pca_2_col_name] = X_pca[:, 1]


# loop through embeddings columns, conduct PCA and save results
cols_for_pca = ['logits_embeddings', 'avg_embeddings', 'max_embeddings']
df_pca = df.copy(deep=True).reset_index(drop=True)
for col in cols_for_pca:
    pca(col, df_pca)

np.savez(
    'emonet_embeddings_pca.npz',
    file_path=df_pca['file_path'].astype(object).astype(str).values.astype(object),
    true_label=df_pca['true_label'].values,
    logits_pca=df_pca[['logits_embeddings_pca_1', 'logits_embeddings_pca_2']].values,
    avg_pca=df_pca[['avg_embeddings_pca_1', 'avg_embeddings_pca_2']].values,
    max_pca=df_pca[['max_embeddings_pca_1', 'max_embeddings_pca_2']].values,
    valence=df_pca['valence'].values,
    arousal=df_pca['arousal'].values
)




# UMAP
import umap
df_umap = df.copy(deep=True).reset_index(drop=True)
n_neighbors_list = [5, 15, 50]
cols_for_umap = ['logits_embeddings', 'avg_embeddings', 'max_embeddings']

# for each embeddings conduct UMAP with each value of n neighbors
for col in cols_for_umap:
    X = np.vstack(df_umap[col].tolist())
    for n in n_neighbors_list:
        umap_vis = umap.UMAP(n_neighbors=n, metric='cosine')
        x_umap = umap_vis.fit_transform(X)
        df_umap[f"{col}_umap_n{n}_1"] = x_umap[:, 0]
        df_umap[f"{col}_umap_n{n}_2"] = x_umap[:, 1]

# columns to save are columns that umap is in them
columns_to_save = [col for col in df_umap.columns if 'umap' in col]
# and also columns for file path and labels
columns_to_save += ['file_path', 'true_label', 'valence', 'arousal']
# save all
np.savez("emonet_embeddings_umap.npz", **{col: df_umap[col].values for col in columns_to_save})

