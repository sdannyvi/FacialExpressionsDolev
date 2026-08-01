import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

knowledge_base_set = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# initialize retrival model clip and extract embeddings to train_df
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", force_download=True)
clip_model.to(dtype=torch.float16, device=device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", force_download=True)
clip_model.eval()

# extract embeddings for one image
def get_clip_embedding(image_path, model, processor):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze(0)
    return embedding.cpu()

# extract embeddings for the kb dataset and add as a column
knowledge_base_set['embedding'] = knowledge_base_set['file_path'].apply(lambda x: get_clip_embedding(x, clip_model, clip_processor))

# stack embeddings into a matrix, each row is sample and each column is a feature
X = torch.stack(knowledge_base_set['embedding'].tolist()).numpy()
print(f"number of features: {len(X[0])}")
# get the values of true labels
y = knowledge_base_set['true_label'].values

# standardize the data before LDA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# fit LDA, None will automatically extract C-1 components
lda = LinearDiscriminantAnalysis(n_components=None)
X_lda = lda.fit_transform(X_scaled, y)

# df to plot
df_plot = knowledge_base_set.copy()
num_components = X_lda.shape[1]
for i in range(num_components):
    df_plot[f"LDA_{i+1}"] = X_lda[:, i]

# plot the top two components
fig = px.scatter(
    df_plot,
    x="LDA_1",
    y="LDA_2",
    color="true_label",
    title="<b>Top Two Components (LDA)</b>",
    labels={"LDA_1": "LDA_Component 1", "LDA_2": "LDA_Component 2"}
)
fig.update_layout(
    legend_title=dict(text="<b>Class</b>", font=dict(size=16)),
    legend=dict(font=dict(size=16), itemsizing='constant')
)

fig.show()

# plot the top three components
if num_components >= 3:
    fig_3d = px.scatter_3d(
        df_plot,
        x="LDA_1",
        y="LDA_2",
        z="LDA_3",
        color="true_label",
        title=f"<b>Top Three Components (LDA)</b>",
        labels={'LDA_1': 'LDA 1', 'LDA_2': 'LDA 2', 'LDA_3': 'LDA 3'},
        size_max=5,
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(
        legend_title=dict(text="<b>Class</b>", font=dict(size=16)),
        legend=dict(font=dict(size=16), itemsizing='constant')
    )

    fig_3d.show()

# plot each pair of components
dimensions = [f"LDA_{i+1}" for i in range(num_components)]
fig_matrix = px.scatter_matrix(
    df_plot,
    dimensions=dimensions,
    color="true_label",
    title="<b>LDA Scatter Plot Matrix</b>"
)
fig_matrix.update_traces(diagonal_visible=False)
fig_matrix.update_layout(
    legend_title=dict(text="<b>Class</b>", font=dict(size=16)),
    legend=dict(font=dict(size=16), itemsizing='constant')
)

fig_matrix.show()


print("LDA eigenvalues (class separation strength):", lda.explained_variance_ratio_)