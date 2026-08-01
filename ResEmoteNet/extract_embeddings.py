import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.ResEmoteNet.resemotenet_rep.approach.ResEmoteNet import ResEmoteNet
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# load data
df5  = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_test.csv")
df = df5.copy(deep=True).reset_index()
print(f"columns: {df.columns.tolist()}")
# import model
# TODO: load the other models I need and store them, and add them under the if else condition
def load_ResEmoteNet(data_model="fer2013"):
    """
    checkpoint (String): must be one of the following thee strings: "fer2013", "affectnet", "rafd"
    returns: pretrained model
    """
    if data_model == "fer2013":
        checkpoint_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ResEmoteNet/resemotenet_rep/checkpoints/fer2013_model.pth"
    else:
        checkpoint_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ResEmoteNet/resemotenet_rep/checkpoints/fer2013_model.pth"
    model = ResEmoteNet()
    model_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.to(device)
    return model


# pr-process the images before sending to the model
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#  load model
resEmotenet_model = load_ResEmoteNet(data_model="fer2013")
resEmotenet_model.eval()



# hook function for 2 layers
hook_outputs = {}
def save_output(name):
    def hook_fn(module, input, output):
        hook_outputs[name] = output.detach()
    return hook_fn
resEmotenet_model.fc1.register_forward_hook(save_output('fc1'))
resEmotenet_model.fc2.register_forward_hook(save_output('fc2'))
resEmotenet_model.fc3.register_forward_hook(save_output('fc3'))



df['prediction'] = None
df['embeddings_f3'] = None
df['embeddings_f2'] = None
df['embeddings_f1'] = None
df['model_output'] = None
# loop through images
for idx, row in df.iterrows():
    image_path = row['file_path']
    pil_image = Image.open(image_path)
    # preprocess the image and add batch dimension
    input_image = transform(pil_image).unsqueeze(0)
    input_image = input_image.to(device)
    print(f"the shape of the input image to the model: {input_image.shape}")
    # forward
    with torch.no_grad():
        model_output = resEmotenet_model(input_image)
    model_output = model_output.squeeze().cpu().numpy()
    # get logits
    df.at[idx, 'model_output'] = model_output
    # get prediction
    prediction = np.argmax(model_output)
    print(f"prediction: {prediction}")
    df.at[idx,'prediction'] = prediction
    # get f1 output
    embeddings_f1 = hook_outputs['fc1']
    embeddings_f1 = embeddings_f1.squeeze(0).cpu().numpy()
    df.at[idx,'embeddings_f1'] = embeddings_f1
    print(f"shape of embeddings from layer fc1: {embeddings_f1.shape}")
    # get f2 output
    embeddings_f2 = hook_outputs['fc2']
    embeddings_f2 = embeddings_f2.squeeze(0).cpu().numpy()
    df.at[idx,'embeddings_f2'] = embeddings_f2
    print(f"shape of embeddings from layer fc2: {embeddings_f2.shape}")
    # get f3 output
    embeddings_f3 = hook_outputs['fc3']
    embeddings_f3 = embeddings_f3.squeeze(0).cpu().numpy()
    df.at[idx,'embeddings_f3'] = embeddings_f3
    print(f"shape of embeddings from layer fc3: {embeddings_f3.shape}")


# PCA
# loop on each embeddings colum in order to plot PCA results for each embeddings
emb_list_cols = ['embeddings_f1', 'embeddings_f2', 'embeddings_f3']
for emb in emb_list_cols:
    X = np.stack(df[emb].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    pc_components = pca.fit_transform(X_scaled)
    print(f"The components and their explained variance:\n{pca.explained_variance_ratio_.tolist()}")
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        pc_components,
        labels=labels,
        dimensions=range(5),
        color=df["true_label"],
        title="<b>Top Components in PCA clustering for {emb}</b>"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    # scatter plot of pc1 and pc2 only
    df_plot = df.copy()
    df_plot["PC1"] = pc_components[:, 0]
    df_plot["PC2"] = pc_components[:, 1]

    fig_2d = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="true_label",
        title=f"<b>PCA Scatter Plot Top 2 Components for {emb}</b>",
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
        }
    )
    fig_2d.show()

    # 3D plot
    total_var = pca.explained_variance_ratio_[:3].sum() * 100
    fig_3d = px.scatter_3d(
        pc_components, x=0, y=1, z=2, color=df['true_label'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        size_max=5,
    )
    fig.update_traces(marker=dict(size=3))
    fig_3d.show()


# create a dictionary to map model's coded labels to how fer2013 is coded
df['coded_true_label'] = df['coded_true_label'].astype(int)
df['prediction'] = df['prediction'].astype(int)
code_to_label = {
    0:3,
    1:5,
    2:4,
    3:0,
    4:1,
    5:2,
    6:6
}
df['prediction'] = df['prediction'].map(code_to_label)
y_true = df['coded_true_label']
y_pred = df['prediction']

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))


