import numpy as np
import pandas as pd
import io
import base64
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, no_update, callback

# # load from PCA
# # which embeddings to visualize?
# col_x = 'valence'
# col_name_x = ''
# col_y = 'arousal'
# col_name_y = ''
# title = "Valence and Arousal"
#
#
# # load data for visualization
# data = np.load('emonet_embeddings_pca.npz', allow_pickle=True)
# print("Keys in .npz file:", data.files)
# df_to_plot = pd.DataFrame({
#     'file_path': data['file_path'],
#     'true_label': data['true_label'],
#     'avg_embeddings_pca_1': data['avg_pca'][:, 0],
#     'avg_embeddings_pca_2': data['avg_pca'][:, 1],
#     'logits_embeddings_pca_1': data['logits_pca'][:, 0],
#     'logits_embeddings_pca_2': data['logits_pca'][:, 1],
#     'max_embeddings_pca_1': data['max_pca'][:, 0],
#     'max_embeddings_pca_2': data['max_pca'][:, 1],
#     'valence': data['valence'],
#     'arousal': data['arousal']
# })
# print(df_to_plot['file_path'][0])

# load from umap
# which embeddings to visualize?
col_x = 'max_embeddings_umap_n5_1'
col_name_x = 'X'
col_y = 'max_embeddings_umap_n5_2'
col_name_y = 'Y'
title = "Average Embeddings UMAP neighbors=5"

# load umap results and convert to df
data = np.load("emonet_embeddings_umap.npz", allow_pickle=True)
df_to_plot = pd.DataFrame({key: data[key] for key in data.files})




# visualize


# first scatter plot
fig_emotions = px.scatter(
    df_to_plot,
    x=col_x,
    y=col_y,
    color='true_label',
    labels={'true_label': 'Emotion'},
    title=f"<b>{title}</b>",
    width=700,
    height=700
)

fig_emotions.update_layout(
    title={'text': f"<b>{title}</b>", 'x': 0.5, 'y':0.9, 'font': dict(size=20)},
    legend_title=dict(text='<b>Emotion</b>', font=dict(size=12)),
    legend=dict(font=dict(size=12)),
    xaxis_title=col_name_x,
    yaxis_title=col_name_y
)

fig_emotions.show()




# convert pil images to base64 for visualization
def pil_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/jpeg;base64," + encoded_image


# make numeric labels for true label in order to be able to send a numeric list for coloring
label_to_int = {label: idx for idx, label in enumerate(df_to_plot['true_label'].unique())}
df_to_plot['label_numeric'] = df_to_plot['true_label'].map(label_to_int)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_to_plot[col_x],
    y=df_to_plot[col_y],
    mode="markers",
    marker=dict(
        size=8,
        color=df_to_plot['label_numeric'],
        colorscale="Viridis",
    ),
    customdata=df_to_plot[["file_path", "true_label"]].values,
    name="Emotion"
))


fig.update_layout(template='plotly_white',
                  title={
                      'text': f"<b>{title}</b>", 'xanchor': 'center', 'x': 0.5, 'y': 0.9, 'font': dict(size=20)
                  },
                  xaxis=dict(title=col_name_x, titlefont=dict(size=14)),
                  yaxis=dict(title=col_name_y, titlefont=dict(size=14)),
                  legend_title=dict(text="<b>Emotion</b>", font=dict(size=12)),
                  legend=dict(font=dict(size=12)),
                  width=700,
                  height=700
                  ),

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
        html.H2("Interactive Scatter Plot with Image Tooltip"),
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction='bottom'),
        html.H2("Scatter Plot Colored by Emotion (Legend Visible)"),
        dcc.Graph(id="static-graph", figure=fig_emotions),  # Second plot here
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
    label = hover_data["customdata"][1]
    pil_image = Image.open(image_path)
    im_url = pil_image_to_base64(pil_image)
    children = [
        html.Div([
            html.Img(src=im_url, style={"width": "100px", 'display': 'block', 'margin': '0 auto'}),
            html.P("Label: " + str(label), style={'font-weight': 'bold'}),
        ])
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True)






























# # Create hover text using HTML <img> tag
# data_to_vis['hover_html'] = data_to_vis.apply(lambda row: f"<b>{row['true_label']}</b><br><img src='{row['image_base64']}' width='80'>", axis=1)
#
# fig = px.scatter(
#     data_to_vis,
#     x='arousal',
#     y='valence',
#     color='true_label',
#     hover_name='hover_html'
# )
#
# fig.update_traces(
#     hovertemplate="%{hovertext}<extra></extra>"
# )
#
# fig.update_layout(
#     title='PCA Scatter',
#     xaxis_title='PCA 1',
#     yaxis_title='PCA 2',
#     template='plotly_white'
# )
#
# fig.show()
