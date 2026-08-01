"""
this file visualize 50 images that where classified as not contain a face, by the RetinaFace model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import numpy as np


def random_samples_plot(df, n=100, title=''):
    """
    sample n samples from dataset, and plot them.
    df (DataFrame): a dataframe must have a column "file_path".
    n (int): number of samples to randomly select. default=100.
    title (String): the title of the plot.
    returns: matplotlib plot.
    """
    # load  non-face images detected
    # df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013icml/cleaned_fer/cleaned_fer_train_07.csv")
    sample_images = df["file_path"].sample(n=n, random_state=42).tolist()
    print(f"number of samples in df: {len(df)} and number of sampled in sampled_images: {len(sample_images)}")
    # create a plot
    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=16)
    # create subplots and favor more columns than rows
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # loop through subplots (each image)
    for i, img_path in enumerate(sample_images):
        image = Image.open(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    return plt

#try: 
#    wandb.login()
#except Exception: 
#    raise RuntimeError("Failed to login to wandb\n. Please check your wandb API key and try again.")
#
# wandb.init(project="fer_plus", name="retinaface estimation- confidence of 0.6",
#             notes="using RetinaFace model to filter out non-face images, visualizing images detected as faces and not faces to estimate false positive and "
#                   "true negative.",
#             tags=["data cleaning", "retinaface"])

# load dataset to plot False Positive and True Positive
cleaned_df = pd.read_csv()
fp_plot = random_samples_plot(df=cleaned_df,n=100,title="Images Classified as Faces by RetinaFace (TP and FP)")
fp_plot.show()

# load dataset non face images for True Negative and False Negative
non_face_df = pd.read_csv()
fp_plot = random_samples_plot(df=non_face_df,n=100,title="Images Classified as Non-Faces by RetinaFace (TN and FN)")

# wandb.finish()


# #
# wandb.init(project="fer_plus", name="retinaface estimation- confidence of 0.7",
#             notes="using RetinaFace model to filter out non-face images, visualizing images detected as faces and not faces to estimate false positive and "
#                   "true negative.",
#             tags=["data cleaning", "retinaface"])

# load dataset to plot False Positive and True Positive
cleaned_df = pd.read_csv()
fp_plot = random_samples_plot(df=cleaned_df,n=100,title="Images Classified as Faces by RetinaFace (TP and FP)")
fp_plot.show()

# load dataset non face images for True Negative and False Negative
non_face_df = pd.read_csv()
fp_plot = random_samples_plot(df=non_face_df,n=100,title="Images Classified as Non-Faces by RetinaFace (TN and FN)")
