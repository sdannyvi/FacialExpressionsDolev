"""
run retina face cleaning, argparse implementation.
"""


import pandas as pd
from insightface.app import FaceAnalysis
import cv2
import gc
import torch
import os
import  argparse

# parser
"""
params:
df (DataFrame): the dataset, must have column name "file_path" to images.
cleaned_data_path (String): the path in which the cleaned data set will be saved.
non_face_data_path (String): the path in which the images detected as non faces will be saved.
confidence_thresh (float): default 0.6. the confidence threshold retina face will use.
"""
parser = argparse.ArgumentParser(description="Run RetinaFace Data Cleaning with different confidence thresholds.")
parser.add_argument('--df_to_clean_path', type=str, required=True,
                    help='path to the dataframe to clean.')
parser.add_argument('--cleaned_data_path', type=str, required=True,
                    help='path in which the cleaned dataframe will be saved.')
parser.add_argument('--non_face_data_path', type=str, required=True,
                    help='path in which the non face dataframe will be saved.')
parser.add_argument('--confidence_thresh', type=float, default=0.6,
                    help='confidence threshold for retinaface model.')
args = parser.parse_args()

# assign to vars
df_path = args.df_to_clean_path


# clean the data
df = pd.read_csv(df_path)

# adding another column for cleaning
df["face_detected"] = 0
# loading retinaface model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(64, 64), det_thresh=args.confidence_thresh)

# looping through images and detect faces
for idx, row in df.iterrows():
    image_path = row["file_path"]
    img = cv2.imread(image_path)
    faces = app.get(img)

    # if faces exist
    if faces:
        df.at[idx, "face_detected"] = 1

    # clear gpu memory
    torch.cuda.empty_cache()
    del img, faces
    gc.collect()
# split the dataset and save clean data and non-face data as csv
cleaned_df = df[df["face_detected"] == 1].drop(columns=["face_detected"]).copy(deep=True).reset_index(drop=True)
non_face_df = df[df["face_detected"] == 0].drop(columns=["face_detected"]).copy(deep=True).reset_index(drop=True)
# save dfs
cleaned_df.to_csv(args.cleaned_data_path, index=False)
non_face_df.to_csv(args.non_face_data_path, index=False)
print(f"data was saved")