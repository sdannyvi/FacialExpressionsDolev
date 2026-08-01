"""
clean the train and test data using RetinaFace with confident threshold of 0.6.
plot 100 images from the set of images that where detected without faces at all, and save the cleaned test and train.
cleaned train and test in path data ->  cleaned fer
"""
import pandas as pd
from insightface.app import FaceAnalysis
import cv2
import gc
import torch
import os



# clean the data
def retinaface_data_cleaning(df_to_clean,cleaned_data_path=None, non_face_data_path=None, confidence_thresh=0.6):
    """
    params:
    df (DataFrame): the dataset, must have column name "file_path" to images.
    cleaned_data_path (String): the path in which the cleaned data set will be saved.
    non_face_data_path (String): the path in which the images detected as non faces will be saved.
    confidence_thresh (float): default 0.6. the confidence threshold retina face will use.
    """
    df = df_to_clean.copy(deep=True).reset_index(drop=True)
    # adding another column for cleaning
    df["face_detected"] = 0
    # loading retinaface model
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(64, 64), det_thresh=confidence_thresh)

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
    cleaned_df.to_csv(cleaned_data_path, index=False)
    non_face_df.to_csv(non_face_data_path, index=False)
    print(f"data was saved")

# # send the entire dataset, and clean it, then save a cleaned data and non face data
# data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/after_duplicates.csv")
# # run with confidence 0.6
# path_to_cleaned_06 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_clean_confidence_0.6.csv"
# path_to_non_face_06 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_non_face_confidence_0.6.csv"
# retinaface_data_cleaning(df_to_clean=data, cleaned_data_path=path_to_cleaned_06,
#                          non_face_data_path=path_to_non_face_06, confidence_thresh=0.6)
#
# # run with confidence 0.7
# path_to_cleaned_07 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_clean_confidence_0.7.csv"
# path_to_non_face_07 = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_non_face_confidence_0.7.csv"
# retinaface_data_cleaning(df_to_clean=data, cleaned_data_path=path_to_cleaned_07,
#                          non_face_data_path=path_to_non_face_07, confidence_thresh=0.7)




















# def split_datasets(is_train=True, classified_df=None, save_to_path=None):
#     """
#     csv_path: the csv I want to split, train or test
#     is_train: if it is train data then save also the csv contains non-face images,
#               otherwise saving only the cleaned data
#     save_to_path: path to save cdv results - the filtered images and cleaned df
#     """
#     # if it's train then save a csv file contains non-face images
#     if is_train:
#         # filter non-face images into new dataframe and save it
#         non_face_images = classified_df[classified_df["face_detected"] == 0].drop(columns=["face_detected"]).reset_index(drop=True)
#         print(non_face_images.columns)
#         non_face_images.to_csv(os.path.join(save_to_path, "non_face_images.csv"), index=False)
#
#
#     # filter face images into new dataframe and save it
#     cleaned_file_name = "cleaned_fer_train.csv" if is_train else "cleaned_fer_test.csv"
#     cleaned_fer = classified_df[classified_df["face_detected"] == 1].drop(columns=["face_detected"]).reset_index(drop=True)
#     cleaned_fer.to_csv(os.path.join(save_to_path, cleaned_file_name), index=False)
#
#     print(f"csv saved at: {os.path.join(save_to_path, cleaned_file_name)}")
#
#
# # clean the data
# def data_cleaning(is_train=True, path_to_dataset=None,save_to_path=None):
#     # loading df
#     df = pd.read_csv(path_to_dataset)
#
#     # adding another column for cleaning
#     df["face_detected"] = 0
#     # loading retinaface model
#     app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     app.prepare(ctx_id=0, det_size=(64, 64), det_thresh=0.6)
#
#     # looping through images and detect faces
#     for idx, row in df.iterrows():
#         image_path = row["file_path"]
#         img = cv2.imread(image_path)
#         faces = app.get(img)
#
#         # if faces exist
#         if faces:
#             df.at[idx, "face_detected"] = 1
#
#         # clear gpu memory
#         torch.cuda.empty_cache()
#         del img, faces
#         gc.collect()
#     # if train df, save two datasets - images and non images
#     #  else save only images
#     if is_train:
#         split_datasets(is_train=True, classified_df=df, save_to_path=save_to_path)
#     else:
#         split_datasets(is_train=False, classified_df=df, save_to_path=save_to_path)
#
# # train_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013icml/data_clean/fer_train_no_duplicate.csv"
# save_to = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013icml/cleaned_fer"
# # data_cleaning(is_train=True, path_to_dataset=train_path, save_to_path=save_to)
# test_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013icml/data_clean/fer_test_no_duplicate.csv"
# data_cleaning(is_train=False, path_to_dataset=test_path, save_to_path=save_to)





