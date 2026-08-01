import gc
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pandas as pd
import torch

def tuning_confidence_thresh(df, save_to_path):
    """
    tuning confidence threshold for RetinaFace model, to remove non face images from the data.
    tuning is done based on number of samples dropped.
    params:
    df (DataFrame): dataframe to clean, contains columns: file_path.
    save_to_path (string): the path in which the result df will be saved.
    returns: dataframe, with cols: confidence_threshold, percentage_data_to_keep
    """
    # loading the train set
    # df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013csv/fer_train_remote.csv")

    # create a dictionary of confidence and proportion of data kept
    confidence_res = {}
    # list of confidence to loop on
    confidence_ls = [round(i * 0.1, 1) for i in range(1, 11)]
    print(f'confidence list to loop on: {confidence_ls}')


    for confidence_level in confidence_ls:
        print(f"run retinaface with confidence: {confidence_level}")

        # setting a counter to count how many images were faces detected
        face_images_count = 0
        total_images = len(df)

        # loading the model with the conf level
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(64, 64), det_thresh=confidence_level)

        for idx, row in df.iterrows():
            image_path = row["file_path"]
            img = cv2.imread(image_path)

            faces = app.get(img)

            # if faces exist, count it
            if faces:
                face_images_count += 1
            # clear gpu memory
            torch.cuda.empty_cache()
            del img, faces
            gc.collect()
        # after looping through the entire dataset, calculate percentage to keep
        percentage_faces = (face_images_count / total_images) * 100

        # storing results in dictionary
        confidence_res[confidence_level] = percentage_faces

        # save the results for now
        results_df = pd.DataFrame(list(confidence_res.items()), columns=["Confidence_level", "Percentage_data_to_keep"])
        results_df.to_csv(save_to_path,index=False)
        # results_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data_cleaning/test_res_tuning.csv", index=False)
    # at the end of the process, convert to df and save it
    results_df = pd.DataFrame(list(confidence_res.items()), columns=["Confidence_level", "Percentage_data_to_keep"])
    results_df.to_csv(save_to_path, index=False)
    return results_df
    # results_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data_cleaning/test_res_tuning.csv", index=False)


# ferplus_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/after_duplicates.csv")
# save_to = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/retinaface_tuning.csv"
#
# tuning_confidence_thresh(df=ferplus_df, save_to_path=save_to)