"""
splitting the classified fer clean remote train/test into 2 csv.
one contains all images classified as faces while the other contains all images classified as non faces
csv contains non-face images saved in classified_fer_cleaning as "non_face_images"
csv contains face images saved in cleaned_fer folder as "cleaned_fer_train", "cleaned_fer_test"

displaying 50 images that were detected as non-face from train only.
"""
import pandas as pd

def split_datasets(csv_path, is_train=True):
    """
    csv_path: the csv I want to split, train or test
    is_train: if it is train data then save also the csv contains non-face images,
              otherwise saving only the cleaned data
    """
    df = pd.read_csv(csv_path)
    # if it's train then save a csv file contains non-face images
    if is_train:
        # filter non-face images into new dataframe and save it
        non_face_images = df[df["face_detected"] == 0].drop(columns=["face_detected"])
        print(non_face_images.columns)
        non_face_images.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/classified_fer_cleaning/non_face_images.csv",
                               index=False)

    # filter face images into new dataframe and save it
    cleaned_file_name = "cleaned_fer_train.csv" if is_train else "cleaned_fer_test.csv"
    cleaned_fer_train = df[df["face_detected"] == 1].drop(columns=["face_detected"])
    cleaned_fer_train.to_csv(f"/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/cleaned_fer/{cleaned_file_name}",
                             index=False)

# splitting the train csv classified
path_train = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/classified_fer_cleaning/classified_fer_clean_remote_train.csv"
split_datasets(path_train, is_train=True)

# splitting the test csv classified
path_test = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/classified_fer_cleaning/classified_fer_clean_remote_test.csv"
split_datasets(path_test, is_train=False)

# check saved files
# how many rows in cleaned_fer_train
df_train = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/cleaned_fer/cleaned_fer_train.csv")
print(f"df_train: {len(df_train)}")

# how many rows in cleans_fer test
df_test = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/cleaned_fer/cleaned_fer_test.csv")
print(f"df_test: {len(df_test)}")
