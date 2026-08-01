import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# vars
csv = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/classified_fer_cleaning/classified_fer_clean_test.csv"
local_base_path = "C:/Users/sdole/PycharmProjects/Vision-Language-Models/data/FER2013/test/"
remote_base_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/FER2013/test/"
saving_to = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/classified_fer_cleaning/classified_fer_clean_remote_test.csv"


def convert_paths(csv_path, old_base, new_base, saving_to_path):
    """
    csv path: the csv contains local paths
    old_base: the base of the local path
    new_base: remote path  base
    saving_to_path: save into the same path, in a different csv in order to have one file with local paths
                    and the other with remote paths.
    """
    # loading the csv file
    df = pd.read_csv(csv_path)

    # update each path in the csv
    def update_file_path(old_path, old_base, new_base):
        relative_path = old_path[len(old_base):].replace("\\", "/")
        return f"{new_base}{relative_path}"

    # loop through the file and create a list of new paths
    new_paths = []
    for old_path in df["file_path"]:
        new_paths.append(update_file_path(old_path, old_base, new_base))

    # replace the older column with new one
    df["file_path"] = new_paths

    # save into the same path without replacing the ole csv file
    df.to_csv(saving_to_path, index=False)
    image_path = df["file_path"][0]
    image = cv2.imread(image_path)
    # Show the image
    cv2.imwrite("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/LLaVa/res.png", image)
    print(image.shape)
    # Open and show the image
    image = Image.open(image_path)
    # Display using Matplotlib
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

convert_paths(csv, local_base_path, remote_base_path, saving_to)


