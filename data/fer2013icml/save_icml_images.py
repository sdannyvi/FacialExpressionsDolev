"""
save ICML (fer 2013) as images png. and add another column of file_path.
save as new csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import plotly.express as px

# load data
icml_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013icml/icml_face_data.csv")

# print columns
print(f"icml data columns: \n {icml_data.columns.tolist()}")

# print data types
print(f"data types: \n {icml_data.dtypes}")

# strip and lowercase column names
icml_data.columns = icml_data.columns.str.strip()
icml_data.columns = icml_data.columns.str.lower()
print(f"Columns: {icml_data.columns.tolist()}")

# if there is a null value in any columns, print the number of nulls
print("Nulls: if there are no nulls, prints nothing.")
for col in icml_data.columns:
    if icml_data[col].isnull().sum() != 0:
        print(f"{col}: {icml_data[col].isnull().sum()} null values")

# change emotions column to coded true label and add true label column
icml_data.rename(columns={"emotion": "coded_true_label"}, inplace=True)

# adding a true_label textual column
icml_data["true_label"] = None
dict_labels = {'0': "angry", '1': "disgust", '2':"fear", '3':"happy", "4":"sad", '5':"surprise", '6':"neutral"}
icml_data["true_label"] = icml_data["coded_true_label"].astype(str).map(dict_labels)

# check if labeling was done right
for code, label in dict_labels.items():
    unique_values = icml_data[icml_data["coded_true_label"] == int(code)]["true_label"].unique()
    print(f"Coded label {code} -> Expected: {label}, Found: {unique_values}")

# convert dtype of coded true label to int
icml_data["coded_true_label"] = icml_data["coded_true_label"].astype(int)
print(icml_data["coded_true_label"].isnull().sum())

# print the first value in pixels column
print(f"value in pixels column: {icml_data['pixels'][0]}")

# save images as png and add file_path column
# directory to store images
save_dir = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013"

# file paths
file_paths = []
counter = 0

# loop through rows and save as png
for index, row in icml_data.iterrows():
    # get usage folder
    usage_folder = os.path.join(save_dir, row['usage'])
    os.makedirs(usage_folder, exist_ok=True)
    # get emotion class folder
    class_folder = os.path.join(usage_folder, row['true_label'])
    # create subfolder for each class
    os.makedirs(class_folder, exist_ok=True)

    # convert pixels to numpy array
    img_pixels = row["pixels"]
    img_array = np.fromstring(img_pixels, sep=' ', dtype=np.uint8).reshape(48,48)
    if counter < 3:
        print(f"img array dtype: {type(img_array)} and shape: {img_array.shape}")
    # Save image
    img = Image.fromarray(img_array)
    file_name = f"{row['true_label']}_{row['usage']}_{index}.png"
    img_path = os.path.join(class_folder, file_name)
    if counter < 3:
        print(f"image was saved to path: {img_path}, the class is: {row['true_label']}, the usage: {row['usage']}")
        counter += 1
    img.save(img_path, format="PNG")
    # add image path to a list of file paths
    file_paths.append(img_path)

# add file paths column
icml_data['file_path'] = file_paths

# check if images can be opened
def check_image_integrity(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False

corrupt_images = [img_path for img_path in file_paths if not check_image_integrity(img_path)]
print(f"the number of images not opened is: {len(corrupt_images)}")



# compare original and saved images pixel by pixel
def check_pixel_match(original_image, saved_image_path, counter=0):
    # Load the saved image and convert it to a NumPy array
    saved_image = Image.open(saved_image_path).convert("L")
    saved_image = np.array(saved_image, dtype=np.uint8)
    if counter < 3:
        print(f"saved image shape: {saved_image.shape}")
    # Compare arrays pixel by pixel
    return np.array_equal(original_image, saved_image)  # True if identical, False otherwise

# Randomly sample 100 images
sampled_rows = icml_data.sample(n=100, random_state=42)

# Check for pixel-wise matches
mismatches = []
counter = 0
for index, row in sampled_rows.iterrows():
    original_image = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48,48)
    if not check_pixel_match(original_image, row["file_path"], counter=counter):
        mismatches.append(index)
    counter += 1

# Print results
if mismatches:
    print(f"{len(mismatches)} images do NOT match their original pixel values.")
    print(f"Mismatched image indices: {mismatches}")
else:
    print("All sampled images are pixel-perfect matches with the original data.")


# drop pixels column
icml_data.drop(columns=["pixels"], inplace=True)


# save csv file in fer2013 folder
icml_data.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013/icml_data.csv", index=False)
