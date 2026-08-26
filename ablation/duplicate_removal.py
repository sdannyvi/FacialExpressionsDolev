import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
from blake3 import blake3
import argparse

from config import resolve_path

# parameters
parser = argparse.ArgumentParser(description="Duplicate removal - ablation study")
parser.add_argument('--kb_path', type=str, required=True,
                    help="Path to KB.")
parser.add_argument("--test_path", type=str, required=True,
                    help="Path to test.")
parser.add_argument("--dup_map_output", type=str, required=True,
                    help="Path to save duplicate map CSV")
parser.add_argument("--kb_clean_output", type=str, required=True,
                    help="Path to save cleaned KB CSV")


args = parser.parse_args()

kb_path = args.kb_path
test_path = args.test_path
dup_map_output = args.dup_map_output
kb_clean_output = args.kb_clean_output

# load CSV data
kb_set = pd.read_csv(kb_path)
test_set = pd.read_csv(test_path)
#kb_set = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_50%.csv")
#test_set = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/private_test_50%.csv")

# merge dataframes for subsequent duplicate removal actions
kb_set = kb_set.assign(split="kb")
test_set = test_set.assign(split="private_test")

df = pd.concat([kb_set, test_set], ignore_index=True)

# load images and return canonical byte buffer
def convert_image_to_bytes(path):
    """
    """
    # open image in grayscale and convert to numpy array
    with Image.open(resolve_path(path)) as img:
        #img = img.convert("L")
        image_arr = np.array(img, dtype=np.uint8)

    ## validate all images are of the same size
    #valid_shapes = [(48, 48), (100, 100), (224, 224)]
    #if image_arr.shape not in valid_shapes:
    #    raise ValueError(f"Unexpected shape {image_arr.shape} for image: {path}")

    # flatten array and convert to bytes
    pixel_bytes = image_arr.tobytes(order="C")  # stable row-major bytes
    return pixel_bytes

# takes a byte buffer, runs BLAKE3, and returns hash string
def blake3_image_to_hash(image_path):
    """
    """
    buf = convert_image_to_bytes(image_path)
    return blake3(buf).hexdigest()


# store hash values
df["hash_blake3"] = df["file_path"].apply(blake3_image_to_hash)
print(df["hash_blake3"].head())

# verify duplicates using pixel bytes (FP)
df["group_id"] = -1
pixel_cache = {} # "image_path": [pixel_bytes]
group_id = 0

# loop through duplicated groups
for h, idx_list in df.groupby("hash_blake3").groups.items():
    idx_list = list(idx_list)
    # if the group contains one image, not a duplicate
    if len(idx_list) == 1:
        continue

    # verify duplicates
    sub_groups = {}  # "pixel_bytes": [image idx]
    for i in idx_list:
        # get image path
        path = df.at[i, "file_path"]
        # store pixel_bytes for each image path key
        if path not in pixel_cache:
            pixel_cache[path] = convert_image_to_bytes(path)
        pixel_bytes = pixel_cache[path]
        if pixel_bytes not in sub_groups:
            sub_groups[pixel_bytes] = []
        sub_groups[pixel_bytes].append(i)

    #
    for _, sub_idx in sub_groups.items():
        # if the list of indices for a certain pixel_bytes is1, not a duplicate
        if len(sub_idx) < 2:
            continue
        # otherwise, give a group id
        for i in sub_idx:
            df.loc[i, "group_id"] = group_id
        group_id += 1

# get all duplicate groups
dup_df = df[df["group_id"] != -1]
# return a series of group id and count of unique split in a group
groups_with_both = (dup_df.groupby("group_id")["split"].nunique().reset_index(name="n_splits"))
# returns a df of group ids contain both kb and test
combined_groups = groups_with_both[groups_with_both["n_splits"] > 1][["group_id"]]
duplicated_test= (dup_df[dup_df["split"] == "private_test"].merge(combined_groups, on="group_id", how="inner"))

percentage = (float((len(duplicated_test)) / (df["split"] == "private_test").sum())) * 100
print(f"% of test images with >=1 KB duplicate: {percentage:.3f}%")


# KB duplicates
dup_df = df[df["group_id"] != -1].copy()

# find group_ids that appear in both splits (kb + private_test)
group_split_counts = (
    dup_df.groupby("group_id")["split"]
    .nunique()
    .reset_index(name="n_splits")
)
combined_groups_clean = group_split_counts[group_split_counts["n_splits"] > 1][["group_id"]]

# total number of kb samples
kb_total = int((df["split"] == "kb").sum())

# count the total number of duplicate samples in KB
kb_dup_df = dup_df[dup_df["split"] == "kb"]
kb_dup_total = len(kb_dup_df)

# count duplicate KB samples with at least one duplicate test sample
kb_dup_leaky = len(kb_dup_df.merge(combined_groups_clean, on="group_id", how="inner"))

# KB duplicate rows that are duplicates only within KB
kb_dup_only = kb_dup_total - kb_dup_leaky

print(f"how many rows the KB has in total: {kb_total}")
print(f"how many KB rows are part of any duplicate group (KB<->KB,KB<->TEST): {kb_dup_total}")

print(f"how many KB rows has at least one matching sample in the TEST (this is a leak): {kb_dup_leaky}")
print(f"how many KB rows are duplicates only against other KB rows: {kb_dup_only}")

print(f"how many KB rows has at least one matching sample in the TEST (in percentage, will be removed): {kb_dup_leaky}/{kb_total} = {100*kb_dup_leaky/kb_total:.3f}%")
print(f"how many KB rows are duplicates only against other KB rows (in percentage, kept): {kb_dup_only}/{kb_total} = {100*kb_dup_only/kb_total:.3f}%")


# create a map csv with: file path, split, group id, is leakage (bool).
# use combined groups to create is leakage column
df["is_leakage"] = df["group_id"].isin(combined_groups_clean["group_id"])
dup_map_df = df[["file_path", "split", "group_id", "is_leakage"]].copy()
#dup_map_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/duplicate_map.csv", index=False)
dup_map_df.to_csv(dup_map_output, index=False)


# validate and print the percentage of kb samples with at least one duplicate in test, and vice versa
test_total = int((dup_map_df["split"] == "private_test").sum())
test_leaky = int(((dup_map_df["split"] == "private_test") & (dup_map_df["is_leakage"])).sum())
percentage = 100 * test_leaky / test_total if test_total else 0.0
print(f"% of test images with >=1 KB duplicate: {percentage:.3f}%")

kb_total = int((dup_map_df["split"] == "kb").sum())

kb_leaky = int(((dup_map_df["split"] == "kb") & (dup_map_df["is_leakage"])).sum())

kb_dup_total = int(((dup_map_df["split"] == "kb") & (dup_map_df["group_id"] != -1)).sum())

kb_dup_only = kb_dup_total - kb_leaky  # duplicates in KB that do NOT overlap test

print(f"how many rows the KB has in total: {kb_total}")
print(f"how many KB rows are part of any duplicate group (KB<->KB,KB<->TEST): {kb_dup_total}")
print(f"how many KB rows has at least one matching sample in the TEST (this is a leak): {kb_leaky}")
print(f"how many KB rows are duplicates only against other KB rows: {kb_dup_only}")

print(f"how many KB rows has at least one matching sample in the TEST (in percentage, will be removed): {kb_leaky}/{kb_total} = {100*kb_leaky/kb_total:.3f}%")
print(f"how many KB rows are duplicates only against other KB rows (in percentage, kept): {kb_dup_only}/{kb_total} = {100*kb_dup_only/kb_total:.3f}%")


# save the clean KB after removing duplicates
# KB paths to keep
kb_keep_paths = dup_map_df[(dup_map_df["split"] == "kb") & (~dup_map_df["is_leakage"])][["file_path"]]

# filter KB by those paths
kb_clean = kb_set.merge(kb_keep_paths, on="file_path", how="inner")

# validate
print(f"number of KB samples before cleanup: {len(kb_set)}")
print(f"number of KB samples after cleanup (duplicate removal): {len(kb_clean)}")
print(f"validate all file_paths are unique, number of all unique file paths: {kb_clean['file_path'].nunique()}")
print(f"validate split column in the KB set contains only KB-split: {kb_clean['split'].unique().tolist()}")
kb_clean = kb_clean.drop(columns=["split"])
#kb_clean.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/ablation/kb_50%_no_leakage.csv", index=False)
kb_clean.to_csv(kb_clean_output, index=False)
