import pandas as pd
import argparse
from pathlib import Path
import json
from PIL import Image
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
parser = argparse.ArgumentParser()
parser.add_argument("--affectnet_root", type=Path, required=True)
args = parser.parse_args()
affectnet_root = args.affectnet_root
DATA_DIR = affectnet_root / "human_annotated"

# coded true label to string labels
code_to_label = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "fear", 5: "disgust",
                 6: "angry", 7: "contempt", 8: "None", 9: "uncertain", 10: "non-face"}

dfs = {}
for split in ["train_set", "validation_set"]:
    image_dir  = DATA_DIR / split / "images"
    ann_dir = DATA_DIR / split / "annotations"

    rows = []
    for image_path in image_dir.glob("*.jpg"):
        image_id = image_path.stem
        ann_path = ann_dir / f"{image_id}.json"
        if not ann_path.exists():
            print(f"image id: {image_id}, didn't have a matching json.")
            continue

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        coded =  ann.get("human-label")
        subset = ann.get("subset")

        rows.append({
            "id": image_id,
            "file_path": str(image_path),
            "coded_true_label": coded,
            "true_label": code_to_label.get(coded, "unknown"),
            "subset": subset
        })

    dfs[split] = pd.DataFrame(rows)

df_train = dfs["train_set"]
df_val = dfs["validation_set"]

# validate
# number of samples in each set
print("train set", len(df_train))
print("val set:", len(df_val))

# number of samples in each class
train_counts = df_train["true_label"].value_counts(dropna=False)
val_counts = df_val["true_label"].value_counts(dropna=False)
print("\nTrain - samples per class:")
print(train_counts.to_string())

print("\nVal - samples per class:")
print(val_counts.to_string())

dist = (
    pd.concat(
        [df_train["true_label"].value_counts(dropna=False),
         df_val["true_label"].value_counts(dropna=False)],
        axis=1,
        keys=["train", "val"]
    )
    .fillna(0)
    .astype(int)
)

print("\nTrain vs Val - samples per class:")
print(dist.to_string())


# remove non emotion classes
non_emotion_class = ["None", "non-face", "uncertain"]
train_df_filtered = df_train[~df_train["true_label"].isin(non_emotion_class)].copy()
val_df_filtered   = df_val[~df_val["true_label"].isin(non_emotion_class)].copy()

# train_df_filtered = df_train.drop(
#     df_train[df_train["true_label"].isin(["non-face", "uncertain"]) | df_train["true_label"].isna()].index
# ).copy()

# val_df_filtered = df_val.drop(
#     df_val[df_val["true_label"].isin(["non-face", "uncertain"]) | df_val["true_label"].isna()].index
# ).copy()


plot_label_distribution({"train": train_df_filtered, "val": val_df_filtered},
                        normalize="percent", plot_name="Class Distribution - AffectNet")


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
visualize_sample_images(train_df_filtered, seed=42, n_images=2, plot_name="Example images - AffectNet")


# save csv
train_df_filtered.to_csv(affectnet_root/ "train_set.csv", index=False)

val_df_filtered.to_csv(affectnet_root / "validation_set.csv", index=False)


train_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/affectnet/train_set.csv")
val_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/affectnet/validation_set.csv")


print("Train set:")
print(f"train df unique values: {train_df_filtered['true_label'].unique().tolist()}")
print(f"unique values in true_label {sorted(train_df_filtered['true_label'].unique().tolist())}")
print(f"unique values in coded true label: {sorted(train_df_filtered['coded_true_label'].unique().tolist())}")
print(f"Validation set:")
print(f"unique values in true_label {sorted(val_df_filtered['true_label'].unique().tolist())}")
print(f"unique values in coded true label: {sorted(val_df_filtered['coded_true_label'].unique().tolist())}")

dist = (
    pd.concat(
        [train_df_filtered["true_label"].value_counts(dropna=False),
         val_df_filtered["true_label"].value_counts(dropna=False)],
        axis=1,
        keys=["train", "val"]
    )
    .fillna(0)
    .astype(int)
)

print("\nTrain vs Val - samples per class:")
print(dist.to_string())