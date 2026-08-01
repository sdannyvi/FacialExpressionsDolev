import pandas as pd
from pathlib import Path
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *
# paths
images_dir = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/aligned")
labels_txt = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/list_patition_label.txt")

code_to_label= {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad" ,6: "angry", 7: "neutral"}

rows = []
with labels_txt.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        fname, code = line.split()[:2]
        coded = int(code)

        # convert txt filename to aligned filename
        stem = Path(fname).stem
        aligned_name = f"{stem}_aligned.jpg"
        image_path = images_dir / aligned_name

        if not image_path.exists():
            print(f"Missing aligned image file for: {aligned_name} (from {fname})")
            continue

        split = stem.split("_", 1)[0]

        rows.append({
            "id": stem,
            "file_path": str(image_path.resolve()),
            "coded_true_label": coded,
            "true_label": code_to_label.get(coded, "unknown"),
            "split": split
        })

df = pd.DataFrame(rows)
print(df["split"].unique().tolist())


# split to two dfs
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)
print(f"total number of samples: {len(df)}")
print(f"number of samples in train: {len(train_df)}")
print(f"unique values in train: {train_df['split'].unique().tolist()}")
print(f"number of samples in train: {len(test_df)}")
print(f"unique values in train: {test_df['split'].unique().tolist()}")


plot_label_distribution(dfs={"train": train_df, "test": test_df}, label_col="true_label", normalize="percent", plot_name="Class Distribution - RAF-DB")
visualize_sample_images(df=train_df, seed=42, n_images=2, plot_name="Example images - RAF-DB")

# save train and test
train_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/train_set.csv", index=False)
test_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/raf-db/test_set.csv", index=False)