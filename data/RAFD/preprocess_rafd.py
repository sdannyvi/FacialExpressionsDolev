from pathlib import Path
import pandas as pd
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.LLaVa.llava_rag.vis_results import   *


folder_path = Path("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/rafd_original_set")


rows = []

for img_path in folder_path.glob("*.jpg"):   # only jpg files
    stem = img_path.stem
    parts = stem.split("_")

    # Safety: handle unexpected filename formats
    if len(parts) != 6:
        # skip or log; here we skip
        continue

    camera_angle, identity, race, gender, true_label, gaze = parts

    rows.append({
        "file_name": img_path.name,
        "file_path": str(img_path.resolve()),
        "camera_angle": camera_angle,
        "identity": identity,
        "race": race,
        "gender": gender,
        "true_label": true_label,
        "gaze": gaze
    })

df = pd.DataFrame(rows)

# change the class names
print("Un-normalized labels:")
print(df["true_label"].unique().tolist())
label_map = {
    "angry": "angry",
    "contemptuous": "contempt",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise"
}
df["true_label"] = (df["true_label"].astype(str).map(label_map))
# check if anything became NaN
print(f"the new labels:")
print(sorted(df['true_label'].unique().tolist()))
print(f"total number of samples: {len(df)}")
print("\n".join([f"{cls}: {cnt}" for cls, cnt in df["true_label"].value_counts(dropna=False).items()]))
plot_label_distribution(dfs={"all": df}, label_col="true_label", normalize="percent", plot_name="Class Distribution - RadBoud")
visualize_sample_images(df=df, seed=1, n_images=2, plot_name="Example images - RadBoud")

df.to_csv("rafd.csv",index=False)