"""
Using RetinaFace to crop images around faces
"""
import argparse
import pandas as pd
from insightface.app import FaceAnalysis
import cv2
import gc
import torch
import os
from PIL import Image
import sys

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default="~/Thesis/VLMs/data/RAFD/rafd.csv",
                    help="Path to input csv (must contain 'file_path' col).")
parser.add_argument("--out_dir", default="~/Thesis/VLMs/data/RAFD",
                    help="path to output directory to save cropped and un-cropped images, and output csv.")
parser.add_argument("--margin", type=float, default=1.50,
                    help="Extra scale around facial-landmark square crop (e.g., 1.5)")

args = parser.parse_args()
csv_path = args.csv_path
out_dir = args.out_dir
margin = args.margin

log_path = os.path.join(out_dir, "crop_run.log")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

# load csv
df = pd.read_csv(csv_path)
df_cropped = df.copy(deep=True)
df_cropped["is_padding"] = False


with Image.open(df['file_path'][0]) as img:
    width, height = img.size
    mode = img.mode
    channels = len(img.getbands())

print(f"Size: {width}x{height}, Channels: {channels}, Mode: {mode}")

# loading retinaface model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)

# where to save
output_dir_cropped = os.path.join(out_dir, "rafd_images_cropped")
os.makedirs(output_dir_cropped, exist_ok=True)
not_found_dir = os.path.join(out_dir, "not_found_face_images")
os.makedirs(not_found_dir, exist_ok=True)
out_csv_path = os.path.join(out_dir, "rafd_cropped.csv")

# looping through images and detect faces
for idx, row in df.iterrows():
    image_path = row["file_path"]
    img = cv2.imread(image_path)
    faces = app.get(img)

    # if faces exist then crop
    if faces:
        best_face = max(faces, key=lambda face: face.det_score)

        # get keypoints
        kps = best_face.kps

        # tight rectangle around keypoints
        min_x = float(kps[:, 0].min())
        max_x = float(kps[:, 0].max())
        min_y = float(kps[:, 1].min())
        max_y = float(kps[:, 1].max())

        # height and width of the rectangle
        w = max_x - min_x
        h = max_y - min_y

        # center point if keypoints rectangle
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # get the largest side
        side = max(w, h)

        # build square around keypoints
        # square h and w are size side + margin
        side = side * (1.0 + margin)
        half = side / 2.0
        left = int(round(cx - half))
        right = int(round(cx + half))
        top = int(round(cy - half))
        bottom = int(round(cy + half))

        # padding if coordinates are outside image boundaries
        H, W = img.shape[:2]

        # How much the intended square goes out of bounds
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - W)
        pad_bottom = max(0, bottom - H)

        needed_padding = (pad_left > 0) or (pad_top > 0) or (pad_right > 0) or (pad_bottom > 0)
        if needed_padding:
            df_cropped.at[idx, "is_padding"] = True

        # Pad the image (neutral padding: reflect)
        padded_bgr = cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REPLICATE
        )

        # Shift crop coordinates because we added pixels on left/top
        left_p = left + pad_left
        right_p = right + pad_left
        top_p = top + pad_top
        bottom_p = bottom + pad_top

        # Crop from padded image
        cropped_img = padded_bgr[top_p:bottom_p, left_p:right_p]

        # resize image
        cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # take the name of the file
        base_name = os.path.basename(image_path)
        file_name, file_type = os.path.splitext(base_name)
        new_filename = f"{file_name}_cropped.jpg"
        save_path = os.path.join(output_dir_cropped,new_filename)
        # save cropped image
        cv2.imwrite(save_path, cropped_img)
        df_cropped.at[idx, 'file_path'] = save_path

    # else, do nothing, and print file name
    else:
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        not_found_path = os.path.join(not_found_dir, f"{file_name}.jpg")
        cv2.imwrite(not_found_path, img)
        df_cropped.at[idx, 'file_path'] = not_found_path

    # clear gpu memory
    torch.cuda.empty_cache()
    del img, faces
    gc.collect()

# save the new df
df_cropped.to_csv(out_csv_path, index=False)
print(f"Saved output CSV to: {out_csv_path}")
print(f"Cropped images dir: {output_dir_cropped}")
print(f"Not-found images dir: {not_found_dir}")


# validate cropped images
bad_size = []

for p in df_cropped["file_path"]:
    try:
        with Image.open(p) as im:
            width, height = im.size

            if width != 224 or height != 224:
                bad_size.append((p, width, height))

    except Exception as e:
        print(f"Could not open image: {p} -> {e}")

# print results
if bad_size:
    print("Images not in size 224x224:")
    for p, w, h in bad_size:
        print(f"{p} -> {w}x{h}")
else:
    print("All images are 224x224.")

print(f"number of samples in df_cropped: {len(df_cropped)}")
print(f"number of samples in original df: {len(df)}")
print(f"are all file paths unique? {df_cropped['file_path'].nunique() == len(df_cropped)}")


sys.stdout.close()


