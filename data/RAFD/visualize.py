import pandas as pd
from insightface.app import FaceAnalysis
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from insightface.utils import face_align




CSV_PATH = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/rafd.csv"
SEED = 42
target_angle = "Rafd090"
SAVE_FIG_PATH = f"Visualize_crop_{target_angle}.png"

DET_THRESH = 0.4
DET_SIZE = (640, 640)
CTX_ID = 0
MARGIN = 1.50

df = pd.read_csv(CSV_PATH)
df_angle = df[df["camera_angle"] == target_angle]

random.seed(SEED)
sample_idx = random.choice(list(df_angle.index))
row = df.loc[sample_idx]
image_path = row["file_path"]
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"cv2.imread failed. Check path: {image_path}")


app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)

faces = app.get(img_bgr)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"wspace": 0.0005})

axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image", fontweight="bold", fontsize=12)
axs[0].axis("off")

if not faces:
    # No faces found: show a message on the right
    axs[1].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[1].set_title("No Face Detected", fontweight="bold", fontsize=12)
    axs[1].axis("off")

else:
    # Pick best face by detection score
    best_face = max(faces, key=lambda f: f.det_score)
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
    side = side * (1.0 + MARGIN)
    half = side / 2.0
    left   = int(round(cx - half))
    right  = int(round(cx + half))
    top    = int(round(cy - half))
    bottom = int(round(cy + half))

    # padding if coordinates are outside image boundaries
    H, W = img_bgr.shape[:2]

    # How much the intended square goes out of bounds
    pad_left   = max(0, -left)
    pad_top    = max(0, -top)
    pad_right  = max(0, right - W)
    pad_bottom = max(0, bottom - H)

    needed_padding = (pad_left > 0) or (pad_top > 0) or (pad_right > 0) or (pad_bottom > 0)
    print("needed padding?", needed_padding, "pads:", (pad_left, pad_top, pad_right, pad_bottom))

    # Pad the image (neutral padding: reflect)
    padded_bgr = cv2.copyMakeBorder(
        img_bgr,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )

    # Shift crop coordinates because we added pixels on left/top
    left_p   = left + pad_left
    right_p  = right + pad_left
    top_p    = top + pad_top
    bottom_p = bottom + pad_top



    # visualize square
    left_vis = max(0, left)
    right_vis = min(W, right)
    top_vis = max(0, top)
    bottom_vis = min(H, bottom)

    rect_w = right_vis - left_vis
    rect_h = bottom_vis - top_vis
    rect = patches.Rectangle(
        (left_vis, top_vis),  # (x, y)
        rect_w,
        rect_h,
        linewidth=2,
        edgecolor="lime",
        facecolor="none"
    )
    axs[0].add_patch(rect)

    # Crop from padded image
    cropped_bgr = padded_bgr[top_p:bottom_p, left_p:right_p]


    axs[1].imshow(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Cropped Image", fontweight="bold", fontsize=12)
    axs[1].axis("off")


plt.tight_layout(pad=0.5)
plt.savefig(SAVE_FIG_PATH, dpi=200, bbox_inches="tight")
plt.close()

# print("Sample index:", sample_idx)
print("Image path:", image_path)
print("Saved figure to:", SAVE_FIG_PATH)









