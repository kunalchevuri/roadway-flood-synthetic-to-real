import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CELLS = [
    # (row_label, col_label, path)
    ("Synthetic", "No Flood", "data/synthetic_flood_images/133w.png"),
    ("Synthetic", "Small Flood", "data/synthetic_flood_images/126s.png"),
    ("Synthetic", "Big Flood", "data/synthetic_flood_images/128l.png"),
    ("Real (STURM)", "No Flood", "data/sturm_flooddepth_real/upscaled_images/Level0/car2-42-Level0_2.jpg"),
    ("Real (STURM)", "Small Flood", "data/sturm_flooddepth_real/upscaled_images/Level1/car-517-Level1_1.jpg"),
    ("Real (STURM)", "Big Flood", "data/sturm_flooddepth_real/upscaled_images/Level4/car-376-Level4_2.jpg"),
]

TARGET_RATIO = 2.2  # width/height; source images range 1.78-3.04, this is a modest-crop compromise

def center_crop_to_ratio(img, target_ratio):
    w, h = img.size
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    elif cur_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))
    return img

plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(2, 3, figsize=(8.0, 8.0 / 3 / TARGET_RATIO * 2 + 0.6),
                          gridspec_kw={"hspace": 0.12, "wspace": 0.04})

col_titles = ["No Flood", "Small Flood", "Big Flood"]
row_labels = ["Synthetic", "Real"]

for idx, (row_label, col_label, path) in enumerate(CELLS):
    r, c = divmod(idx, 3)
    ax = axes[r][c]
    img = Image.open(path).convert("RGB")
    img = center_crop_to_ratio(img, TARGET_RATIO)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.8)
    if r == 0:
        ax.set_title(col_titles[c], fontsize=12, fontweight="bold", pad=6)

for r, label in enumerate(row_labels):
    bbox = axes[r][0].get_position()
    fig.text(0.02, (bbox.y0 + bbox.y1) / 2, label, fontsize=12, fontweight="bold",
              rotation=90, va="center", ha="center")

fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.02)
fig.savefig("figures/dataset_examples.png", dpi=300)
fig.savefig("figures/dataset_examples.pdf")
print("Saved figures/dataset_examples.png and .pdf")
