import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["no flood", "small flood", "big flood"]
SHORT = ["No\nFlood", "Small\nFlood", "Big\nFlood"]


def load(fn):
    return json.load(open(fn, encoding="utf-8"))


def pooled_confusion_matrix(files):
    cm = defaultdict(lambda: defaultdict(int))
    for fn in files:
        for r in load(fn):
            cm[r["ground_truth"]][r["prediction"]] += 1
    mat = np.zeros((3, 3))
    for i, gt in enumerate(CLASSES):
        for j, pred in enumerate(CLASSES):
            mat[i, j] = cm[gt].get(pred, 0)
    return mat


methods = [
    ("SmolVLM2-2.2B", [f"new_results/eval_smolvlm_split_80pct_seed{s}.json" for s in [42, 123, 7]]),
    ("Qwen2-VL-2B", [f"new_results/eval_qwen_split_80pct_seed{s}.json" for s in [42, 123, 7]]),
    ("I-JEPA + LogReg", [f"results/ijepa_split_80pct_seed{s}.json" for s in [42, 123, 7]]),
]

mats = [pooled_confusion_matrix(files) for _, files in methods]
mats_pct = [m / m.sum(axis=1, keepdims=True) * 100 for m in mats]

plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
fig.subplots_adjust(wspace=0.12)
vmax = 100

for ax, (name, _), mat, matp in zip(axes, methods, mats, mats_pct):
    im = ax.imshow(matp, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    # Row labels only on the leftmost panel: all three panels share the same
    # class order, and repeating them collided with the neighbouring panel.
    if ax is axes[0]:
        ax.set_yticklabels(SHORT, fontsize=9)
        ax.set_ylabel("True", fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_title(name + " (80% blend)", fontsize=10, fontweight="bold", pad=8)
    for i in range(3):
        for j in range(3):
            pct = matp[i, j]
            n = int(mat[i, j])
            color = "white" if pct > 55 else "black"
            ax.text(j, i, f"{pct:.1f}%\n(n={n})", ha="center", va="center",
                     fontsize=8, color=color)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("Row-normalized % (recall)", fontsize=9)

fig.savefig("figures/confusion_matrices.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/confusion_matrices.pdf", bbox_inches="tight")
print("Saved figures/confusion_matrices.png and .pdf")
