import json
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["no flood", "small flood", "big flood"]
CLASS_LABELS = ["No Flood", "Small Flood", "Big Flood"]

# Same palette as figures/dose_response_curve.pdf (plot_figure.py)
COLOR_SMOLVLM = "#0072B2"  # blue
COLOR_QWEN = "#E69F00"     # orange
COLOR_IJEPA = "#009E73"    # bluish green

def per_class_recall(fname):
    with open(fname, "r", encoding="utf-8") as f:
        records = json.load(f)
    total = defaultdict(int)
    correct = defaultdict(int)
    for r in records:
        gt = r["ground_truth"]
        total[gt] += 1
        if r["correct"]:
            correct[gt] += 1
    recalls = [correct[c] / total[c] * 100 for c in CLASSES]
    return recalls, total

smolvlm_recall, smolvlm_totals = per_class_recall("results/exp1_smolvlm_zeroshot_real.json")
qwen_recall, qwen_totals = per_class_recall("results/exp2_qwen_zeroshot_real.json")
ijepa_recall, ijepa_totals = per_class_recall("results/ijepa_synthetic_only.json")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
})

fig, ax = plt.subplots(figsize=(3.5, 3.4))

x = np.arange(len(CLASSES))
width = 0.26

ax.set_axisbelow(True)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4, color="lightgray")

bars_smolvlm = ax.bar(x - width, smolvlm_recall, width,
                       label="SmolVLM2-2.2B (zero-shot)", color=COLOR_SMOLVLM, zorder=3)
bars_qwen = ax.bar(x, qwen_recall, width,
                    label="Qwen2-VL-2B (zero-shot)", color=COLOR_QWEN, zorder=3)
bars_ijepa = ax.bar(x + width, ijepa_recall, width,
                     label="I-JEPA (synthetic-only trained)", color=COLOR_IJEPA, zorder=3)

def label_bars(bars, values):
    for bar, v in zip(bars, values):
        h = bar.get_height()
        # Tall bars (near 100%) get their label inside, near the top, in white;
        # short bars (near 0%) get their label just above the bar in black,
        # since a label above a ~0-height bar would otherwise collide with the axis.
        if h >= 90:
            y, va, color = h - 4, "top", "white"
        else:
            y, va, color = h + 2, "bottom", "black"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{v:.1f}",
                 ha="center", va=va, fontsize=6.3, color=color, zorder=4, rotation=0)

label_bars(bars_smolvlm, smolvlm_recall)
label_bars(bars_qwen, qwen_recall)
label_bars(bars_ijepa, ijepa_recall)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS)
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 100)
ax.set_title("Per-Class Recall: Pre-Adaptation Failure Modes", fontsize=9.5)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True, framealpha=0.95)

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
legend = ax.get_legend()
fig.savefig("figures/failure_modes_barchart.pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
fig.savefig("figures/failure_modes_barchart.png", dpi=300, bbox_extra_artists=(legend,), bbox_inches="tight")

print("Saved figures/failure_modes_barchart.pdf and .png\n")
print(f"{'Class':<14}{'n':>6}{'SmolVLM':>12}{'Qwen':>10}{'I-JEPA':>10}")
for c, label, sv, qw, ij in zip(CLASSES, CLASS_LABELS, smolvlm_recall, qwen_recall, ijepa_recall):
    print(f"{label:<14}{smolvlm_totals[c]:>6}{sv:>11.2f}%{qw:>9.2f}%{ij:>9.2f}%")
