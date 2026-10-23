import json
import os
from collections import defaultdict
import statistics as stats
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["no flood", "small flood", "big flood"]
CLASS_LABELS = ["No Flood", "Small Flood", "Big Flood"]
SEEDS = [42, 123, 7]

# Same palette as figures/dose_response_curve.pdf and failure_modes_barchart.pdf
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
    return [correct[c] / total[c] * 100 for c in CLASSES]

def avg_over_seeds(template):
    per_seed = [per_class_recall(template.format(seed)) for seed in SEEDS]
    return [stats.mean(seed_vals[i] for seed_vals in per_seed) for i in range(3)]

smolvlm_recall = avg_over_seeds("new_results/eval_smolvlm_split_80pct_seed{}.json")
qwen_recall = avg_over_seeds("new_results/eval_qwen_split_80pct_seed{}.json")
ijepa_recall = avg_over_seeds("results/ijepa_split_80pct_seed{}.json")

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
                       label="SmolVLM2-2.2B (fine-tuned, 80% blend)", color=COLOR_SMOLVLM, zorder=3)
bars_qwen = ax.bar(x, qwen_recall, width,
                    label="Qwen2-VL-2B (fine-tuned, 80% blend)", color=COLOR_QWEN, zorder=3)
bars_ijepa = ax.bar(x + width, ijepa_recall, width,
                     label="I-JEPA (80% blend)", color=COLOR_IJEPA, zorder=3)

def label_bars(bars, values):
    for bar, v in zip(bars, values):
        h = bar.get_height()
        if h >= 90:
            y, va, color = h - 4, "top", "white"
        else:
            y, va, color = h + 2, "bottom", "black"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{v:.1f}",
                 ha="center", va=va, fontsize=6.3, color=color, zorder=4)

label_bars(bars_smolvlm, smolvlm_recall)
label_bars(bars_qwen, qwen_recall)
label_bars(bars_ijepa, ijepa_recall)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS)
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 100)
ax.set_title("Per-Class Recall After Adaptation (80% Blend)", fontsize=9.5)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True, framealpha=0.95)

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
legend = ax.get_legend()
fig.savefig("figures/post_adaptation_recall.pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
fig.savefig("figures/post_adaptation_recall.png", dpi=300, bbox_extra_artists=(legend,), bbox_inches="tight")

print("Saved figures/post_adaptation_recall.pdf and .png\n")
print(f"{'Class':<14}{'SmolVLM':>12}{'Qwen':>10}{'I-JEPA':>10}")
for label, sv, qw, ij in zip(CLASS_LABELS, smolvlm_recall, qwen_recall, ijepa_recall):
    print(f"{label:<14}{sv:>11.2f}%{qw:>9.2f}%{ij:>9.2f}%")

print(f"\nMacro-recall check: SmolVLM={sum(smolvlm_recall)/3:.2f}%  Qwen={sum(qwen_recall)/3:.2f}%  I-JEPA={sum(ijepa_recall)/3:.2f}%")
