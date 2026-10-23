import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("assembled_dataset.json") as f:
    data = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Okabe-Ito colorblind-safe palette
COLOR_SMOLVLM = "#0072B2"  # blue
COLOR_QWEN = "#E69F00"     # orange
COLOR_IJEPA = "#009E73"    # bluish green

def mean_std(seed_dict):
    vals = np.array(list(seed_dict.values())) * 100.0
    return vals.mean(), vals.std(ddof=0), len(vals)

def series_from(method_key, ratios):
    xs, means, stds, n_seeds = [], [], [], []
    for r in ratios:
        seed_dict = data[method_key][str(r)]
        m, s, n = mean_std(seed_dict)
        xs.append(r)
        means.append(m)
        # Only show an error bar/band where we actually have >=3 seeds --
        # never fabricate uncertainty for single-seed points.
        stds.append(s if n >= 3 else 0.0)
        n_seeds.append(n)
    return np.array(xs), np.array(means), np.array(stds), n_seeds

all_ratios = [15, 20, 30, 40, 50, 60, 70, 80, 90]

ijepa_x, ijepa_m, ijepa_s, ijepa_n = series_from("ijepa", all_ratios)
smolvlm_x, smolvlm_m, smolvlm_s, smolvlm_n = series_from("smolvlm", all_ratios)
qwen_x, qwen_m, qwen_s, qwen_n = series_from("qwen", all_ratios)

incomplete = [(name, r, n) for name, xs, ns in
              [("SmolVLM", smolvlm_x, smolvlm_n), ("Qwen", qwen_x, qwen_n), ("I-JEPA", ijepa_x, ijepa_n)]
              for r, n in zip(xs, ns) if n < 3]

fig, ax = plt.subplots(figsize=(7.0, 6.7))

def plot_series(ax, x, m, s, color, label):
    ax.plot(x, m, color=color, marker="o", markersize=8, linewidth=2.4,
             label=label, zorder=3)
    has_err = s > 0
    if has_err.any():
        ax.errorbar(x[has_err], m[has_err], yerr=s[has_err], fmt="none",
                     ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0,
                     zorder=2, alpha=0.9)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.28, linewidth=0, zorder=1)

plot_series(ax, smolvlm_x, smolvlm_m, smolvlm_s, COLOR_SMOLVLM, "SmolVLM2-2.2B (fine-tuned)")
plot_series(ax, qwen_x, qwen_m, qwen_s, COLOR_QWEN, "Qwen2-VL-2B (fine-tuned)")
plot_series(ax, ijepa_x, ijepa_m, ijepa_s, COLOR_IJEPA, "I-JEPA (frozen + LogReg)")

ax.set_xlabel("Percentage of Real Data in Training Set")
ax.set_ylabel("Macro-Recall (%)")
ax.set_title("Macro-Recall vs. Real-Data Blend Ratio\nAcross Model Approaches")
ax.set_xlim(10, 95)
ax.set_ylim(25, 85)
ax.set_xticks(all_ratios)
ax.set_xticklabels([f"{r}%" for r in all_ratios], rotation=40, ha="right", fontsize=12)
ax.grid(True, linewidth=0.6, alpha=0.4)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Placed fully outside the axes (below the x-axis) so it can never overlap
# any data, error bar, or band -- regardless of where the series values fall.
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
                    ncol=1, frameon=True, framealpha=0.95)

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
# bbox_extra_artists + bbox_inches="tight" expands the saved canvas so the
# external legend is fully included rather than clipped.
fig.savefig("figures/dose_response_curve.pdf",
            bbox_extra_artists=(legend,), bbox_inches="tight")
fig.savefig("figures/dose_response_curve.png", dpi=300,
            bbox_extra_artists=(legend,), bbox_inches="tight")

print("Saved figures/dose_response_curve.pdf and .png")
print("\nSmolVLM plotted ratios:", list(smolvlm_x))
print("Qwen plotted ratios:   ", list(qwen_x))
print("I-JEPA plotted ratios: ", list(ijepa_x))

if incomplete:
    print("\nWARNING -- plotted WITHOUT error bars at these single/partial-seed points:")
    for name, r, n in incomplete:
        print(f"  - {name} at {r}%: n={n} seed(s)")

def report(name, x, m, s, n, ratio):
    i = x.tolist().index(ratio)
    print(f"{name} at {ratio}%: mean={m[i]:.2f}%, std={s[i]:.2f}% (n={n[i]})")

print("\n--- Values at 80% (all three methods) ---")
report("SmolVLM", smolvlm_x, smolvlm_m, smolvlm_s, smolvlm_n, 80)
report("Qwen", qwen_x, qwen_m, qwen_s, qwen_n, 80)
report("I-JEPA", ijepa_x, ijepa_m, ijepa_s, ijepa_n, 80)
