"""
Two-panel dose-response figure for CogMI.

Left:  full QLoRA fine-tuning of each VLM, with I-JEPA (frozen + linear probe) as reference.
Right: each VLM's own frozen vision tower + linear probe, same I-JEPA reference.

Same metric, splits, seeds, and classifier throughout, so the two panels differ only in
how many parameters are exposed to the real-data sample.
"""
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
# Okabe-Ito colorblind-safe palette, matching the other figures in the paper.
C_SMOL, C_QWEN, C_IJEPA = "#0072B2", "#E69F00", "#009E73"

ds = json.load(open("assembled_dataset.json"))
sweep = json.load(open("results/linear_probe_sweep_summary.json"))


def ft_curve(model):
    m, s = [], []
    for r in RATIOS:
        v = [x * 100 for x in ds[model][str(r)].values()]
        m.append(st.mean(v))
        s.append(st.pstdev(v))
    return np.array(m), np.array(s)


def probe_curve(model):
    m = np.array([sweep[model][str(r)]["mr_mean"] for r in RATIOS])
    s = np.array([sweep[model][str(r)]["mr_std"] for r in RATIOS])
    return m, s


x = np.array(RATIOS)
ij_m, ij_s = ft_curve("ijepa")  # I-JEPA is frozen+probe in both panels

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)

panels = [
    (axes[0], "Full QLoRA fine-tuning",
     [("SmolVLM2-2.2B", ft_curve("smolvlm"), C_SMOL),
      ("Qwen2-VL-2B", ft_curve("qwen"), C_QWEN)]),
    (axes[1], "Frozen vision tower + linear probe",
     [("SmolVLM2-2.2B", probe_curve("smolvlm"), C_SMOL),
      ("Qwen2-VL-2B", probe_curve("qwen"), C_QWEN)]),
]

for ax, title, series in panels:
    for label, (m, s), c in series:
        ax.plot(x, m, "o-", color=c, label=label, markersize=4, linewidth=1.6)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
    ax.plot(x, ij_m, "s--", color=C_IJEPA, label="I-JEPA (frozen + probe)",
            markersize=4, linewidth=1.6)
    ax.fill_between(x, ij_m - ij_s, ij_m + ij_s, color=C_IJEPA, alpha=0.15)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("Real data in training set (%)", fontsize=8.5)
    ax.set_xticks(RATIOS)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Macro-recall (%)", fontsize=8.5)
axes[0].set_ylim(28, 90)
axes[1].legend(fontsize=7.2, frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig("overleaf/figures/probe_dose_response.png", dpi=300)
fig.savefig("overleaf/figures/probe_dose_response.pdf")
print("saved overleaf/figures/probe_dose_response.{png,pdf}")
