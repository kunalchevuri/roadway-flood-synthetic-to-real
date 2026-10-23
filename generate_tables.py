import json
from collections import defaultdict

CLASSES = ["no flood", "small flood", "big flood"]

def load(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_from_raw(fname):
    d = load(fname)
    total = defaultdict(int)
    correct = defaultdict(int)
    n_correct = 0
    for r in d:
        gt = r["ground_truth"]
        total[gt] += 1
        if r["correct"]:
            correct[gt] += 1
            n_correct += 1
    recalls = [correct[c] / total[c] * 100 for c in CLASSES]
    return n_correct / len(d) * 100, sum(recalls) / 3

def compute_latency(fname):
    d = load(fname)
    return sum(r["latency_s"] for r in d) / len(d)

# ===========================================================================
# TABLE (a): Summary of baseline results
# ===========================================================================
rows_a = []

# SmolVLM zero-shot / 3-shot / Qwen zero-shot: computed directly from raw
# per-image files (results/exp1_*, results/exp2_*), independently verified
# against final_summary_table.json in the same run -- exact match (<0.05pp).
acc, mr = compute_from_raw("results/exp1_smolvlm_zeroshot_real.json")
rows_a.append(("SmolVLM2-2.2B, zero-shot", acc, mr, "computed from raw predictions (results/exp1_smolvlm_zeroshot_real.json)"))

acc, mr = compute_from_raw("results/exp1_smolvlm_3shot_real.json")
rows_a.append(("SmolVLM2-2.2B, 3-shot", acc, mr, "computed from raw predictions (results/exp1_smolvlm_3shot_real.json)"))

acc, mr = compute_from_raw("results/exp2_qwen_zeroshot_real.json")
rows_a.append(("Qwen2-VL-2B, zero-shot", acc, mr, "computed from raw predictions (results/exp2_qwen_zeroshot_real.json)"))

# SmolVLM synthetic-only fine-tune: NO raw per-image file exists in results/
# for this configuration. Only source is the pre-aggregated, hand-annotated
# row in final_summary_table.json, whose own footnote says this macro-recall
# figure "was computed by hand from the per-class breakdown, not by the
# notebook" -- i.e. it is NOT independently re-verifiable from source data
# the way the rows above are. Flagged explicitly rather than silently
# presented with the same confidence as the verified rows.
final_table = load("results/final_summary_table.json")
smolvlm_synth_row = next(r for r in final_table if r[1] == "fine-tuned (synthetic-only)")
synth_acc = float(smolvlm_synth_row[3].rstrip("%"))
synth_mr = float(smolvlm_synth_row[4].rstrip("%*"))
rows_a.append(("SmolVLM2-2.2B FT (synthetic-only)", synth_acc, synth_mr,
                "UNVERIFIED: no raw per-image file exists; sourced from final_summary_table.json, "
                "whose own footnote says this macro-recall was hand-computed, not notebook-computed"))

# Best ensemble result: checked EVERY ensemble-family summary file in results/
# (ensemble_summary.json, ensemble_three_way_summary.json, learned_combiner_summary.json,
# ensemble_qwen_finetuned_summary.json) and took the max macro-recall found, rather than
# assuming which one the draft calls "best". This is qwen-zeroshot + smolvlm-finetuned(30%
# split) combined via ensemble_qwen_finetuned_summary.json's rule -- NOT the Rule-1 result
# that appears in the draft's current Table I.
ensemble_candidates = [
    ("ensemble_summary.json:rule1", load("results/ensemble_summary.json")["rule1"]),
    ("ensemble_summary.json:rule2", load("results/ensemble_summary.json")["rule2"]),
    ("ensemble_three_way_summary.json", load("results/ensemble_three_way_summary.json")),
    ("learned_combiner_summary.json:learned_combiner", load("results/learned_combiner_summary.json")["learned_combiner"]),
    ("learned_combiner_summary.json:rule1_same_test", load("results/learned_combiner_summary.json")["rule1_same_test"]),
    ("ensemble_qwen_finetuned_summary.json:ensemble", load("results/ensemble_qwen_finetuned_summary.json")["ensemble"]),
]
best_source, best = max(ensemble_candidates, key=lambda kv: kv[1]["macro_recall"])
rows_a.append(("Best ensemble result", best["accuracy"] * 100, best["macro_recall"] * 100,
                f"max macro-recall across all ensemble-family summary files; winner = {best_source}"))

cv = load("results/classical_cv_baseline_summary.json")
rows_a.append(("Classical CV baseline", cv["accuracy"] * 100, cv["macro_recall"] * 100,
                "results/classical_cv_baseline_summary.json"))

print("=" * 100)
print("TABLE (a) SOURCE PROVENANCE")
print("=" * 100)
for name, acc, mr, source in rows_a:
    print(f"{name:<38}acc={acc:6.2f}%  macro={mr:6.2f}%   <- {source}")

latex_a = []
latex_a.append(r"\begin{table}[t]")
latex_a.append(r"\caption{Summary of Key Results on Real-World Data (STURM-FloodDepth)}")
latex_a.append(r"\label{tab:summary}")
latex_a.append(r"\centering")
latex_a.append(r"\begin{tabular}{lcc}")
latex_a.append(r"\toprule")
latex_a.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Macro-Recall} \\")
latex_a.append(r"\midrule")
for name, acc, mr, _ in rows_a:
    latex_a.append(f"{name} & {acc:.1f}\\% & {mr:.1f}\\% \\\\")
latex_a.append(r"\bottomrule")
latex_a.append(r"\end{tabular}")
latex_a.append(r"\end{table}")
latex_a_str = "\n".join(latex_a)

print("\n" + "=" * 100)
print("TABLE (a) LATEX OUTPUT")
print("=" * 100)
print(latex_a_str)

# ===========================================================================
# TABLE (b): Full dose-response table (9 ratios x 3 methods, mean +- std)
# ===========================================================================
data = load("assembled_dataset.json")
RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
METHODS = ["smolvlm", "qwen", "ijepa"]

import statistics as stats

def cell(method, ratio):
    d = data[method][str(ratio)]
    vals = [v * 100 for v in d.values()]
    n = len(vals)
    mean = stats.mean(vals)
    if n >= 3:
        std = stats.pstdev(vals)
        return f"{mean:.1f} $\\pm$ {std:.1f}", n
    else:
        return f"{mean:.1f}*", n

print("\n" + "=" * 100)
print("TABLE (b) CELL-BY-CELL (method, ratio, n_seeds, formatted value)")
print("=" * 100)
table_b_cells = {}
any_incomplete = False
for method in METHODS:
    for r in RATIOS:
        text, n = cell(method, r)
        table_b_cells[(method, r)] = text
        if n < 3:
            any_incomplete = True
        print(f"{method:<10}{r:>4}%  n={n}  {text}")

latex_b = []
latex_b.append(r"\begin{table}[t]")
latex_b.append(r"\caption{Blended Fine-Tuning Dose-Response Comparison (Mean $\pm$ Std Macro-Recall, 3 Seeds per Ratio)}")
latex_b.append(r"\label{tab:doseresponse}")
latex_b.append(r"\centering")
latex_b.append(r"\begin{tabular}{lccc}")
latex_b.append(r"\toprule")
latex_b.append(r"\textbf{Ratio} & \textbf{SmolVLM} & \textbf{Qwen} & \textbf{I-JEPA} \\")
latex_b.append(r"\midrule")
for r in RATIOS:
    sv = table_b_cells[("smolvlm", r)]
    qw = table_b_cells[("qwen", r)]
    ij = table_b_cells[("ijepa", r)]
    latex_b.append(f"{r}\\%  & {sv} & {qw} & {ij} \\\\")
latex_b.append(r"\bottomrule")
if any_incomplete:
    latex_b.append(r"\multicolumn{4}{l}{\footnotesize *Single seed only (see Methodology).}")
latex_b.append(r"\end{tabular}")
latex_b.append(r"\end{table}")
latex_b_str = "\n".join(latex_b)

print("\n" + "=" * 100)
print("TABLE (b) LATEX OUTPUT")
print("=" * 100)
print(latex_b_str)


# ===========================================================================
# TABLE (c): Compute / VRAM / latency comparison
# ===========================================================================
params = {
    "SmolVLM2-2.2B": 2246784880,
    "Qwen2-VL-2B": 2208985600,
    "I-JEPA (ViT-H/14) + LogReg": 630762240 + 3843,
}
vram_measured = load("results/vram_summary.json")
sv_vram, qw_vram = vram_measured["smolvlm_2.2b"], vram_measured["qwen_2b"]
sv_overhead = sv_vram / (params["SmolVLM2-2.2B"] * 2 / 1e9)
qw_overhead = qw_vram / (params["Qwen2-VL-2B"] * 2 / 1e9)
ij_vram_est = (630762240 * 2 / 1e9) * ((sv_overhead + qw_overhead) / 2)

sv_lat = compute_latency("results/exp1_smolvlm_zeroshot_real.json")
qw_lat = compute_latency("results/exp2_qwen_zeroshot_real.json")
ij_lat = load("ijepa_latency_result.json")["per_image_latency_s"]

print("\n" + "=" * 100)
print("TABLE (c) SOURCE PROVENANCE")
print("=" * 100)
print(f"SmolVLM2-2.2B : {params['SmolVLM2-2.2B']/1e9:.2f}B params (HF hub safetensors metadata) | "
      f"{sv_vram:.2f} GB VRAM (results/vram_summary.json, measured) | {sv_lat:.2f} s/img "
      f"(computed from results/exp1_smolvlm_zeroshot_real.json, GPU-measured)")
print(f"Qwen2-VL-2B   : {params['Qwen2-VL-2B']/1e9:.2f}B params (HF hub safetensors metadata) | "
      f"{qw_vram:.2f} GB VRAM (results/vram_summary.json, measured) | {qw_lat:.2f} s/img "
      f"(computed from results/exp2_qwen_zeroshot_real.json, GPU-measured)")
print(f"I-JEPA+LogReg : {params['I-JEPA (ViT-H/14) + LogReg']/1e9:.2f}B params (HF hub safetensors metadata + LogReg head) | "
      f"{ij_vram_est:.2f} GB VRAM ESTIMATED from param count (no GPU available in this analysis environment; "
      f"uses the same fp16-loading overhead ratio observed for the other two rows) | {ij_lat:.2f} s/img "
      f"CPU-measured (no GPU available in this analysis environment -- NOT directly comparable in magnitude "
      f"to the two GPU-measured rows above; I-JEPA's single frozen forward pass with no autoregressive "
      f"generation would be expected to be substantially faster on equivalent GPU hardware)")

latex_c = []
latex_c.append(r"\begin{table}[t]")
latex_c.append(r"\caption{Compute Footprint and Inference Latency}")
latex_c.append(r"\label{tab:compute}")
latex_c.append(r"\centering")
latex_c.append(r"\begin{tabular}{lccc}")
latex_c.append(r"\toprule")
latex_c.append(r"\textbf{Method} & \textbf{Params} & \textbf{Peak VRAM} & \textbf{Latency/image} \\")
latex_c.append(r"\midrule")
latex_c.append(f"SmolVLM2-2.2B & {params['SmolVLM2-2.2B']/1e9:.2f}B & {sv_vram:.2f} GB & {sv_lat:.2f} s \\\\")
latex_c.append(f"Qwen2-VL-2B & {params['Qwen2-VL-2B']/1e9:.2f}B & {qw_vram:.2f} GB & {qw_lat:.2f} s \\\\")
latex_c.append(f"I-JEPA + LogReg & {params['I-JEPA (ViT-H/14) + LogReg']/1e9:.2f}B & {ij_vram_est:.2f} GB$^{{*}}$ & {ij_lat:.2f} s$^{{\\dagger}}$ \\\\")
latex_c.append(r"\bottomrule")
latex_c.append(r"\end{tabular}")
latex_c.append(r"\\[2pt]")
latex_c.append(r"{\footnotesize $^{*}$Estimated from parameter count; SmolVLM/Qwen VRAM measured directly, I-JEPA not (no GPU available for re-measurement).}\\")
latex_c.append(r"{\footnotesize $^{\dagger}$Measured on CPU (no GPU available); SmolVLM/Qwen measured on GPU. Not directly comparable in magnitude -- I-JEPA's single non-autoregressive forward pass would be expected to be substantially faster on equivalent GPU hardware.}")
latex_c.append(r"\end{table}")
latex_c_str = "\n".join(latex_c)

print("\n" + "=" * 100)
print("TABLE (c) LATEX OUTPUT")
print("=" * 100)
print(latex_c_str)


# ===========================================================================
# TABLE (d): Per-class confidence calibration breakdown
# ===========================================================================
# NOTE: degradation-condition (blur/dark/noise) confidence was never recorded --
# confirmed by inspecting flood_vlm_paper_final_experiments.ipynb: run_confidence_analysis()
# (cell 31) was only ever invoked on the clean 30%-blend test set; run_robustness_test()
# (cell 33) saves only filename/ground_truth/prediction/correct, no confidence/logit-score
# field. So only the by-class breakdown below is available from existing data; the
# by-degradation-condition half of this request cannot be built without a new
# generation run (autoregressive .generate() with output_scores=True on 3x238 perturbed
# images through the fine-tuned adapter) which was not attempted given time constraints.
conf_data = load("results/confidence_analysis_blend_30pct.json")
print("\n" + "=" * 100)
print("TABLE (d) SOURCE PROVENANCE")
print("=" * 100)
print("Source: results/confidence_analysis_blend_30pct.json (SmolVLM, 30%-blend adapter, clean condition, n=238)")
print("Degradation-condition (blur/dark/noise) confidence: NOT AVAILABLE -- never recorded in this project (see note above)")

calib_rows = []
for c in CLASSES:
    items = [r for r in conf_data if r["ground_truth"] == c]
    correct = [r["confidence"] for r in items if r["correct"]]
    wrong = [r["confidence"] for r in items if not r["correct"]]
    ac = sum(correct) / len(correct) * 100 if correct else float("nan")
    aw = sum(wrong) / len(wrong) * 100 if wrong else float("nan")
    calib_rows.append((c, len(correct), ac, len(wrong), aw))
    print(f"{c:<14} n_correct={len(correct):>4} avg_conf_correct={ac:5.1f}%   n_wrong={len(wrong):>4} avg_conf_wrong={aw:5.1f}%")

latex_d = []
latex_d.append(r"\begin{table}[t]")
latex_d.append(r"\caption{Per-Class Confidence Calibration (SmolVLM2-2.2B, 30\% Blend, Clean Condition)}")
latex_d.append(r"\label{tab:calibration}")
latex_d.append(r"\centering")
latex_d.append(r"\begin{tabular}{lcccc}")
latex_d.append(r"\toprule")
latex_d.append(r"\textbf{Class} & \textbf{Conf. (correct)} & \textbf{$n$} & \textbf{Conf. (wrong)} & \textbf{$n$} \\")
latex_d.append(r"\midrule")
for c, nc, ac, nw, aw in calib_rows:
    latex_d.append(f"{c} & {ac:.1f}\\% & {nc} & {aw:.1f}\\% & {nw} \\\\")
latex_d.append(r"\bottomrule")
latex_d.append(r"\end{tabular}")
latex_d.append(r"\\[2pt]")
latex_d.append(r"{\footnotesize Breakdown by degradation condition (blur/dark/noise) was not available -- confidence scores were only recorded for the clean-condition evaluation in this project.}")
latex_d.append(r"\end{table}")
latex_d_str = "\n".join(latex_d)

print("\n" + "=" * 100)
print("TABLE (d) LATEX OUTPUT")
print("=" * 100)
print(latex_d_str)

# ===========================================================================
# TABLE (e): Parameter-matched linear-probe control (Tier 1b)
# ===========================================================================
def acc_mr(fn):
    d = load(fn)
    total = defaultdict(int)
    correct = defaultdict(int)
    n_correct = 0
    for r in d:
        gt = r["ground_truth"]
        total[gt] += 1
        if r["correct"]:
            correct[gt] += 1
            n_correct += 1
    recalls = [correct[c] / total[c] * 100 for c in CLASSES]
    return n_correct / len(d) * 100, sum(recalls) / 3

def mean_std(fname_tmpl):
    accs, mrs = [], []
    for s in [42, 123, 7]:
        a, m = acc_mr(fname_tmpl.format(s))
        accs.append(a)
        mrs.append(m)
    return (sum(accs) / 3, (sum((x - sum(accs) / 3) ** 2 for x in accs) / 3) ** 0.5,
            sum(mrs) / 3, (sum((x - sum(mrs) / 3) ** 2 for x in mrs) / 3) ** 0.5)

probe_rows = [
    ("SmolVLM linear-probe (frozen + LogReg)", *mean_std("results/linear_probe_smolvlm_80pct_seed{}.json")),
    ("Qwen linear-probe (frozen + LogReg)", *mean_std("results/linear_probe_qwen_80pct_seed{}.json")),
    ("I-JEPA (frozen + LogReg)", *mean_std("results/ijepa_split_80pct_seed{}.json")),
    ("SmolVLM full QLoRA fine-tune", *mean_std("new_results/eval_smolvlm_split_80pct_seed{}.json")),
    ("Qwen full QLoRA fine-tune", *mean_std("new_results/eval_qwen_split_80pct_seed{}.json")),
]

print("\n" + "=" * 100)
print("TABLE (e) SOURCE PROVENANCE -- Tier 1b linear-probe control, all computed from raw per-image files, 80% blend, 3 seeds")
print("=" * 100)
for name, am, asd, mm, msd in probe_rows:
    print(f"{name:<42} acc={am:5.1f}+-{asd:4.1f}%  macro_recall={mm:5.1f}+-{msd:4.1f}%")

gap_smolvlm_full = probe_rows[2][3] - probe_rows[3][3]
gap_qwen_full = probe_rows[2][3] - probe_rows[4][3]
gap_smolvlm_probe = probe_rows[2][3] - probe_rows[0][3]
gap_qwen_probe = probe_rows[2][3] - probe_rows[1][3]
print(f"\nGap to I-JEPA macro-recall -- SmolVLM full fine-tune: {gap_smolvlm_full:.1f}pp | SmolVLM linear-probe: {gap_smolvlm_probe:.1f}pp")
print(f"Gap to I-JEPA macro-recall -- Qwen full fine-tune:     {gap_qwen_full:.1f}pp | Qwen linear-probe:     {gap_qwen_probe:.1f}pp")
print("CONCLUSION: linear-probing closes the vast majority of the gap for BOTH VLMs (16-38pp -> ~3pp),")
print("indicating trainable-parameter count / limited-data overfitting -- not the self-supervised-vs-")
print("vision-language pretraining distinction -- is the dominant factor behind I-JEPA's original advantage.")
print("I-JEPA still retains a small residual edge (~3pp) over both linear-probed VLMs, consistent with a")
print("secondary, smaller benefit from its self-supervised pretraining on top of the parameter-count effect.")

latex_e = []
latex_e.append(r"\begin{table}[t]")
latex_e.append(r"\caption{Parameter-Matched Linear-Probe Control (80\% Blend, 3 Seeds)}")
latex_e.append(r"\label{tab:linearprobe}")
latex_e.append(r"\centering")
latex_e.append(r"\begin{tabular}{lcc}")
latex_e.append(r"\toprule")
latex_e.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Macro-Recall} \\")
latex_e.append(r"\midrule")
for name, am, asd, mm, msd in probe_rows:
    latex_e.append(f"{name} & {am:.1f} $\\pm$ {asd:.1f}\\% & {mm:.1f} $\\pm$ {msd:.1f}\\% \\\\")
latex_e.append(r"\bottomrule")
latex_e.append(r"\end{tabular}")
latex_e.append(r"\end{table}")
latex_e_str = "\n".join(latex_e)

print("\n" + "=" * 100)
print("TABLE (e) LATEX OUTPUT")
print("=" * 100)
print(latex_e_str)

with open("generated_tables.tex", "w") as f:
    f.write(latex_a_str + "\n\n" + latex_b_str + "\n\n" + latex_c_str + "\n\n" + latex_d_str + "\n\n" + latex_e_str + "\n")
print("\nSaved generated_tables.tex")
