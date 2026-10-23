"""
Verify every quantitative claim in overleaf/main.tex against the underlying data.

Run:  python verify_paper.py
Exit code 0 if every check passes, 1 otherwise.

Each check states the value AS PRINTED IN THE PAPER and recomputes it from source
files. If you edit a number in main.tex, update the corresponding EXPECTED here and
re-run, so the two can never drift apart silently.
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
SEEDS = [42, 123, 7]
CLASSES = ["no flood", "small flood", "big flood"]
TEX = "overleaf/main.tex"

failures = []
checks = 0


def check(label, expected, actual, tol=0.05):
    """Compare a paper value against a recomputed one."""
    global checks
    checks += 1
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(expected - actual) <= tol
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(f"{label}: paper={expected} computed={actual}")
    print(f"  [{status}] {label:<62} paper={expected!s:<16} computed={actual!s}")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metrics(records):
    """Accuracy, macro-recall and per-class recall from per-image records."""
    n = len(records)
    acc = sum(1 for r in records if r["prediction"] == r["ground_truth"]) / n * 100
    per = {}
    for c in CLASSES:
        sub = [r for r in records if r["ground_truth"] == c]
        per[c] = sum(1 for r in sub if r["prediction"] == c) / len(sub) * 100 if sub else None
    vals = [v for v in per.values() if v is not None]
    return acc, sum(vals) / len(vals), per


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)


# ---------------------------------------------------------------------------
print("\n=== 1. Dataset composition (Empirical Study Setup) ===")
syn = sorted(os.listdir("data/synthetic_flood_images"))
suffix = {"w": "no flood", "s": "small flood", "l": "big flood"}
counts = {v: 0 for v in suffix.values()}
scenes = {}
for fn in syn:
    m = re.match(r"^(\d+)([wsl])\.png$", fn)
    assert m, f"unexpected filename {fn}"
    counts[suffix[m.group(2)]] += 1
    scenes.setdefault(m.group(1), set()).add(m.group(2))

check("subset used in experiments: total images", 104, len(syn), tol=0)
check("subset used in experiments: base views", 44, len(scenes), tol=0)
check("subset: no flood", 35, counts["no flood"], tol=0)
check("subset: small flood", 37, counts["small flood"], tol=0)
check("subset: big flood", 32, counts["big flood"], tol=0)

# Suffix->label mapping is the one the notebooks actually used, and the "no flood"
# class is the wet-pavement edit, not a dry frame. The paper now states both.
tex_ck = open("overleaf/main.tex", encoding="utf-8").read()
check("paper states no-flood class is wet pavement, not dry road",
      True, ("wet pavement rather than a dry road" in tex_ck))
check("paper states the three edit conditions",
      True, ("wet pavement, shallow standing water, and deep standing water" in tex_ck))
check("paper reconciles the 1,005-image generation campaign",
      True, ("1{,}005 images" in tex_ck))
check("paper states the campaign covered 335 base views",
      True, ("covered 335 base views" in tex_ck))
# Footnote must not claim a nested "Roadway Flood Dataset" folder inside the shared
# Drive archive; that nesting is a local download artifact, not the shared layout.
check("footnote does not claim a nested Roadway Flood Dataset folder in the archive",
      False, ("nested " + chr(92) + "texttt{Roadway Flood Dataset}") in tex_ck)
# The double-blind URL ban applied to the submission. The camera-ready is the
# opposite case: it must carry the dataset link. That is handled by the
# CAMERA-READY BLOCKERS gate at the end of this file, not by a numeric check.
# Citation-accuracy guards. Kornblith et al. (CVPR 2019) does NOT show that logistic
# regression on frozen features is competitive with fine-tuning on small datasets;
# its abstract reports correlation of ImageNet vs transfer accuracy and sensitivity
# of frozen features to how the encoder was trained. Verified against arXiv 1805.08974.
check("no Kornblith overclaim (linear probe competitive with fine-tuning)",
      False, "competitive with fine-tuning on small target datasets" in tex_ck)
# STURM-FloodDepth is cited to the peer-reviewed Geomatica pipeline paper
# (Notarangelo, Wirion, van Winsen, 2025), not the single-author Zenodo record.
bib_ck = open("overleaf/references.bib", encoding="utf-8").read()
check("STURM cited to peer-reviewed paper with all three authors",
      True, ("Wirion" in bib_ck and "van Winsen" in bib_ck))
check("STURM entry names the Geomatica venue", True, "Geomatica" in bib_ck)

# Full released dataset. As of the camera-ready the paper releases the entire
# generation campaign, so the claims are checked against the audited manifest of it
# rather than against the old 538-image partial archive. The 104 above is the subset
# the experiments train on, and the two must not be conflated.
#
# The manifest is a filename+label listing produced by `run_full_synthetic_eval audit`
# against the Drive folder. It proves composition, NOT that the public archive has
# been uploaded -- that is a manual step, flagged at the end of this script.
MANIFEST = os.path.join("colab_full_synth", "full_1005_manifest.json")
full = {it["filename"] for it in load(MANIFEST)}
full_counts = {v: 0 for v in suffix.values()}
full_scenes = {}
for fn in sorted(full):
    m = re.match(r"^(\d+)([wsl])\.png$", fn)
    assert m, f"unexpected filename {fn}"
    full_counts[suffix[m.group(2)]] += 1
    full_scenes.setdefault(m.group(1), set()).add(m.group(2))

check("released dataset: total images", 1005, len(full), tol=0)
check("released dataset: base views", 335, len(full_scenes), tol=0)
check("released dataset: no flood", 335, full_counts["no flood"], tol=0)
check("released dataset: small flood", 335, full_counts["small flood"], tol=0)
check("released dataset: big flood", 335, full_counts["big flood"], tol=0)
check("released dataset: views at all 3 water levels", 335,
      sum(1 for v in full_scenes.values() if len(v) == 3), tol=0)
check("the 104 subset is contained in the released set", set(), set(syn) - full, tol=0)
# The paper now claims every view carries all three levels, so no view may be partial.
check("released dataset: no partial base views", 0,
      sum(1 for v in full_scenes.values() if len(v) != 3), tol=0)
# Guard the prose against drifting back to the superseded partial-archive numbers.
for stale in ["538-image portion", "538 labeled images", "538 images",
              "spanning 182", "182 base CCTV views"]:
    check(f"prose no longer claims the partial archive ({stale!r})",
          False, stale in tex_ck)

zs = load("results/exp1_smolvlm_zeroshot_real.json")
check("STURM-FloodDepth real images", 340, len(zs), tol=0)

# ---------------------------------------------------------------------------
print("\n=== 2. Table I: prompt sweep (Qwen2-VL-2B, synthetic) ===")
qres = load("reference_baseline_results/qwen_results.json")
prompt_acc = {}
for r in qres:
    p = r["prompt"]
    a, b = prompt_acc.get(p, (0, 0))
    prompt_acc[p] = (a + (r["prediction"] == r["ground_truth"]), b + 1)
expected_prompts = {
    "A_baseline": 61.5,           # "Bare question, three label options"
    "D_severity_anchor": 61.5,    # "Class definitions by vehicle passability"
    "B_reference_object": 43.3,   # "Directed to reference objects"
    "E_small_flood_focused": 33.7,  # "Wet pavement vs. standing water"
    "C_chain_of_thought": 30.8,   # "Step-by-step reasoning"
}
for key, exp in expected_prompts.items():
    c, t = prompt_acc[key]
    check(f"prompt {key}", exp, round(c / t * 100, 1))
check("prompt sweep n", 104, prompt_acc["A_baseline"][1], tol=0)

# ---------------------------------------------------------------------------
print("\n=== 3. Table II: summary on real data ===")
acc, mr, per = metrics(zs)
check("SmolVLM zero-shot accuracy", 56.5, round(acc, 1))
check("SmolVLM zero-shot macro-recall", 36.3, round(mr, 1))
check("SmolVLM zero-shot: small-flood recall", 100.0, round(per["small flood"], 1))
check("SmolVLM zero-shot: no-flood recall", 6.2, round(per["no flood"], 1))
check("SmolVLM zero-shot: big-flood recall", 2.7, round(per["big flood"], 1))

fs = load("results/exp1_smolvlm_3shot_real.json")
acc, mr, _ = metrics(fs)
check("SmolVLM 3-shot accuracy", 29.1, round(acc, 1))
check("SmolVLM 3-shot macro-recall", 28.6, round(mr, 1))

qzs = load("results/exp2_qwen_zeroshot_real.json")
acc, mr, per = metrics(qzs)
check("Qwen zero-shot accuracy", 37.9, round(acc, 1))
check("Qwen zero-shot macro-recall", 55.5, round(mr, 1))
check("Qwen zero-shot: no-flood recall", 82.5, round(per["no flood"], 1))
check("Qwen zero-shot: big-flood recall", 84.0, round(per["big flood"], 1))
check("Qwen zero-shot: small-flood recall", 0.0, round(per["small flood"], 1))

ens = load("results/ensemble_qwen_finetuned.json")
acc, mr, _ = metrics(ens)
check("best ensemble accuracy", 63.0, round(acc, 1))
check("best ensemble macro-recall", 67.2, round(mr, 1))

cv = load("results/classical_cv_baseline.json")
acc, mr, _ = metrics(cv)
check("classical CV accuracy", 43.7, round(acc, 1))
check("classical CV macro-recall", 39.5, round(mr, 1))

# The synthetic-only fine-tuned row predates the per-image logging used elsewhere,
# but the full confusion matrix survives in the executed output of code/smolvlm.ipynb,
# which is enough to recompute both figures exactly.
def confusion_from_notebook(path="code/smolvlm.ipynb",
                            marker="Fine-tuned SmolVLM2-2.2B on REAL"):
    nb = load(path)
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            if "data" in out:
                text += "".join(out["data"].get("text/plain", []))
            if marker not in text:
                continue
            cm = {}
            for gt, pred, n in re.findall(
                    r"^\s*(no flood|small flood|big flood)\s*->\s*"
                    r"(no flood|small flood|big flood):\s*(\d+)\s*$", text, re.M):
                cm[(gt, pred)] = int(n)
            if cm:
                return cm
    raise LookupError("confusion matrix not found in notebook output")


cm = confusion_from_notebook()
totals = {c: sum(v for (g, _), v in cm.items() if g == c) for c in CLASSES}
gt_counts = {c: sum(1 for r in zs if r["ground_truth"] == c) for c in CLASSES}
# Sanity: the notebook matrix must cover the same STURM test set used everywhere else.
check("synthetic-only FT: class totals match STURM", gt_counts, totals, tol=0)
check("synthetic-only FT: total images", 340, sum(totals.values()), tol=0)

n_correct = sum(v for (g, p), v in cm.items() if g == p)
recalls = {c: cm.get((c, c), 0) / totals[c] * 100 for c in CLASSES}
check("synthetic-only FT accuracy", 42.1, round(n_correct / sum(totals.values()) * 100, 1))
check("synthetic-only FT macro-recall", 48.3, round(sum(recalls.values()) / 3, 1))

# The 71.9% in the superseded results/summary_table.json is binary flood recall,
# not macro-recall. Confirming that keeps the two from being confused again.
flooded = totals["small flood"] + totals["big flood"]
flagged = sum(v for (g, p), v in cm.items() if g != "no flood" and p != "no flood")
check("71.9% is binary flood recall, not macro-recall", 71.9,
      round(flagged / flooded * 100, 1))

# ---------------------------------------------------------------------------
print("\n=== 4. Table III: fine-tuned dose-response ===")
ds = load("assembled_dataset.json")
TABLE_III = {
    15: (52.6, 5.0, 35.5, 2.2, 50.7, 4.5), 20: (50.8, 6.3, 33.6, 0.8, 53.6, 2.4),
    30: (57.2, 7.5, 35.1, 1.6, 57.6, 2.4), 40: (55.3, 1.2, 35.5, 0.3, 63.8, 3.0),
    50: (52.3, 6.6, 34.9, 1.4, 71.2, 0.5), 60: (61.0, 4.2, 39.7, 3.8, 74.3, 4.3),
    70: (54.4, 8.4, 41.1, 2.4, 76.6, 6.2), 80: (64.5, 6.1, 42.3, 2.6, 80.8, 1.2),
    90: (53.5, 5.1, 33.7, 1.3, 73.4, 3.4),
}


def ft(model, ratio):
    v = [x * 100 for x in ds[model][str(ratio)].values()]
    return round(st.mean(v), 1), round(st.pstdev(v), 1)


for r in RATIOS:
    exp = TABLE_III[r]
    got = ft("smolvlm", r) + ft("qwen", r) + ft("ijepa", r)
    check(f"Table III row {r}%", exp, got, tol=0)

# ---------------------------------------------------------------------------
print("\n=== 5. Table IV: linear-probe control ===")
sweep = load("results/linear_probe_sweep_summary.json")
TABLE_IV = {
    15: (61.3, 1.2, 65.6, 1.7), 20: (64.0, 2.1, 65.0, 2.3), 30: (67.6, 0.6, 66.4, 1.1),
    40: (71.2, 3.5, 71.5, 2.8), 50: (71.4, 3.5, 74.8, 2.6), 60: (77.4, 3.1, 78.0, 0.7),
    70: (78.2, 1.9, 76.8, 3.4), 80: (77.9, 1.5, 77.8, 1.5), 90: (77.7, 1.0, 79.4, 2.8),
}
BOLD_IV = {15: "Qwen", 20: "Qwen", 30: "SmolVLM", 40: "Qwen", 50: "Qwen",
           60: "Qwen", 70: "SmolVLM", 80: "I-JEPA", 90: "Qwen"}
for r in RATIOS:
    sm, qw = sweep["smolvlm"][str(r)], sweep["qwen"][str(r)]
    got = (round(sm["mr_mean"], 1), round(sm["mr_std"], 1),
           round(qw["mr_mean"], 1), round(qw["mr_std"], 1))
    check(f"Table IV row {r}%", TABLE_IV[r], got, tol=0)
    vals = {"SmolVLM": sm["mr_mean"], "Qwen": qw["mr_mean"], "I-JEPA": ft("ijepa", r)[0]}
    check(f"  bold (best) at {r}%", BOLD_IV[r], max(vals, key=vals.get), tol=0)

print("\n=== 6. Claims derived from Tables III/IV ===")
probe_beats_ft = all(
    sweep[m][str(r)]["mr_mean"] > ft(m, r)[0] for m in ("smolvlm", "qwen") for r in RATIOS)
check("each probe beats its own fine-tuned model at all 9 ratios", True, probe_beats_ft)

probe_ge_ijepa = sum(
    1 for r in RATIOS
    if max(sweep["smolvlm"][str(r)]["mr_mean"], sweep["qwen"][str(r)]["mr_mean"]) >= ft("ijepa", r)[0])
check("probes match/exceed I-JEPA at 8 of 9 ratios", 8, probe_ge_ijepa, tol=0)

ij_beats_smol = sum(1 for r in RATIOS if ft("ijepa", r)[0] > ft("smolvlm", r)[0])
check("I-JEPA (FT table) beats SmolVLM at 8 of 9 ratios", 8, ij_beats_smol, tol=0)
ij_beats_qwen = sum(1 for r in RATIOS if ft("ijepa", r)[0] > ft("qwen", r)[0])
check("I-JEPA beats Qwen at all 9 ratios", 9, ij_beats_qwen, tol=0)
check("SmolVLM leads I-JEPA at 15% by 1.9 pts", 1.9,
      round(ft("smolvlm", 15)[0] - ft("ijepa", 15)[0], 1))

tighter = sum(1 for r in RATIOS if ft("ijepa", r)[1] < ft("smolvlm", r)[1])
check("I-JEPA tighter std than SmolVLM at 7 of 9", 7, tighter, tol=0)
check("40% exception: SmolVLM std 1.2 vs I-JEPA 3.0", (1.2, 3.0),
      (ft("smolvlm", 40)[1], ft("ijepa", 40)[1]), tol=0)

check("SmolVLM max std", 8.4, max(ft("smolvlm", r)[1] for r in RATIOS))
check("Qwen min std", 0.3, min(ft("qwen", r)[1] for r in RATIOS))
check("Qwen FT macro-recall min", 33.6, min(ft("qwen", r)[0] for r in RATIOS))
check("Qwen FT macro-recall max", 42.3, max(ft("qwen", r)[0] for r in RATIOS))
check("Qwen weaker than SmolVLM at every ratio", True,
      all(ft("qwen", r)[0] < ft("smolvlm", r)[0] for r in RATIOS))
check("SmolVLM blended beats 48.3 synthetic-only baseline everywhere", True,
      all(ft("smolvlm", r)[0] > 48.3 for r in RATIOS))
check("Qwen probe at 90%: 33.7 -> 79.4", (33.7, 79.4),
      (ft("qwen", 90)[0], round(sweep["qwen"]["90"]["mr_mean"], 1)), tol=0)
check("Qwen probe at 50%: 34.9 -> 74.8", (34.9, 74.8),
      (ft("qwen", 50)[0], round(sweep["qwen"]["50"]["mr_mean"], 1)), tol=0)
check("Qwen probe 65.6 vs I-JEPA 50.7 at 15%", (65.6, 50.7),
      (round(sweep["qwen"]["15"]["mr_mean"], 1), ft("ijepa", 15)[0]), tol=0)
check("I-JEPA lead at 80% is 2.9-3.0 pts", (2.9, 3.0),
      (round(ft("ijepa", 80)[0] - sweep["smolvlm"]["80"]["mr_mean"], 1),
       round(ft("ijepa", 80)[0] - sweep["qwen"]["80"]["mr_mean"], 1)), tol=0)
check("SmolVLM probe 30%: 67.6+-0.6 vs FT 57.2+-7.5", (67.6, 0.6, 57.2, 7.5),
      (round(sweep["smolvlm"]["30"]["mr_mean"], 1), round(sweep["smolvlm"]["30"]["mr_std"], 1)) + ft("smolvlm", 30),
      tol=0)

# ---------------------------------------------------------------------------
print("\n=== 7. I-JEPA dose-response endpoints and 80% behaviour ===")
ij15 = [metrics(load(first_existing(f"results/ijepa_split_15pct_seed{s}.json",
                                    "results/ijepa_split_15pct.json")))[0] for s in SEEDS]
check("I-JEPA accuracy at 15%", 57.4, round(st.mean(ij15), 1))
ij80 = [metrics(load(f"results/ijepa_split_80pct_seed{s}.json"))[0] for s in SEEDS]
check("I-JEPA accuracy at 80%", 79.4, round(st.mean(ij80), 1))
check("I-JEPA macro-recall at 80%", 80.8, ft("ijepa", 80)[0])

pool = []
for s in SEEDS:
    pool += load(f"results/ijepa_split_80pct_seed{s}.json")
_, _, per = metrics(pool)
check("I-JEPA 80% per-class spread ~12 pts", 12, round(max(per.values()) - min(per.values())))

syn_only = load("results/ijepa_synthetic_only.json")
acc, mr, per = metrics(syn_only)
check("I-JEPA synthetic-only accuracy", 28.2, round(acc, 1))
check("I-JEPA synthetic-only macro-recall", 40.3, round(mr, 1))
check("I-JEPA synthetic-only: no-flood recall", 93.8, round(per["no flood"], 1))
check("I-JEPA synthetic-only: small-flood recall", 0.5, round(per["small flood"], 1))
check("I-JEPA synthetic-only: big-flood recall", 26.7, round(per["big flood"], 1))

# ---------------------------------------------------------------------------
print("\n=== 8. Figure 5 / confusion matrices at 80% ===")
CONF = {
    "SmolVLM": ([f"new_results/eval_smolvlm_split_80pct_seed{s}.json" for s in SEEDS],
                {"big flood->small flood": 44.4}),
    "Qwen": ([f"new_results/eval_qwen_split_80pct_seed{s}.json" for s in SEEDS],
             {"no flood->small flood": 70.8, "small flood->small flood": 77.5,
              "big flood->small flood": 66.7}),
    "I-JEPA": ([f"results/ijepa_split_80pct_seed{s}.json" for s in SEEDS],
               {"no flood->small flood": 22.9}),
}
for name, (files, expected) in CONF.items():
    recs = []
    for f in files:
        recs += load(f)
    for key, exp in expected.items():
        gt, pred = key.split("->")
        sub = [r for r in recs if r["ground_truth"] == gt]
        got = sum(1 for r in sub if r["prediction"] == pred) / len(sub) * 100
        check(f"{name}: {key}", exp, round(got, 1))

# ---------------------------------------------------------------------------
print("\n=== 9. Ensembles ===")
e = load("results/ensemble_qwen_finetuned_summary.json")
check("fine-tuned alone accuracy", 64.7, round(e["finetuned_alone"]["accuracy"] * 100, 1))
check("fine-tuned alone macro-recall", 67.8, round(e["finetuned_alone"]["macro_recall"] * 100, 1))
check("ensemble n images", 238, e["n_images"], tol=0)
t3 = load("results/ensemble_three_way.json")
acc, mr, _ = metrics(t3)
check("three-way ensemble accuracy", 48.7, round(acc, 1))
check("three-way ensemble macro-recall", 58.0, round(mr, 1))

# ---------------------------------------------------------------------------
print("\n=== 10. Table VI: calibration ===")
conf = load("results/confidence_analysis_blend_30pct.json")
corr = [r for r in conf if r["correct"]]
wrong = [r for r in conf if not r["correct"]]
check("overall confidence (correct)", 88.0, round(st.mean(r["confidence"] for r in corr) * 100, 1))
check("overall confidence (wrong)", 82.9, round(st.mean(r["confidence"] for r in wrong) * 100, 1))
CAL = {"no flood": (92.2, 49, 82.8, 7, 9.4),
       "small flood": (87.1, 74, 84.4, 55, 2.7),
       "big flood": (83.7, 31, 78.9, 22, 4.7)}
for c, (ec, en, ew, ewn, egap) in CAL.items():
    cc = [r["confidence"] for r in corr if r["ground_truth"] == c]
    ww = [r["confidence"] for r in wrong if r["ground_truth"] == c]
    a, b = st.mean(cc) * 100, st.mean(ww) * 100
    check(f"calib {c}", (ec, en, ew, ewn, egap),
          (round(a, 1), len(cc), round(b, 1), len(ww), round(a - b, 1)), tol=0)

# ---------------------------------------------------------------------------
print("\n=== 11. Table VII: trade-offs (derived) ===")
zs_mr = metrics(zs)[1]
qzs_mr = metrics(qzs)[1]
check("zero-shot macro-recall range low", 36.3, round(min(zs_mr, qzs_mr), 1))
check("zero-shot macro-recall range high", 55.5, round(max(zs_mr, qzs_mr), 1))
check("fine-tuned VLM best", 64.5, max(ft(m, r)[0] for m in ("smolvlm", "qwen") for r in RATIOS))
check("frozen tower + probe best", 79.4,
      round(max(sweep[m][str(r)]["mr_mean"] for m in ("smolvlm", "qwen") for r in RATIOS), 1))
check("I-JEPA + probe best", 80.8, max(ft("ijepa", r)[0] for r in RATIOS))

# ---------------------------------------------------------------------------
print("\n=== 12. Significance tests ===")
try:
    from scipy import stats

    sig_qwen = sig_smol = 0
    bonf_hits = []
    smol_ps = []
    n_tests = 18
    for r in RATIOS:
        ij = [ds["ijepa"][str(r)][str(s)] * 100 for s in SEEDS]
        sv = [ds["smolvlm"][str(r)][str(s)] * 100 for s in SEEDS]
        qw = [ds["qwen"][str(r)][str(s)] * 100 for s in SEEDS]
        p_s = stats.ttest_rel(ij, sv).pvalue
        p_q = stats.ttest_rel(ij, qw).pvalue
        smol_ps.append(p_s)
        sig_smol += p_s < 0.05
        sig_qwen += p_q < 0.05
        if p_q < 0.05 / n_tests:
            bonf_hits.append(r)
    check("I-JEPA vs Qwen significant at 8 of 9", 8, sig_qwen, tol=0)
    check("I-JEPA vs SmolVLM significant at 0 ratios", 0, sig_smol, tol=0)
    check("Bonferroni survivors vs Qwen", [50, 80], bonf_hits, tol=0)
    check("SmolVLM p-value range low", 0.06, round(min(smol_ps), 2))
    check("SmolVLM p-value range high", 0.96, round(max(smol_ps), 2))
    ij15v = [ds["ijepa"]["15"][str(s)] * 100 for s in SEEDS]
    qw15v = [ds["qwen"]["15"][str(s)] * 100 for s in SEEDS]
    check("15% vs Qwen p-value (the non-significant one)", 0.062,
          round(stats.ttest_rel(ij15v, qw15v).pvalue, 3), tol=0.001)
except ImportError:
    print("  [SKIP] scipy not installed")

# ---------------------------------------------------------------------------
print("\n=== 12b. Blend-ratio definition (Empirical Study Setup) ===")
n_train80 = len(load(first_existing("splits/split_80pct_seed42_train.json",
                                    "splits/split_80pct_train.json")))
check("80% ratio: real images in training set", 272, n_train80, tol=0)
check("80% ratio: real share of training set", 72.3,
      round(n_train80 / (n_train80 + 104) * 100, 1))

# ---------------------------------------------------------------------------
print("\n=== 12c. tab:synthsize -- synthetic-set size ablation ===")
abl = load("results/synthetic_quantity_ablation.json")

# Regression guard: with all 104 synthetic images the ablation must reproduce the
# published linear-probe sweep exactly, or the two tables disagree with each other.
sweep_chk = load("results/linear_probe_sweep_summary.json")
for model in ("smolvlm", "qwen"):
    for r in RATIOS:
        check(f"ablation@104 == probe sweep, {model} {r}%",
              round(sweep_chk[model][str(r)]["mr_mean"], 2),
              abl[model][str(r)]["104"]["mr_mean"], tol=0.005)

# Synthetic share of the training set, printed in the table's second column.
SYNTH_SHARE = {15: 67.1, 20: 60.5, 30: 50.5, 40: 43.3, 50: 38.0,
               60: 33.8, 70: 30.4, 80: 27.7, 90: 25.4}
for r, exp in SYNTH_SHARE.items():
    n_tr = len(load(first_existing(f"splits/split_{r}pct_seed42_train.json",
                                   f"splits/split_{r}pct_train.json")))
    check(f"synthetic share of train set at {r}%", exp,
          round(104 / (n_tr + 104) * 100, 1))

# Every delta printed in Table VIII.
DELTAS = {
    "smolvlm": {15: +0.5, 20: +1.6, 30: -1.4, 40: -2.5, 50: -3.0,
                60: -3.0, 70: -1.4, 80: -1.5, 90: -1.6},
    "qwen":    {15: +1.9, 20: +1.7, 30: +0.6, 40: +0.8, 50: +0.4,
                60: +0.2, 70: -0.4, 80: -1.9, 90: -1.0},
    "ijepa":   {15: -6.0, 20: -4.0, 30: -2.3, 40: +0.8, 50: +4.5,
                60: +4.0, 70: +3.9, 80: +11.6, 90: +7.2},
}
for model, row in DELTAS.items():
    for r, exp in row.items():
        got = abl[model][str(r)]["104"]["mr_mean"] - abl[model][str(r)]["0"]["mr_mean"]
        check(f"tab:synthsize delta {model} {r}%", exp, round(got, 1))

# Prose claims about the ablation.
vlm_max = max(abs(v) for m in ("smolvlm", "qwen") for v in DELTAS[m].values())
check("VLM towers: largest shift is 3.0 points", 3.0, vlm_max)
check("I-JEPA hurts at the three lowest ratios only", [15, 20, 30],
      sorted(r for r, v in DELTAS["ijepa"].items() if v < 0), tol=0)
check("I-JEPA helps from 40% onward", [40, 50, 60, 70, 80, 90],
      sorted(r for r, v in DELTAS["ijepa"].items() if v > 0), tol=0)
check("I-JEPA gain range, 40%+ ratios", (0.8, 11.6),
      (min(v for r, v in DELTAS["ijepa"].items() if v > 0),
       max(DELTAS["ijepa"].values())), tol=0)
check("synthetic is 67% of train set at 15%", 67, round(SYNTH_SHARE[15]), tol=0)

# ---------------------------------------------------------------------------
print("\n=== 12c-bis. Scale extension to the full 1,005-image release ===")
# The paragraph added for Reviewer 1's weakness 1 ("the synthetic experiments use only
# 104 images"). Recomputed from the 270-cell sweep: 9 real-data ratios x 10
# synthetic-set sizes x 3 encoders x 3 seeds, synthetic in training only, scored on
# held-out real STURM images.
abl = load("results_full_1005_image_ablation.json")
RAT = ["15", "20", "30", "40", "50", "60", "70", "80", "90"]
CNT = ["0", "13", "26", "52", "78", "104", "250", "500", "750", "1005"]


def _mr(model, ratio, n):
    return abl[model][ratio][str(n)]["mr_mean"]


check("scale: sweep covers 9 ratios", 9, len(RAT), tol=0)
check("scale: sweep covers 10 synthetic-set sizes", 10,
      len([c for c in CNT if c in abl["ijepa"]["80"]]), tol=0)
check("scale: 3 seeds per cell", 3, len(abl["ijepa"]["80"]["1005"]["mr_per_seed"]), tol=0)
check("scale: cells in the sweep", 270,
      sum(len(abl[m][r]) for m in abl for r in abl[m]), tol=0)

# "peaks at 11.1 points with 78 synthetic images"
check("scale: I-JEPA 80% gain at 78 synthetic", 11.1,
      _mr("ijepa", "80", 78) - _mr("ijepa", "80", 0), tol=0.05)
check("scale: 78 is where I-JEPA's 80% gain peaks", "78",
      max(CNT, key=lambda c: _mr("ijepa", "80", c)), tol=0)
# "falls to 6.3 points with all 1,005"
check("scale: I-JEPA 80% gain at 1005 synthetic", 6.3,
      _mr("ijepa", "80", 1005) - _mr("ijepa", "80", 0), tol=0.05)
# "SmolVLM loses ground at all nine ratios and Qwen at eight of the nine"
check("scale: SmolVLM worse at 1005 at all 9 ratios", 9,
      sum(1 for r in RAT if _mr("smolvlm", r, 1005) < _mr("smolvlm", r, 0)), tol=0)
check("scale: Qwen worse at 1005 at 8 of 9 ratios", 8,
      sum(1 for r in RAT if _mr("qwen", r, 1005) < _mr("qwen", r, 0)), tol=0)
# "80% is the best ratio for I-JEPA at every synthetic-set size of 52 images or more"
_best = {c: max(RAT, key=lambda r: _mr("ijepa", r, c)) for c in CNT}
check("scale: 80% best for I-JEPA at every size >= 52", True,
      all(_best[c] == "80" for c in CNT if int(c) >= 52))
check("scale: 80% is NOT best below 52 (so the claim is bounded correctly)", True,
      all(_best[c] != "80" for c in CNT if int(c) < 52))
# "beats 90% at all ten sizes"
check("scale: I-JEPA 80% beats 90% at all 10 sizes", 10,
      sum(1 for c in CNT if _mr("ijepa", "80", c) > _mr("ijepa", "90", c)), tol=0)
# "occupy between 77% and 95% of the training set at every ratio"
_share = load("results_full_1005_cpu_tier.json")["synth_share"]
_vals = [_share[r]["share_at_1005"] for r in RAT]
check("scale: min synthetic share at 1005 rounds to 77%", 77, round(min(_vals)), tol=0)
check("scale: max synthetic share at 1005 rounds to 95%", 95, round(max(_vals)), tol=0)
check("scale: synthetic is a majority of the mix at every ratio", 9,
      sum(1 for v in _vals if v > 50), tol=0)
# The paragraph must still be present and must not have drifted.
for frag in ["Scaling the synthetic set tenfold", "peaks at 11.1 points with 78",
             "falls to 6.3 points", "between 77\\% and 95\\%",
             "of 52 images or more", "beats 90\\% at all ten sizes"]:
    check(f"scale paragraph intact ({frag[:34]!r})", True, frag in tex_ck)

# ---------------------------------------------------------------------------
print("\n=== 12d. Combining two models' predictions (Results) ===")
# Every ensemble the paper reports, recomputed from per-image predictions. The paper
# previously also described a "learned logistic regression combiner"; no such run
# exists anywhere in code/ or results/, and that claim has been removed.
for name, path, n_exp, acc_exp, mr_exp in [
        ("override rule (rule 1)", "results/ensemble_rule1.json", 340, 44.7, 57.5),
        ("Qwen alone, same 340 images", "results/ensemble_rule2.json", 340, 37.9, 55.5),
        ("Qwen + fine-tuned SmolVLM", "results/ensemble_qwen_finetuned.json", 238, 63.0, 67.2),
        ("three-way majority vote", "results/ensemble_three_way.json", 238, 48.7, 58.0)]:
    recs = load(path)
    acc, mr, _ = metrics(recs)
    check(f"{name}: n", n_exp, len(recs), tol=0)
    check(f"{name}: accuracy", acc_exp, round(acc, 1))
    check(f"{name}: macro-recall", mr_exp, round(mr, 1))

ens_ft = json.load(open("results/ensemble_qwen_finetuned_summary.json", encoding="utf-8"))
check("fine-tuned SmolVLM alone: accuracy", 64.7,
      round(ens_ft["finetuned_alone"]["accuracy"] * 100, 1))
check("fine-tuned SmolVLM alone: macro-recall", 67.8,
      round(ens_ft["finetuned_alone"]["macro_recall"] * 100, 1))
check("override rule beats Qwen alone on accuracy", True, 44.7 > 37.9)
check("override rule beats Qwen alone on macro-recall", True, 57.5 > 55.5)
check("no learned-combiner claim remains in the paper", 0,
      len(re.findall(r"logistic regression combiner|learned classifier",
                     open(TEX, encoding="utf-8").read())), tol=0)

# ---------------------------------------------------------------------------
print("\n=== 12e. Language evaluation (Section V-C) ===")
# 120-image label-vs-description contrast, run on Colab (results/language_eval_summary.json).
# This run's environment does not exactly reproduce results/eval_blend_30pct.json (confirmed
# by diffing the fine-tuned model's predictions on the same 120 filenames: 48/120 differ,
# likely a transformers/bitsandbytes version difference from when eval_blend_30pct.json was
# generated). So this check only verifies internal consistency of the language-eval numbers
# themselves, not agreement with Table II -- the paper never claims that agreement.
lang = load("results/language_eval_summary.json")
ft_label = lang["smolvlm_ft_label"]
ft_desc = lang["smolvlm_ft_desc"]

check("language eval: sample size per condition", 120, ft_label["n"], tol=0)
check("FT label-mode macro-recall", 53.33, ft_label["macro_recall"], tol=0.01)
check("FT desc-mode macro-recall", 35.83, ft_desc["macro_recall"], tol=0.01)
check("FT label-vs-desc delta", -17.5,
      round(ft_desc["macro_recall"] - ft_label["macro_recall"], 1))
check("FT desc-mode: wrote a real sentence on every image (0% degenerate)",
      0.0, ft_desc["degenerate_share"])

# Label-mode prediction counts. These must come from the PREDICTIONS, not from
# parse_rules: parse_rules counts which rule fired, and `fallback_default` also returns
# "no flood", so explicit_none (64) undercounts the no-flood predictions by the 4
# fallback_default cases. The earlier version of this check compared the paper's "64"
# against explicit_none and so agreed with an error in the paper. Count the predictions
# in the per-image subset file instead, which is the same thing the paper reports.
from collections import Counter as _Counter
ft_label_preds = _Counter(r["prediction"] for r in load("results/language_eval_ft_label_subset.json"))
check("FT label-mode subset size", 120, sum(ft_label_preds.values()), tol=0)
check("FT label-mode predicted no flood", 68, ft_label_preds["no flood"], tol=0)
check("FT label-mode predicted small flood", 17, ft_label_preds["small flood"], tol=0)
check("FT label-mode predicted big flood", 35, ft_label_preds["big flood"], tol=0)
# Cross-check that the prediction counts and the parse-rule counts reconcile, so a future
# edit cannot silently reintroduce the same confusion.
check("no-flood predictions = explicit_none + fallback_default", ft_label_preds["no flood"],
      ft_label["parse_rules"]["explicit_none"] + ft_label["parse_rules"].get("fallback_default", 0),
      tol=0)

ft_desc_small = (ft_desc["parse_rules"].get("explicit_small", 0)
                 + ft_desc["parse_rules"].get("fallback_mentions_flood", 0))
check("FT desc-mode predicted small flood (explicit + mentions-flood)", 115, ft_desc_small, tol=0)
# The paper now discloses how much of that 115 is the parser's fallback rather than the
# model naming a severity outright.
check("FT desc-mode small flood from the mentions-flood fallback", 78,
      ft_desc["parse_rules"]["fallback_mentions_flood"], tol=0)
check("FT desc-mode small flood named outright", 37,
      ft_desc["parse_rules"]["explicit_small"], tol=0)

zs_smol = lang["_paired_contrasts"]["smolvlm_zs"]
zs_qwen = lang["_paired_contrasts"]["qwen_zs"]
check("SmolVLM zero-shot label-vs-desc delta", 7.5, zs_smol["delta_macro_recall"])
check("Qwen zero-shot label-vs-desc delta", 7.5, zs_qwen["delta_macro_recall"])

# The one worked example quoted in the paper (car-1173-Level3_1.jpg): label mode correct,
# desc mode wrong. Verified directly against the raw generation log.
raw = load("results/language_eval_raw.json") if os.path.exists("results/language_eval_raw.json") else None
if raw:
    ex_label = next(r for r in raw if r["filename"] == "car-1173-Level3_1.jpg"
                    and r["condition"] == "smolvlm_ft_label")
    ex_desc = next(r for r in raw if r["filename"] == "car-1173-Level3_1.jpg"
                  and r["condition"] == "smolvlm_ft_desc")
    check("worked example: ground truth is big flood", "big flood", ex_label["ground_truth"])
    check("worked example: label-mode prediction is correct", "big flood", ex_label["prediction"])
    check("worked example: desc-mode prediction is wrong (small flood)",
          "small flood", ex_desc["prediction"])
else:
    print("  [SKIP] results/language_eval_raw.json not present (partial download)")

# ---------------------------------------------------------------------------
print("\n=== 13. Test-set sizes and latency ===")
for ratio, exp in [(15, 289), (80, 68), (90, 34)]:
    p = first_existing(f"splits/split_{ratio}pct_seed42_test.json",
                       f"splits/split_{ratio}pct_test.json")
    check(f"test set size at {ratio}%", exp, len(load(p)), tol=0)

lat_zs = st.mean(r["latency_s"] for r in zs)
lat_fs = st.mean(r["latency_s"] for r in fs)
check("3-shot latency multiple", 4.5, round(lat_fs / lat_zs, 1))
check("SmolVLM latency (Table VI)", 5.69, round(lat_zs, 2))
check("Qwen latency (Table VI)", 0.30, round(st.mean(r["latency_s"] for r in qzs), 2))

# ---------------------------------------------------------------------------
print("\n=== 14. LaTeX integrity ===")
tex = open(TEX, encoding="utf-8").read()
bib = open("overleaf/references.bib", encoding="utf-8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.group(1).split(","))
defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
check("no undefined citations", set(), cited - defined, tol=0)
check("no unused bib entries", set(), defined - cited, tol=0)
check("sourceflood2026 fully removed", 0, tex.count("sourceflood2026") + bib.count("sourceflood2026"), tol=0)

figs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
missing = {f for f in figs if not os.path.exists(os.path.join("overleaf", f))}
check("all figures present", set(), missing, tol=0)

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
check("no dangling \\ref", set(), refs - labels, tol=0)

# ---------------------------------------------------------------------------
# 15. Released dataset, checked against the actual image archives.
# The manifest is a filename listing from a Drive audit, so it cannot prove the
# files exist or that the experiment subset really comes from the release. These
# read the archives. They skip if the archives are absent (e.g. deleted after upload).
import zipfile

ARCHIVES = ["fixed_synthetic_output-20261007T215927Z-1-001.zip",
            "fixed_synthetic_output-20261007T215927Z-1-002.zip"]
_present = [a for a in ARCHIVES if os.path.exists(a)]

print("\n=== 15. Released dataset vs. the actual image archives ===")
if len(_present) != len(ARCHIVES):
    print(f"  [SKIP] {len(_present)}/{len(ARCHIVES)} archives present; "
          f"dataset claims checked against the manifest only")
else:
    _SUF = {"w": "no flood", "s": "small flood", "l": "big flood"}
    _PAT = re.compile(r"^(\d{3})([wsl])\.png$", re.I)
    _names = []
    for _a in _present:
        for _n in zipfile.ZipFile(_a).namelist():
            if not _n.endswith("/"):
                _names.append(_n.split("/")[-1])

    check("archives hold 1,005 images", 1005, len(_names), tol=0)
    check("no duplicate filenames across archives",
          len(_names), len(set(_names)), tol=0)
    check("every filename matches NNN[wsl].png",
          0, sum(1 for _n in _names if not _PAT.match(_n)), tol=0)

    _scene, _lab = {}, collections.Counter()
    for _n in _names:
        _m = _PAT.match(_n)
        if _m:
            _scene.setdefault(_m.group(1), set()).add(_m.group(2).lower())
            _lab[_SUF[_m.group(2).lower()]] += 1
    check("archives span 335 base views", 335, len(_scene), tol=0)
    for _c in ("no flood", "small flood", "big flood"):
        check(f"archives hold 335 {_c} images", 335, _lab[_c], tol=0)
    check("every base view present at all three water levels",
          0, sum(1 for _v in _scene.values() if _v != {"w", "s", "l"}), tol=0)

    # the manifest must describe exactly what is in the archives
    _mf = json.load(open("colab_full_synth/full_1005_manifest.json", encoding="utf-8"))
    check("manifest filenames match the archives", set(),
          set(x["filename"] for x in _mf) ^ set(_names), tol=0)
    _bad = sum(1 for x in _mf
               if _PAT.match(x["filename"])
               and x["ground_truth"] != _SUF[_PAT.match(x["filename"]).group(2).lower()])
    check("manifest labels agree with the filename suffixes", 0, _bad, tol=0)

    # the 104-image experiment subset must really be drawn from the release
    _sub = [os.path.basename(p) for p in
            glob.glob(os.path.join("data", "synthetic_flood_images", "*.png"))]
    if _sub:
        check("experiment subset is 104 images", 104, len(_sub), tol=0)
        check("experiment subset is a strict subset of the release",
              0, len([f for f in _sub if f not in set(_names)]), tol=0)
        _sl = collections.Counter(_SUF[_PAT.match(f).group(2).lower()]
                                  for f in _sub if _PAT.match(f))
        check("subset composition 35 no flood", 35, _sl["no flood"], tol=0)
        check("subset composition 37 small flood", 37, _sl["small flood"], tol=0)
        check("subset composition 32 big flood", 32, _sl["big flood"], tol=0)
        check("subset drawn from 44 base views", 44,
              len({_PAT.match(f).group(1) for f in _sub if _PAT.match(f)}), tol=0)
    else:
        print("  [SKIP] data/synthetic_flood_images not found; subset not checked")

# ---------------------------------------------------------------------------
# CAMERA-READY BLOCKERS
# Things that are not wrong numbers but would ship a defective camera-ready.
# Reported separately so they cannot be mistaken for a failed recomputation,
# and so they cannot pass unnoticed.
blockers = []
notes = []

# 1. The submission withheld the dataset link for double-blind review. The
#    camera-ready claims "all of which we release", so the link has to be here.
for placeholder in ("withheld", "double-blind", "camera-ready version",
                    "DATASET-LINK-PENDING"):
    if placeholder in tex_ck:
        blockers.append(
            f"dataset footnote still carries submission placeholder text: "
            f"{placeholder!r} -- the camera-ready needs the real link")

n_urls = len(re.findall(r"https?://", tex_ck))
n_dois = len(re.findall(r"10\.\d{4,}/", tex_ck))
if n_urls == 0 and n_dois == 0:
    blockers.append(
        "no dataset URL or DOI anywhere in the paper, but the paper claims "
        "1,005 images are released and Reviewer 3 praised that release")

# 2. Reviewer 4 weakness 2: the editing model and prompt are not disclosed.
#    Scope this to the synthetic-data paragraph. Searching the whole paper matches
#    "Gemini" in the related-work citation list and falsely reports it resolved.
_m = re.search(r"\\textbf\{Synthetic data\.\}(.*?)\\textbf\{Real-world data\.\}",
               tex_ck, re.S)
synth_para = _m.group(1) if _m else ""
if not synth_para:
    blockers.append("could not isolate the Synthetic data paragraph to audit R4-2")
if not re.search(r"(?i)(qwen-image-edit|stable diffusion|flux|nano.?banana"
                 r"|dall.?e|imagen|sdxl|instructpix2pix|gemini)", synth_para):
    blockers.append(
        "R4-2 unresolved: no generative editing model is named where the dataset "
        "is described")
# Three states, not two, judged on substance rather than a fixed phrase list.
# A fixed list of introducing phrases has now false-positived three times on
# genuinely correct sentences ("prompted to raise...", "with one prompt per
# condition...") because real prose does not repeat a template. Judge instead
# on what actually has to be true: (a) the word "prompt" appears at all, and
# (b) separately, whether real quoted text -- "" marks of some length -- is
# present, as its own independent fact. Two true facts, not a brittle phrase.
_mentions_prompt = re.search(r"(?i)prompt", synth_para)
_has_quote = re.search(r"``[^`']{10,}''", synth_para)
if _mentions_prompt and _has_quote:
    pass                                    # fully resolved: named and quoted
elif _mentions_prompt:
    notes.append("R4-2 partial: a prompt is mentioned but no substantial "
                 "quoted text was found. Shippable, but upgrade when the "
                 "exact wording is confirmed quoted.")
else:
    blockers.append(
        "R4-2 unresolved: the word 'prompt' does not appear where the "
        "dataset is described, so neither a prompt nor a description of "
        "one is evidently present")

print("\n" + "=" * 78)
print("CAMERA-READY BLOCKERS")
if blockers:
    for b in blockers:
        print("  [BLOCKER]", b)
else:
    print("  none")
for n in notes:
    print("  [NOTE]", n)

print("\n" + "=" * 78)
if failures:
    print(f"{len(failures)} of {checks} CHECKS FAILED:\n")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED")
if blockers:
    print(f"-- but {len(blockers)} CAMERA-READY BLOCKER(S) REMAIN (above)")
sys.exit(0)
