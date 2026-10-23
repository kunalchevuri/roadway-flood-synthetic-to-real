import json
from scipy import stats

with open("assembled_dataset.json") as f:
    data = json.load(f)

RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
SEEDS = [42, 123, 7]

def seed_values(method, ratio):
    """Return the 3 per-seed values in a fixed seed order, or None if incomplete."""
    d = data[method][str(ratio)]
    if not all(str(s) in d for s in SEEDS):
        return None
    return [d[str(s)] * 100 for s in SEEDS]

# ---------------------------------------------------------------------------
# Build comparison list: I-JEPA vs SmolVLM at all 9 ratios (both always n=3);
# I-JEPA vs Qwen only at 60-90pct, since Qwen at 15-50pct is single-seed (n=1)
# and a paired test is not meaningful/defined for an unpaired single value.
# ---------------------------------------------------------------------------
comparisons = []
for r in RATIOS:
    ijepa_vals = seed_values("ijepa", r)
    smolvlm_vals = seed_values("smolvlm", r)
    assert ijepa_vals is not None and smolvlm_vals is not None, f"unexpected gap at {r}%"
    comparisons.append(("I-JEPA vs SmolVLM", r, ijepa_vals, smolvlm_vals))

skipped_qwen_ratios = []
for r in RATIOS:
    ijepa_vals = seed_values("ijepa", r)
    qwen_vals = seed_values("qwen", r)
    if qwen_vals is None:
        skipped_qwen_ratios.append(r)
        continue
    comparisons.append(("I-JEPA vs Qwen", r, ijepa_vals, qwen_vals))

n_tests = len(comparisons)
alpha = 0.05
bonferroni_alpha = alpha / n_tests

print("=" * 92)
print(f"Skipped I-JEPA vs Qwen at {skipped_qwen_ratios} -- Qwen has only 1 seed there, "
      f"so no paired comparison is defined (nothing to pair against).")
print(f"Total comparisons run: {n_tests}  ->  Bonferroni-corrected alpha = 0.05 / {n_tests} = {bonferroni_alpha:.5f}")
print("=" * 92)

print("\nNote on test choice: with n=3 paired seeds, we use a paired t-test (scipy.stats.ttest_rel).")
print("A Wilcoxon signed-rank test was also considered, but at n=3 its EXACT two-sided p-value")
print("floor is 2*(1/2)^3 = 0.25 (only 8 possible sign patterns) -- it can never reach significance")
print("at this sample size regardless of effect size, so it would be uninformative here and is")
print("reported for completeness only, not as the primary test.")

rows = []
for label, r, a_vals, b_vals in comparisons:
    t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
    try:
        w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
    except ValueError:
        # all differences identical sign with zero variance, or a zero-difference case
        w_stat, w_p = float("nan"), float("nan")
    rows.append((label, r, t_stat, t_p, w_stat, w_p))

print("\n" + "=" * 92)
print(f"{'Ratio':>6} {'Comparison':<20}{'t-stat':>9}{'t p-value':>11}{'sig@.05':>9}{'sig@Bonf':>10}{'Wilcoxon p':>12}")
print("=" * 92)
sig_uncorrected = 0
sig_corrected = 0
for label, r, t_stat, t_p, w_stat, w_p in rows:
    sig05 = "yes" if t_p < alpha else "no"
    sigb = "yes" if t_p < bonferroni_alpha else "no"
    if t_p < alpha:
        sig_uncorrected += 1
    if t_p < bonferroni_alpha:
        sig_corrected += 1
    w_p_str = f"{w_p:.4f}" if w_p == w_p else "n/a (tied)"
    print(f"{r:>5}% {label:<20}{t_stat:>9.3f}{t_p:>11.4f}{sig05:>9}{sigb:>10}{w_p_str:>12}")

print("=" * 92)
print(f"\nSignificant at uncorrected alpha=0.05: {sig_uncorrected}/{n_tests}")
print(f"Significant at Bonferroni-corrected alpha={bonferroni_alpha:.5f}: {sig_corrected}/{n_tests}")

print("\n" + "=" * 92)
print("CAVEAT (do not present this section as strong statistical evidence on its own):")
print("=" * 92)
print("""
n=3 seeds per condition is a very small sample for any hypothesis test. A paired t-test with
df=2 has very low power, and a single seed swapped for another could plausibly flip both the
statistic and the significance call at several ratios. The comparisons above should be read as
a directional, seed-consistency check on top of the mean+-std figure already shown in the
dose-response plot, not as a substitute for a larger-n reproducibility study. We recommend
describing these results in the paper as "consistent with" rather than "statistically confirming"
a significant advantage, and reporting the exact n=3 caveat alongside any p-value cited in text.
""")
