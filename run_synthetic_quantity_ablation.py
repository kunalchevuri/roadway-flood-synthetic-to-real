"""
Synthetic-quantity ablation.

Chuprov's review asks "at which % does synthetic data stop being helpful, or which is
the minimum necessary to actually help". The dose-response study in the paper cannot
answer that: it pins the synthetic set at all 104 images in every run and varies only
how much REAL data is blended in. This script varies the other axis.

For each encoder, real-data ratio, and seed, it refits the same StandardScaler +
LogisticRegression probe on N synthetic images (stratified by class, drawn with the
run seed) plus the real training split, and evaluates on the same held-out real test
split used everywhere else. N = 0 is the real-only control.

No new inference: every embedding is already in the caches built by
run_linear_probe_control.py, so this is pure refitting.
"""
import json
import os
import random
import statistics as st
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_linear_probe_control import (
    CLASSES, SEEDS, get_embedding_matrix, load_synthetic_items, summarize,
)
from run_linear_probe_sweep import split_path

RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
SYNTH_N = [0, 13, 26, 52, 78, 104]

CACHES = {
    "smolvlm": "embeddings_cache/smolvlm_embeddings_cache.json",
    "qwen": "embeddings_cache/qwen_embeddings_cache.json",
    "ijepa": "ijepa_embeddings/embeddings_cache.json",
}


def stratified_subset(items, n, seed):
    """Draw n synthetic images keeping the class proportions of the full 104."""
    if n == 0:
        return []
    if n >= len(items):
        return list(items)
    by_class = defaultdict(list)
    for it in items:
        by_class[it["ground_truth"]].append(it)
    for c in by_class:
        by_class[c].sort(key=lambda it: it["filename"])
    rng = random.Random(seed)
    out, total = [], len(items)
    # largest-remainder allocation so the subset sums to exactly n
    quotas = {c: len(v) * n / total for c, v in by_class.items()}
    base = {c: int(q) for c, q in quotas.items()}
    for c in sorted(base, key=lambda c: quotas[c] - base[c], reverse=True):
        if sum(base.values()) >= n:
            break
        base[c] += 1
    for c in CLASSES:
        if c in by_class:
            out.extend(rng.sample(by_class[c], base[c]))
    assert len(out) == n, (len(out), n)
    return out


def run_one(cache, synth_items, ratio, seed, n_synth):
    train_real = json.load(open(split_path(ratio, seed, "train"), encoding="utf-8"))
    test_real = json.load(open(split_path(ratio, seed, "test"), encoding="utf-8"))

    chosen = stratified_subset(synth_items, n_synth, seed)
    X_train_real, y_train_real, _ = get_embedding_matrix(train_real, cache)
    X_test, y_test, test_kept = get_embedding_matrix(test_real, cache)

    if len(X_train_real) != len(train_real) or len(X_test) != len(test_real):
        raise RuntimeError(f"missing embeddings at ratio={ratio} seed={seed}")

    if chosen:
        X_synth, y_synth, kept = get_embedding_matrix(chosen, cache)
        if len(kept) != len(chosen):
            raise RuntimeError(f"missing synthetic embeddings ({len(kept)}/{len(chosen)})")
        X_train = np.concatenate([X_synth, X_train_real], axis=0)
        y_train = y_synth + y_train_real
    else:
        X_train, y_train = X_train_real, y_train_real

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.fit_transform(X_train), y_train)
    preds = clf.predict(scaler.transform(X_test))

    results = [{"filename": it["filename"], "ground_truth": gt,
                "prediction": pred, "correct": pred == gt}
               for it, pred, gt in zip(test_kept, preds, y_test)]
    return summarize(results)


def main():
    synth_items = load_synthetic_items()
    print(f"synthetic pool: {len(synth_items)} images")
    out = {}
    for model, path in CACHES.items():
        cache = json.load(open(path, encoding="utf-8"))
        out[model] = {}
        for ratio in RATIOS:
            out[model][str(ratio)] = {}
            for n in SYNTH_N:
                accs, mrs = [], []
                for seed in SEEDS:
                    a, m = run_one(cache, synth_items, ratio, seed, n)
                    accs.append(a)
                    mrs.append(m)
                out[model][str(ratio)][str(n)] = {
                    "acc_mean": round(st.mean(accs), 2), "acc_std": round(st.pstdev(accs), 2),
                    "mr_mean": round(st.mean(mrs), 2), "mr_std": round(st.pstdev(mrs), 2),
                }
                print(f"{model:8s} ratio={ratio:2d}%  n_synth={n:3d}  "
                      f"MR={st.mean(mrs):5.1f} +/- {st.pstdev(mrs):4.1f}   "
                      f"acc={st.mean(accs):5.1f}")
            print()
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/synthetic_quantity_ablation.json", "w"), indent=2)
    print("wrote results/synthetic_quantity_ablation.json")


if __name__ == "__main__":
    main()
