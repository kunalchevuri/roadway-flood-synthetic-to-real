"""
Extends the parameter-matched linear-probe control (run_linear_probe_control.py)
from the single 80% blend ratio to the full 15-90% dose-response range, for both
SmolVLM2-2.2B and Qwen2-VL-2B.

No new model inference: both VLMs' frozen vision-tower embeddings for all 444
images are already cached by run_linear_probe_control.py, so this only refits the
StandardScaler + LogisticRegression head per (ratio, seed) on cached features.
The classifier code is imported unchanged from run_linear_probe_control so the
methodology is identical by construction; the 80% column is a regression check
against the previously reported numbers.
"""
import json
import os
import statistics as st

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_linear_probe_control import (
    CLASSES, SEEDS, build_local_path_index, get_embedding_matrix,
    load_synthetic_items, summarize,
)

RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]


def split_path(ratio, seed, which):
    """Seed 42 is stored unsuffixed at ratios 15-50 and explicitly at 60-90.
    (Verified: unsuffixed split_30pct == seed 42, matching eval_blend_30pct.json.)"""
    explicit = f"splits/split_{ratio}pct_seed{seed}_{which}.json"
    if os.path.exists(explicit):
        return explicit
    fallback = f"splits/split_{ratio}pct_{which}.json"
    if seed == 42 and os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"no split for ratio={ratio} seed={seed} {which}")


def run_one(cache, synthetic_items, ratio, seed):
    train_real = json.load(open(split_path(ratio, seed, "train"), encoding="utf-8"))
    test_real = json.load(open(split_path(ratio, seed, "test"), encoding="utf-8"))

    X_synth, y_synth, _ = get_embedding_matrix(synthetic_items, cache)
    X_train_real, y_train_real, _ = get_embedding_matrix(train_real, cache)
    X_test, y_test, test_kept = get_embedding_matrix(test_real, cache)

    # every split filename must be in the cache, otherwise the comparison is silently
    # fit on a subset of the data the fine-tuned runs used
    missing_tr = len(train_real) - len(X_train_real)
    missing_te = len(test_real) - len(X_test)
    if missing_tr or missing_te:
        raise RuntimeError(f"ratio={ratio} seed={seed}: {missing_tr} train / {missing_te} test embeddings missing from cache")

    X_train = np.concatenate([X_synth, X_train_real], axis=0)
    y_train = y_synth + y_train_real

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)

    results = [{"filename": it["filename"], "ground_truth": gt, "prediction": pred, "correct": pred == gt}
               for it, pred, gt in zip(test_kept, preds, y_test)]
    return summarize(results), results


def main():
    build_local_path_index()
    synthetic_items = load_synthetic_items()

    caches = {
        "smolvlm": json.load(open("embeddings_cache/smolvlm_embeddings_cache.json", encoding="utf-8")),
        "qwen": json.load(open("embeddings_cache/qwen_embeddings_cache.json", encoding="utf-8")),
    }

    out = {}
    for model_name, cache in caches.items():
        print(f"\n=== {model_name} linear probe, full dose-response ===")
        out[model_name] = {}
        for ratio in RATIOS:
            accs, mrs = [], []
            for seed in SEEDS:
                (acc, mr), results = run_one(cache, synthetic_items, ratio, seed)
                accs.append(acc)
                mrs.append(mr)
                json.dump(results, open(f"results/linear_probe_{model_name}_{ratio}pct_seed{seed}.json", "w"), indent=2)
            out[model_name][str(ratio)] = {
                "acc_mean": st.mean(accs), "acc_std": st.pstdev(accs),
                "mr_mean": st.mean(mrs), "mr_std": st.pstdev(mrs),
                "acc_per_seed": accs, "mr_per_seed": mrs,
            }
            print(f"  {ratio:2d}%: acc={st.mean(accs):5.1f}+-{st.pstdev(accs):4.1f}%  "
                  f"macro_recall={st.mean(mrs):5.1f}+-{st.pstdev(mrs):4.1f}%")

    json.dump(out, open("results/linear_probe_sweep_summary.json", "w"), indent=2)

    # regression check against the previously published 80% control
    print("\n=== 80% regression check vs. published Table III ===")
    expected = {"smolvlm": (80.9, 77.9), "qwen": (80.4, 77.8)}
    for m, (eacc, emr) in expected.items():
        g = out[m]["80"]
        ok = abs(g["acc_mean"] - eacc) < 0.1 and abs(g["mr_mean"] - emr) < 0.1
        print(f"  {m}: got acc={g['acc_mean']:.1f}% mr={g['mr_mean']:.1f}% | "
              f"expected acc={eacc}% mr={emr}% -> {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
