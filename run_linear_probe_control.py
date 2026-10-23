"""
Tier 1b: parameter-matched linear-probe control for SmolVLM2-2.2B and Qwen2-VL-2B.

Methodology (mirrors the I-JEPA pipeline in flood_vlm_ijepa_comparison.ipynb exactly):
  - Freeze the ENTIRE model (vision tower + connector/merger + language model are
    never fine-tuned; in fact the connector/merger and language model are never even
    executed -- only the frozen vision-tower forward pass runs).
  - Extract one embedding per image by mean-pooling the vision tower's raw
    last_hidden_state over the patch-token dimension, taken BEFORE any cross-modal
    projection into the language model's embedding space:
      * SmolVLM2-2.2B: model.vision_model(pixel_values).last_hidden_state, mean over
        patches -> 1152-dim (SigLIP-SO400M tower). do_image_splitting=False so each
        image is a single 384x384 global crop (one embedding per image, consistent
        with I-JEPA's single global embedding, and ~4x cheaper on CPU than the
        default multi-tile splitting).
      * Qwen2-VL-2B: model.visual(pixel_values, grid_thw).last_hidden_state, mean
        over patches -> 1280-dim (raw ViT block output, before the PatchMerger that
        projects into the 1536-dim language-model space). Resolution capped at a
        fixed 196*28*28 px (~504x504) via min_pixels=max_pixels so every image gets
        a consistent patch count -- this is a CPU-time concession (no GPU available
        in this environment); it is applied uniformly to every image.
      * I-JEPA (existing, for reference): AutoModel last_hidden_state, mean over
        patches -> 1280-dim (ViT-H/14, 224x224).
  - Train ONLY a StandardScaler + LogisticRegression(max_iter=2000) on top of the
    frozen embeddings -- identical classifier code to the I-JEPA notebook.
  - 80% blend ratio only, 3 seeds (42/123/7), reusing the existing
    splits/split_80pct_seed{s}_{train,test}.json splits (not rebuilt).

Embeddings are cached to disk per model (resumable, saved every 20 images) so this
script can be safely re-run or interrupted without losing completed work.
"""
import glob
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ["no flood", "small flood", "big flood"]
SEEDS = [42, 123, 7]
SYNTH_DIR = "data/synthetic_flood_images"
STURM_DIR = "data/sturm_flooddepth_real/upscaled_images"


def level_to_class(level):
    return {"0": "no flood", "1": "small flood", "2": "small flood",
            "3": "big flood", "4": "big flood"}[level]


def build_local_path_index():
    idx = {}
    for fp in glob.glob(os.path.join(STURM_DIR, "Level*", "*.jpg")):
        idx[os.path.basename(fp)] = fp
    return idx


def load_synthetic_items():
    d = json.load(open("reference_baseline_results/smolvlm_results_2.2B.json", encoding="utf-8"))
    return [{"filename": r["filename"], "path": os.path.join(SYNTH_DIR, r["filename"]),
             "ground_truth": r["ground_truth"]} for r in d]


def load_real_items(local_index):
    items = {}
    for seed in SEEDS:
        for split in ["train", "test"]:
            fn = f"splits/split_80pct_seed{seed}_{split}.json"
            for r in json.load(open(fn, encoding="utf-8")):
                fname = r["filename"]
                m = re.search(r"Level(\d)", fname)
                assert m, f"Could not parse level from {fname}"
                assert level_to_class(m.group(1)) == r["ground_truth"], f"Level/class mismatch for {fname}"
                items[fname] = {"filename": fname, "path": local_index[fname], "ground_truth": r["ground_truth"]}
    return list(items.values())


# ---------------------------------------------------------------------------
# Embedding extraction (per model)
# ---------------------------------------------------------------------------

def extract_smolvlm_embeddings(items, cache_path):
    from transformers import AutoModel, AutoProcessor

    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    remaining = [it for it in items if it["filename"] not in cache]
    print(f"[SmolVLM] {len(cache)} cached, {len(remaining)} remaining")
    if not remaining:
        return cache

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModel.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.float32)
    model.eval()

    t_start = time.time()
    with torch.no_grad():
        for i, item in enumerate(remaining):
            img = Image.open(item["path"]).convert("RGB")
            inputs = processor(images=[img], text="<image>", return_tensors="pt", do_image_splitting=False)
            pv = inputs["pixel_values"].view(-1, *inputs["pixel_values"].shape[-3:])
            out = model.vision_model(pixel_values=pv)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
            cache[item["filename"]] = emb.tolist()
            if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
                json.dump(cache, open(cache_path, "w"))
                elapsed = time.time() - t_start
                rate = elapsed / (i + 1)
                eta = rate * (len(remaining) - i - 1)
                print(f"[SmolVLM] {i+1}/{len(remaining)} done, {rate:.1f}s/img, ETA {eta/60:.1f}min")
    json.dump(cache, open(cache_path, "w"))
    return cache


def extract_qwen_embeddings(items, cache_path):
    from transformers import AutoModel, AutoProcessor

    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    remaining = [it for it in items if it["filename"] not in cache]
    print(f"[Qwen] {len(cache)} cached, {len(remaining)} remaining")
    if not remaining:
        return cache

    px = 196 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=px, max_pixels=px)
    model = AutoModel.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", dtype=torch.float32)
    model.eval()

    t_start = time.time()
    with torch.no_grad():
        for i, item in enumerate(remaining):
            img = Image.open(item["path"]).convert("RGB")
            inputs = processor(images=[img], text="<|image_pad|>", return_tensors="pt")
            pv, grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
            out = model.visual(pv, grid_thw=grid_thw)
            emb = out.last_hidden_state.mean(dim=0).numpy()
            cache[item["filename"]] = emb.tolist()
            if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
                json.dump(cache, open(cache_path, "w"))
                elapsed = time.time() - t_start
                rate = elapsed / (i + 1)
                eta = rate * (len(remaining) - i - 1)
                print(f"[Qwen] {i+1}/{len(remaining)} done, {rate:.1f}s/img, ETA {eta/60:.1f}min")
    json.dump(cache, open(cache_path, "w"))
    return cache


# ---------------------------------------------------------------------------
# Classifier fitting (identical to I-JEPA notebook's run_blended_ratio)
# ---------------------------------------------------------------------------

def get_embedding_matrix(items, cache):
    X, y, kept = [], [], []
    for item in items:
        if item["filename"] in cache:
            X.append(cache[item["filename"]])
            y.append(item["ground_truth"])
            kept.append(item)
    return np.array(X), y, kept


def summarize(results):
    from collections import defaultdict
    total, correct, n_correct = defaultdict(int), defaultdict(int), 0
    for r in results:
        total[r["ground_truth"]] += 1
        if r["correct"]:
            correct[r["ground_truth"]] += 1
            n_correct += 1
    recalls = [correct[c] / total[c] * 100 if total[c] else float("nan") for c in CLASSES]
    return n_correct / len(results) * 100, sum(recalls) / 3


def run_seed(model_name, cache, synthetic_items, seed):
    train_real = json.load(open(f"splits/split_80pct_seed{seed}_train.json", encoding="utf-8"))
    test_real = json.load(open(f"splits/split_80pct_seed{seed}_test.json", encoding="utf-8"))

    X_synth, y_synth, _ = get_embedding_matrix(synthetic_items, cache)
    X_train_real, y_train_real, _ = get_embedding_matrix(train_real, cache)
    X_test, y_test, test_kept = get_embedding_matrix(test_real, cache)

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
    acc, mr = summarize(results)
    out_path = f"results/linear_probe_{model_name}_80pct_seed{seed}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"  seed {seed}: acc={acc:.2f}%  macro_recall={mr:.2f}%  (n_train={len(X_train)}, n_test={len(X_test)}) -> {out_path}")
    return acc, mr


def main():
    local_index = build_local_path_index()
    synthetic_items = load_synthetic_items()
    real_items = load_real_items(local_index)
    all_items = synthetic_items + real_items
    print(f"Total unique images needing embeddings: {len(all_items)} ({len(synthetic_items)} synthetic + {len(real_items)} real)")

    os.makedirs("embeddings_cache", exist_ok=True)

    print("\n=== Extracting SmolVLM2-2.2B frozen vision-tower embeddings ===")
    sv_cache = extract_smolvlm_embeddings(all_items, "embeddings_cache/smolvlm_embeddings_cache.json")

    print("\n=== Extracting Qwen2-VL-2B frozen vision-tower embeddings ===")
    qw_cache = extract_qwen_embeddings(all_items, "embeddings_cache/qwen_embeddings_cache.json")

    print("\n=== Training linear-probe classifiers (80% blend, 3 seeds) ===")
    summary = {}
    for model_name, cache in [("smolvlm", sv_cache), ("qwen", qw_cache)]:
        print(f"\n{model_name}:")
        accs, mrs = [], []
        for seed in SEEDS:
            acc, mr = run_seed(model_name, cache, synthetic_items, seed)
            accs.append(acc)
            mrs.append(mr)
        import statistics as st
        summary[model_name] = {
            "acc_mean": st.mean(accs), "acc_std": st.pstdev(accs),
            "mr_mean": st.mean(mrs), "mr_std": st.pstdev(mrs),
            "acc_per_seed": accs, "mr_per_seed": mrs,
        }

    json.dump(summary, open("results/linear_probe_control_summary.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    for model_name, s in summary.items():
        print(f"{model_name}: acc={s['acc_mean']:.1f}±{s['acc_std']:.1f}%  macro_recall={s['mr_mean']:.1f}±{s['mr_std']:.1f}%")
    print("\nSaved results/linear_probe_control_summary.json")


if __name__ == "__main__":
    main()
