"""
Full-synthetic-set experiments (Colab GPU for `zeroshot` and `embed`).

WHY THIS EXISTS
---------------
A PhD student in the lab pointed out that the 114 "images" Kunal was looking at are
114 CARLA *scenes* (342 files, 3 water levels each), and that the AI-generated set is
the one to evaluate on: 1,005 images in

  https://drive.google.com/drive/folders/1U4rH2BcNEsj5DfK9swA8NTnjXTSjoO3H

The CogMI and URTC papers used 104 synthetic images, which are only **44 unique
scenes**. So the real headline is not "10x more images", it is roughly 7x more scene
diversity. Two things follow:

  1. The zero-shot synthetic baseline (reference_baseline_results/smolvlm_results_2.2B
     .json, n=104) can be re-measured on the whole set, which is what the student
     asked for.
  2. run_synthetic_quantity_ablation.py says in its own docstring that it cannot
     answer Chuprov's "at which point does synthetic data stop helping", because the
     synthetic pool is pinned at all 104 images in every run. With the full set that
     question becomes answerable.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
  - Nothing writes to results/, splits/, adapters/ or either overleaf directory.
    Every output lands in results_full/. The published numbers stay untouched, and
    the real train/test splits are opened read-only, so new numbers remain
    comparable to the ones already in print.
  - No VLM fine-tuning. Re-training the blend adapters with a bigger synthetic pool
    is 9 ratios x 3 seeds x 2 models; that is days of T4 and only worth it if
    `ablate` shows a real gain first.
  - No depth evaluation. AI-generated images have no metric depth ground truth --
    that is exactly why the CARLA subset exists. Monocular depth stays a
    CARLA-based limitations study.

THE COMPARABILITY TRAP, AND THE GATE FOR IT
-------------------------------------------
The cached embeddings in embeddings_cache/ and ijepa_embeddings/ were extracted in
float32 on CPU. If the new synthetic embeddings come out of a float16 GPU pass and
get merged into those caches, then "more synthetic data helps" is confounded with
dtype/device drift, and the effect could be pure numerics.

So `embed` re-extracts *everything* (synthetic + the 340 real images the splits
reference) in one consistent float32 pass, and `verify` then refits the published
80%-blend 3-seed probe from the fresh cache and compares against
baseline/linear_probe_{model}_80pct_seed{seed}.json. If those macro recalls do not
reproduce, the pipeline drifted and the ablation is not trustworthy -- `verify` says
so loudly instead of quietly carrying on.

SCENE-LEVEL SAMPLING
--------------------
Each scene id appears at up to three water levels (001w / 001s / 001l), so 1,005
images are NOT 1,005 independent samples. There is no test leakage -- synthetic data
is training-only and the test set is real STURM imagery -- but an x-axis counting
images would overstate how much independent data was added by roughly 3x. `ablate`
therefore samples whole *scenes* and reports both counts.

USAGE
-----
  # Step 0, no GPU, run this first and read it before anything else
  python run_full_synthetic_eval.py audit --synth-dir /content/drive/MyDrive/<folder>

  # Step 1, GPU, ~35 min on a T4
  python run_full_synthetic_eval.py zeroshot --synth-dir <dir>

  # Step 2a, GPU, ~45 min on a T4 (float32 on purpose, see above)
  python run_full_synthetic_eval.py embed --synth-dir <dir>
  python run_full_synthetic_eval.py verify

  # Step 2a, light variant: skip moving the 340 real images by copying their
  # embeddings out of the published caches. `verify` is then partly circular, so
  # `driftcheck` becomes the gate instead -- it re-embeds images that are in both the
  # folder and the published cache and checks this GPU agrees with the original CPU run.
  python run_full_synthetic_eval.py embed --synth-dir <dir> \
      --real-from-cache published_caches
  python run_full_synthetic_eval.py driftcheck --synth-dir <dir> \
      --baseline-cache-dir published_caches

  # Step 2b, CPU, minutes -- pure sklearn refitting, no GPU needed
  python run_full_synthetic_eval.py ablate

OUTPUTS (all under results_full/)
---------------------------------
  full_synth_audit.json            what is actually in the folder
  zeroshot_{smolvlm,qwen}.json     every generation, verbatim, checkpointed
  zeroshot_summary.json            per-model acc / macro recall / per-class recall
  embeddings/{model}_cache.json    fresh float32 embeddings
  verify_report.json               reproduction check against the published probe
  scene_ablation.json              macro recall vs number of synthetic scenes
"""
import argparse
import json
import os
import re
import statistics as st
import time
from collections import Counter, defaultdict

CLASSES = ["no flood", "small flood", "big flood"]
SEEDS = [42, 123, 7]
RATIOS = [15, 20, 30, 40, 50, 60, 70, 80, 90]
IMG_EXT = (".png", ".jpg", ".jpeg")
OUT = "results_full"

# Filename suffix -> class. Verified against all 104 labels in
# reference_baseline_results/smolvlm_results_2.2B.json: 35 w / 37 s / 32 l, no
# exceptions. Copied from code/VLM_preliminary_experiments.ipynb.
LABEL_MAP = {"w": "no flood", "s": "small flood", "l": "big flood"}

# Copied verbatim from run_language_eval.py so this is the same measurement as the
# label-mode row in the paper.
PROMPT_LABEL = "Answer with one label only: no flood, small flood, big flood."
SMOL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
QWEN_ID = "Qwen/Qwen2-VL-2B-Instruct"
IJEPA_ID = "facebook/ijepa_vith14_1k"


def out_path(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def scene_id(filename):
    m = re.match(r"(\d+)", os.path.basename(filename))
    return m.group(1) if m else None


def label_of(filename):
    return LABEL_MAP.get(os.path.splitext(os.path.basename(filename))[0][-1].lower())


def list_synthetic(synth_dir):
    """Every suffix-labelled image in the folder, sorted, as item dicts."""
    items, skipped = [], []
    for fn in sorted(os.listdir(synth_dir)):
        if not fn.lower().endswith(IMG_EXT):
            continue
        lab, sid = label_of(fn), scene_id(fn)
        if lab is None or sid is None:
            skipped.append(fn)
            continue
        items.append({"filename": fn, "path": os.path.join(synth_dir, fn),
                      "ground_truth": lab, "scene": sid})
    return items, skipped


# ---------------------------------------------------------------------------
# Step 0: audit
# ---------------------------------------------------------------------------
def cmd_audit(args):
    from PIL import Image

    items, skipped = list_synthetic(args.synth_dir)
    all_files = [f for f in sorted(os.listdir(args.synth_dir))
                 if f.lower().endswith(IMG_EXT)]
    scenes = defaultdict(list)
    for it in items:
        scenes[it["scene"]].append(it["ground_truth"])

    res = Counter()
    for it in items:
        try:
            res[Image.open(it["path"]).size] += 1
        except Exception as e:
            res[("unreadable", str(e)[:40])] += 1

    # Does this folder contain the 104 images the papers were built on? If not, new
    # numbers are a fresh evaluation set rather than an extension of the old one,
    # and nothing here can be plotted on the same axes as the published results.
    paper, overlap = [], []
    if os.path.exists(args.paper_manifest):
        paper = [r["filename"] for r in json.load(open(args.paper_manifest, encoding="utf-8"))]
        have = {it["filename"] for it in items}
        overlap = sorted(set(paper) & have)

    masks = {}
    if args.mask_dir and os.path.isdir(args.mask_dir):
        mf = [f for f in sorted(os.listdir(args.mask_dir)) if f.lower().endswith(IMG_EXT)]
        mask_scenes = {scene_id(f) for f in mf} - {None}
        # Masks whose scene has no image here cannot be used for Stage-1 evaluation,
        # and a mask count larger than the image-scene count means the two folders
        # were exported from different batches.
        masks = {"n_masks": len(mf), "n_scenes_with_mask": len(mask_scenes),
                 "n_mask_scenes_without_image": len(mask_scenes - set(scenes)),
                 "n_image_scenes_without_mask": len(set(scenes) - mask_scenes),
                 "by_class": dict(Counter(label_of(f.replace("_mask", "")) for f in mf))}

    report = {
        "synth_dir": args.synth_dir,
        "n_image_files": len(all_files),
        "n_labelled": len(items),
        "n_skipped_unlabelled": len(skipped),
        "skipped_examples": skipped[:20],
        "class_distribution": dict(Counter(it["ground_truth"] for it in items)),
        "n_scenes": len(scenes),
        "images_per_scene_histogram": dict(Counter(len(v) for v in scenes.values())),
        "resolutions": {str(k): v for k, v in res.most_common()},
        "paper_manifest_found": bool(paper),
        "n_paper_images": len(paper),
        "n_paper_images_present_here": len(overlap),
        "masks": masks,
    }
    json.dump(report, open(out_path("full_synth_audit.json"), "w"), indent=2)

    print(json.dumps(report, indent=2))
    print("\n" + "=" * 70)
    print(f"{len(items)} labelled images across {len(scenes)} scenes")
    if skipped:
        print(f"WARNING: {len(skipped)} files did not match the w/s/l naming "
              f"convention and were DROPPED. Check these before trusting anything:")
        for f in skipped[:10]:
            print(f"    {f}")
    if paper:
        if len(overlap) == len(paper):
            print(f"OK: all {len(paper)} published synthetic images are in this folder. "
                  f"New results extend the published ones.")
        else:
            print(f"WARNING: only {len(overlap)}/{len(paper)} published synthetic images "
                  f"are here. This is a DIFFERENT set, not a superset -- new numbers "
                  f"are a fresh evaluation, not an extension of Table II.")
    else:
        print(f"NOTE: {args.paper_manifest} not found, so overlap with the published "
              f"104 was not checked. Copy it in and re-run.")
    if masks.get("n_mask_scenes_without_image"):
        print(f"WARNING: {masks['n_mask_scenes_without_image']} scenes have a mask but "
              f"no image in this folder, so the mask and image exports do not match. "
              f"Stage-1 segmentation can only use the "
              f"{len(set(scenes)) - masks['n_image_scenes_without_mask']} scenes that "
              f"have both.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: zero-shot severity
# ---------------------------------------------------------------------------
# parse_prediction copied verbatim from run_language_eval.py. It also reports which
# rule fired, because a score propped up by the "unparseable -> no flood" default is
# a property of the parser, not the model.
def parse_prediction(raw_text):
    text = raw_text.lower().strip()
    if "big flood" in text or "large flood" in text or "severe flood" in text:
        return "big flood", "explicit_big"
    if "small flood" in text or "shallow flood" in text or "minor flood" in text:
        return "small flood", "explicit_small"
    if "no flood" in text or "no flooding" in text or "wet pavement" in text or "dry" in text:
        return "no flood", "explicit_none"
    if "flood" in text:
        return "small flood", "fallback_mentions_flood"
    return "no flood", "fallback_default"


def metrics(records):
    n = len(records)
    if n == 0:
        return None
    acc = sum(r["prediction"] == r["ground_truth"] for r in records) / n * 100
    per = {}
    for c in CLASSES:
        sub = [r for r in records if r["ground_truth"] == c]
        per[c] = (sum(r["prediction"] == c for r in sub) / len(sub) * 100) if sub else None
    vals = [v for v in per.values() if v is not None]
    return {"n": n, "accuracy": round(acc, 2),
            "macro_recall": round(sum(vals) / len(vals), 2),
            "per_class_recall": {k: (round(v, 2) if v is not None else None)
                                 for k, v in per.items()},
            "confusion": {f"{r}->{p}": c for (r, p), c in
                          sorted(Counter((x["ground_truth"], x["prediction"])
                                         for x in records).items())},
            "parse_rules": dict(Counter(r["parse_rule"] for r in records))}


def _ckpt(name):
    path = out_path(f"{name}.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        print(f"  resuming {name}: {len(data)} already done")
        return path, data, {r["filename"] for r in data}
    return path, [], set()


def _progress(tag, i, total, t0):
    if (i + 1) % 25 and (i + 1) != total:
        return
    rate = (time.time() - t0) / (i + 1)
    print(f"  [{tag}] {i+1}/{total}  {rate:.2f}s/img  ETA {rate*(total-i-1)/60:.1f}min")


def cmd_zeroshot(args):
    import torch

    models = [m for m in args.models if m != "ijepa"]
    if "ijepa" in args.models:
        print("NOTE: I-JEPA has no language head, so it cannot be prompted. Skipping it "
              "here -- it only appears in embed/ablate.")
    if not models:
        raise SystemExit("zeroshot needs smolvlm and/or qwen")

    items, _ = list_synthetic(args.synth_dir)
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} images, prompt: {PROMPT_LABEL!r}")

    summary = {}
    for model_name in models:
        path, records, done = _ckpt(f"zeroshot_{model_name}")
        todo = [it for it in items if it["filename"] not in done]
        if todo:
            runner = {"smolvlm": _run_smolvlm, "qwen": _run_qwen}[model_name]
            runner(todo, records, path, torch)
        summary[model_name] = metrics(records)
        print(f"\n{model_name}: {json.dumps(summary[model_name]['per_class_recall'])}"
              f"  macro_recall={summary[model_name]['macro_recall']}%"
              f"  acc={summary[model_name]['accuracy']}%  n={summary[model_name]['n']}")

    json.dump(summary, open(out_path("zeroshot_summary.json"), "w"), indent=2)
    print("\nwrote " + out_path("zeroshot_summary.json"))


def _save(path, records):
    json.dump(records, open(path, "w", encoding="utf-8"), indent=2)


def _run_smolvlm(todo, records, path, torch):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(SMOL_ID)
    model = load_model(AutoModelForImageTextToText, SMOL_ID, torch.float16,
                       device_map="auto")
    model.eval()
    t0 = time.time()
    for i, item in enumerate(todo):
        messages = [{"role": "user", "content": [
            {"type": "image", "path": item["path"]},
            {"type": "text", "text": PROMPT_LABEL}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", do_image_splitting=False,
        ).to(model.device, dtype=torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=10)
        text = processor.batch_decode(out, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()
        records.append(_record(item, text))
        _save(path, records)
        _progress("smolvlm", i, len(todo), t0)
    del model


def _run_qwen(todo, records, path, torch):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as QwenAutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as QwenAutoModel
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(
        QWEN_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = load_model(QwenAutoModel, QWEN_ID, torch.float16, device_map="auto")
    model.eval()
    t0 = time.time()
    for i, item in enumerate(todo):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": item["path"]},
            {"type": "text", "text": PROMPT_LABEL}]}]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_in], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=10)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        records.append(_record(item, processor.decode(trimmed, skip_special_tokens=True).strip()))
        _save(path, records)
        _progress("qwen", i, len(todo), t0)
    del model


def _record(item, text):
    pred, rule = parse_prediction(text)
    return {"filename": item["filename"], "scene": item["scene"],
            "ground_truth": item["ground_truth"], "raw_output": text,
            "prediction": pred, "parse_rule": rule,
            "correct": pred == item["ground_truth"]}


# ---------------------------------------------------------------------------
# Step 2a: embeddings (float32 on purpose -- see module docstring)
# ---------------------------------------------------------------------------
def load_real_items(args):
    """The real STURM images the published splits reference, read-only."""
    import glob
    idx = {os.path.basename(p): p
           for p in glob.glob(os.path.join(args.sturm_dir, "Level*", "*.jpg"))}
    items = {}
    for ratio in RATIOS:
        for seed in SEEDS:
            for which in ["train", "test"]:
                fp = split_path(args.splits_dir, ratio, seed, which)
                if not fp:
                    continue
                for r in json.load(open(fp, encoding="utf-8")):
                    fn = r["filename"]
                    if fn not in idx:
                        raise SystemExit(
                            f"real image missing from {args.sturm_dir}: {fn}\n"
                            f"the splits reference it, so embeddings cannot be built")
                    items[fn] = {"filename": fn, "path": idx[fn],
                                 "ground_truth": r["ground_truth"], "scene": None}
    return list(items.values())


def split_path(splits_dir, ratio, seed, which):
    """Seed 42 is stored unsuffixed at ratios 15-50 and explicitly at 60-90.
    Same resolution rule as run_linear_probe_sweep.split_path."""
    explicit = os.path.join(splits_dir, f"split_{ratio}pct_seed{seed}_{which}.json")
    if os.path.exists(explicit):
        return explicit
    fallback = os.path.join(splits_dir, f"split_{ratio}pct_{which}.json")
    if seed == 42 and os.path.exists(fallback):
        return fallback
    return None


def baseline_cache_for(model_name, baseline_dir):
    """The published cache file for this model. The originals live under two different
    directories with two different names, so match on the model name appearing
    anywhere in the basename."""
    import glob as _glob
    hits = [p for p in sorted(_glob.glob(os.path.join(baseline_dir, "*.json")))
            if model_name in os.path.basename(p).lower()]
    return hits[0] if hits else None


def cmd_embed(args):
    import numpy as np
    import torch
    from PIL import Image

    synth, _ = list_synthetic(args.synth_dir)
    if args.limit:
        # Smoke test: embed only a handful so a configuration error surfaces in seconds
        # instead of after a 45-minute run. The completeness check below uses this same
        # list, so a limited run is judged against what it was asked to do.
        synth = synth[:args.limit]
        print(f"DEBUG --limit: only the first {len(synth)} synthetic images. "
              f"This is a smoke test, NOT a usable cache.")

    # Two ways to get the real-image embeddings the probe needs.
    #
    #   default              re-extract them here, in the same float32 pass as the
    #                        synthetic ones. Strongest, because `verify` then becomes
    #                        an independent reproduction check -- but it needs the 340
    #                        real STURM images present.
    #   --real-from-cache    copy them out of the published caches. Avoids moving 340
    #                        image files around, at the cost of making `verify`
    #                        partly circular, so `driftcheck` becomes the real gate.
    if args.real_from_cache:
        real = []
        print(f"{len(synth)} synthetic images to embed; real-image embeddings will be "
              f"copied from {args.real_from_cache}")
    else:
        real = load_real_items(args)
        print(f"{len(synth) + len(real)} images to embed "
              f"({len(synth)} synthetic + {len(real)} real)")

    extractors = {"smolvlm": _emb_smolvlm, "qwen": _emb_qwen, "ijepa": _emb_ijepa}
    provenance = {}
    prov_path = out_path("embeddings", "provenance.json")
    if os.path.exists(prov_path):
        provenance = json.load(open(prov_path, encoding="utf-8"))

    for model_name in args.models:
        cache_path = out_path("embeddings", f"{model_name}_cache.json")
        cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
        prov = provenance.setdefault(model_name, {"copied": [], "fresh": []})

        if args.real_from_cache:
            src = baseline_cache_for(model_name, args.real_from_cache)
            if not src:
                raise SystemExit(f"[{model_name}] no published cache found in "
                                 f"{args.real_from_cache} -- upload it, or drop "
                                 f"--real-from-cache and supply the real images")
            old = json.load(open(src, encoding="utf-8"))
            synth_names = {it["filename"] for it in synth}
            n_new = 0
            for k, v in old.items():
                # Real images only. A synthetic filename that also appears in the
                # published cache is left alone so it gets a fresh GPU embedding and
                # can serve as a drift probe in `driftcheck`.
                if k in synth_names or k in cache:
                    continue
                cache[k] = v
                prov["copied"].append(k)
                n_new += 1
            print(f"[{model_name}] copied {n_new} real embeddings from "
                  f"{os.path.basename(src)}")

        todo = [it for it in synth + real if it["filename"] not in cache]
        print(f"[{model_name}] {len(cache)} cached, {len(todo)} to compute")
        if todo:
            extractors[model_name](todo, cache, cache_path, torch, np, Image)
            prov["fresh"].extend(it["filename"] for it in todo)
        json.dump(cache, open(cache_path, "w"))
        json.dump(provenance, open(prov_path, "w"), indent=2)

        # Say plainly whether this model is now complete. Without this, a run that died
        # partway leaves a cache that later steps happily use for a subset of the data.
        missing = [it["filename"] for it in synth if it["filename"] not in cache]
        if missing:
            raise SystemExit(
                f"\n[{model_name}] INCOMPLETE: {len(missing)} of {len(synth)} synthetic "
                f"images have no embedding.\nRe-run this same cell -- it resumes from the "
                f"cache and will finish the rest.\nFirst few missing: {missing[:5]}")
        print(f"[{model_name}] complete: {len(synth)}/{len(synth)} synthetic images "
              f"embedded, {len(cache)} entries in cache")

        # Quantify drift against the CPU float32 cache this is meant to reproduce.
        if args.baseline_cache_dir:
            _drift_report(model_name, cache, args.baseline_cache_dir, np)


def _drift_report(model_name, cache, baseline_dir, np):
    """Quantify how far the fresh embeddings sit from the original CPU float32 ones.
    Accepts any filename in baseline_dir containing the model name, since the
    originals live under two different directories with two different names
    (embeddings_cache/{model}_embeddings_cache.json, ijepa_embeddings/embeddings_cache.json)."""
    path = baseline_cache_for(model_name, baseline_dir)
    if not path:
        print(f"  [{model_name}] no baseline cache found in {baseline_dir}, drift not measured")
        return
    old = json.load(open(path, encoding="utf-8"))
    shared = [k for k in old if k in cache]
    if not shared:
        print(f"  [{model_name}] no overlapping filenames, drift not measured")
        return
    diffs, cos = [], []
    for k in shared:
        a, b = np.asarray(old[k], dtype=np.float64), np.asarray(cache[k], dtype=np.float64)
        if a.shape != b.shape:
            print(f"  [{model_name}] DIM MISMATCH on {k}: {a.shape} vs {b.shape}")
            return
        diffs.append(float(np.abs(a - b).max()))
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"  [{model_name}] drift vs {os.path.basename(path)} over {len(shared)} "
          f"shared images: max|diff|={max(diffs):.3e}  min cosine={min(cos):.6f}")


def _save_cache(cache, path, tag, i, total, t0):
    if (i + 1) % 20 and (i + 1) != total:
        return
    json.dump(cache, open(path, "w"))
    rate = (time.time() - t0) / (i + 1)
    print(f"  [{tag}] {i+1}/{total}  {rate:.2f}s/img  ETA {rate*(total-i-1)/60:.1f}min")


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cls, model_id, dtype, **kw):
    """transformers renamed the precision kwarg: <=4.55 takes `torch_dtype`, >=4.56
    takes `dtype`. run_linear_probe_control.py was written against the newer spelling
    (the local env is 5.x), but the notebooks pin 4.51.0 for the VLM-prediction track,
    where `dtype` falls through to the model constructor and raises TypeError. Accept
    both so the same script runs under either pin."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kw)


def _emb_smolvlm(todo, cache, path, torch, np, Image):
    """Frozen SigLIP-SO400M tower, mean-pooled over patches -> 1152-dim.
    do_image_splitting=False so each image is one global 384x384 crop, exactly as in
    run_linear_probe_control.extract_smolvlm_embeddings."""
    from transformers import AutoModel, AutoProcessor
    dev = _device(torch)
    processor = AutoProcessor.from_pretrained(SMOL_ID)
    model = load_model(AutoModel, SMOL_ID, torch.float32).to(dev).eval()
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(todo):
            img = Image.open(item["path"]).convert("RGB")
            inputs = processor(images=[img], text="<image>", return_tensors="pt",
                               do_image_splitting=False)
            pv = inputs["pixel_values"].view(-1, *inputs["pixel_values"].shape[-3:]).to(dev)
            out = model.vision_model(pixel_values=pv)
            cache[item["filename"]] = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().tolist()
            _save_cache(cache, path, "smolvlm", i, len(todo), t0)
    json.dump(cache, open(path, "w"))
    # Free before the next model loads. Three fp32 towers in one process exhausts a
    # T4 runtime's host RAM and the OOM killer takes the process (exit -9), which is
    # why the notebook also runs one model per subprocess.
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emb_qwen(todo, cache, path, torch, np, Image):
    """Raw Qwen2-VL ViT block output, mean-pooled -> 1280-dim, taken BEFORE the
    PatchMerger. Resolution pinned at 196*28*28 so every image yields a consistent
    patch count, as in run_linear_probe_control.extract_qwen_embeddings."""
    from transformers import AutoModel, AutoProcessor
    dev = _device(torch)
    px = 196 * 28 * 28
    processor = AutoProcessor.from_pretrained(QWEN_ID, min_pixels=px, max_pixels=px)
    model = load_model(AutoModel, QWEN_ID, torch.float32).to(dev).eval()
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(todo):
            img = Image.open(item["path"]).convert("RGB")
            inputs = processor(images=[img], text="<|image_pad|>", return_tensors="pt")
            out = model.visual(inputs["pixel_values"].to(dev),
                               grid_thw=inputs["image_grid_thw"].to(dev))
            cache[item["filename"]] = out.last_hidden_state.mean(dim=0).cpu().numpy().tolist()
            _save_cache(cache, path, "qwen", i, len(todo), t0)
    json.dump(cache, open(path, "w"))
    # Free before the next model loads. Three fp32 towers in one process exhausts a
    # T4 runtime's host RAM and the OOM killer takes the process (exit -9), which is
    # why the notebook also runs one model per subprocess.
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emb_ijepa(todo, cache, path, torch, np, Image):
    """I-JEPA ViT-H/14 last_hidden_state, mean-pooled -> 1280-dim, 224x224."""
    from transformers import AutoModel, AutoProcessor
    dev = _device(torch)
    processor = AutoProcessor.from_pretrained(IJEPA_ID)
    model = load_model(AutoModel, IJEPA_ID, torch.float32).to(dev).eval()
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(todo):
            img = Image.open(item["path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(dev)
            out = model(**inputs)
            cache[item["filename"]] = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().tolist()
            _save_cache(cache, path, "ijepa", i, len(todo), t0)
    json.dump(cache, open(path, "w"))
    # Free before the next model loads. Three fp32 towers in one process exhausts a
    # T4 runtime's host RAM and the OOM killer takes the process (exit -9), which is
    # why the notebook also runs one model per subprocess.
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared probe fitting
# ---------------------------------------------------------------------------
def get_matrix(items, cache, np):
    X, y, kept = [], [], []
    for it in items:
        if it["filename"] in cache:
            X.append(cache[it["filename"]])
            y.append(it["ground_truth"])
            kept.append(it)
    return np.array(X), y, kept


def summarize(results):
    total, correct, n_correct = defaultdict(int), defaultdict(int), 0
    for r in results:
        total[r["ground_truth"]] += 1
        if r["correct"]:
            correct[r["ground_truth"]] += 1
            n_correct += 1
    recalls = [correct[c] / total[c] * 100 if total[c] else float("nan") for c in CLASSES]
    return n_correct / len(results) * 100, sum(recalls) / 3


def fit_probe(X_train, y_train, X_test, y_test, test_kept):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.fit_transform(X_train), y_train)
    preds = clf.predict(scaler.transform(X_test))
    results = [{"filename": it["filename"], "ground_truth": gt,
                "prediction": p, "correct": p == gt}
               for it, p, gt in zip(test_kept, preds, y_test)]
    acc, mr = summarize(results)
    return acc, mr, results


# ---------------------------------------------------------------------------
# driftcheck: is this GPU's float32 pass numerically the same as the original
# CPU float32 pass? This is the gate when `embed --real-from-cache` was used, since
# `verify` is then partly circular.
# ---------------------------------------------------------------------------
def cmd_driftcheck(args):
    import numpy as np
    import torch
    from PIL import Image

    if not args.baseline_cache_dir:
        raise SystemExit("driftcheck needs --baseline-cache-dir")
    synth, _ = list_synthetic(args.synth_dir)
    extractors = {"smolvlm": _emb_smolvlm, "qwen": _emb_qwen, "ijepa": _emb_ijepa}

    report, verdict = {}, "PASS"
    for model_name in args.models:
        src = baseline_cache_for(model_name, args.baseline_cache_dir)
        if not src:
            print(f"[{model_name}] no published cache in {args.baseline_cache_dir}, skipping")
            continue
        old = json.load(open(src, encoding="utf-8"))
        probes = [it for it in synth if it["filename"] in old][:args.n_probe]
        if not probes:
            report[model_name] = {"n_probe": 0, "note":
                "no image in this folder is also in the published cache, so drift "
                "could not be measured at all"}
            verdict = "UNMEASURED"
            print(f"[{model_name}] NO OVERLAP -- drift unmeasurable")
            continue

        fresh = {}
        extractors[model_name](probes, fresh, out_path("embeddings", f"_drift_{model_name}.json"),
                               torch, np, Image)
        diffs, cos = [], []
        for it in probes:
            a = np.asarray(old[it["filename"]], dtype=np.float64)
            b = np.asarray(fresh[it["filename"]], dtype=np.float64)
            if a.shape != b.shape:
                raise SystemExit(f"[{model_name}] dim mismatch on {it['filename']}: "
                                 f"{a.shape} vs {b.shape} -- wrong cache or wrong model")
            diffs.append(float(np.abs(a - b).max()))
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        rel = max(diffs) / max(1e-12, float(np.abs(np.asarray(
            old[probes[0]["filename"]], dtype=np.float64)).max()))
        ok = min(cos) >= args.min_cosine
        verdict = verdict if ok else "FAIL"
        report[model_name] = {"n_probe": len(probes),
                              "max_abs_diff": max(diffs),
                              "max_abs_diff_relative_to_scale": rel,
                              "min_cosine": min(cos),
                              "threshold_min_cosine": args.min_cosine,
                              "ok": ok}
        print(f"[{model_name}] {len(probes)} probe images: max|diff|={max(diffs):.3e}  "
              f"min cosine={min(cos):.8f}  -> {'ok' if ok else 'FAIL'}")

    report["verdict"] = verdict
    json.dump(report, open(out_path("driftcheck_report.json"), "w"), indent=2)
    print("\n" + "=" * 70)
    if verdict == "PASS":
        print(f"PASS: this GPU's float32 pass matches the original CPU float32 pass.\n"
              f"Embeddings copied from the published cache can be mixed with fresh ones\n"
              f"without confounding the ablation.")
    elif verdict == "UNMEASURED":
        print("UNMEASURED: none of the images in this folder are in the published cache,\n"
              "so drift could not be checked. Either accept that the ablation stands on\n"
              "its own and is NOT comparable to the published numbers, or re-run\n"
              "`embed` without --real-from-cache so `verify` becomes a real gate.")
    else:
        print("FAIL: this GPU's embeddings differ from the published ones by more than\n"
              "the cosine threshold. Do NOT mix cached and fresh embeddings -- re-run\n"
              "`embed` without --real-from-cache so everything comes from one pass.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# verify: do the fresh embeddings reproduce the published probe numbers?
# ---------------------------------------------------------------------------
def cmd_verify(args):
    import numpy as np

    paper = json.load(open(args.paper_manifest, encoding="utf-8"))
    synth_items = [{"filename": r["filename"], "ground_truth": r["ground_truth"],
                    "scene": scene_id(r["filename"])} for r in paper]

    report, worst, n_compared, worst_flips = {}, 0.0, 0, 0
    for model_name in args.models:
        cache_path = out_path("embeddings", f"{model_name}_cache.json")
        if not os.path.exists(cache_path):
            print(f"[{model_name}] no fresh cache, skipping")
            continue
        cache = json.load(open(cache_path, encoding="utf-8"))
        report[model_name] = {}
        for seed in SEEDS:
            base = os.path.join(args.baseline_dir,
                                f"linear_probe_{model_name}_80pct_seed{seed}.json")
            if not os.path.exists(base):
                continue
            published_acc, published_mr = summarize(
                json.load(open(base, encoding="utf-8")))

            train_real = json.load(open(split_path(args.splits_dir, 80, seed, "train"), encoding="utf-8"))
            test_real = json.load(open(split_path(args.splits_dir, 80, seed, "test"), encoding="utf-8"))
            Xs, ys, _ = get_matrix(synth_items, cache, np)
            Xtr, ytr, _ = get_matrix(train_real, cache, np)
            Xte, yte, kept = get_matrix(test_real, cache, np)
            if len(Xs) != len(synth_items) or len(Xte) != len(test_real):
                raise SystemExit(f"[{model_name}] cache is missing images "
                                 f"(synth {len(Xs)}/{len(synth_items)}, "
                                 f"test {len(Xte)}/{len(test_real)}) -- run embed first")
            acc, mr, refit_results = fit_probe(
                np.concatenate([Xs, Xtr]), ys + ytr, Xte, yte, kept)
            published = {r["filename"]: r["prediction"]
                         for r in json.load(open(base, encoding="utf-8"))}
            n_flipped = sum(1 for r in refit_results
                            if published.get(r["filename"]) != r["prediction"])
            worst_flips = max(worst_flips, n_flipped)

            d_mr = abs(mr - published_mr)
            worst = max(worst, d_mr)
            n_compared += 1
            report[model_name][str(seed)] = {
                "published_macro_recall": round(published_mr, 2),
                "refit_macro_recall": round(mr, 2),
                "delta_macro_recall": round(mr - published_mr, 2),
                "published_accuracy": round(published_acc, 2),
                "refit_accuracy": round(acc, 2),
                "n_test_images_changing_prediction": n_flipped,
            }
            print(f"[{model_name}] seed {seed}: published MR={published_mr:.2f}%  "
                  f"refit MR={mr:.2f}%  delta={mr - published_mr:+.2f}pp  "
                  f"({n_flipped}/{len(refit_results)} images changed)")

    # How much of this check was independent? If `embed --real-from-cache` supplied
    # the real embeddings, and the published 104 synthetic embeddings came along with
    # them, then this is refitting the published probe on the published embeddings and
    # it MUST reproduce -- which proves nothing about the new GPU pass. Say so rather
    # than letting a green PASS imply more than it does.
    prov_path = out_path("embeddings", "provenance.json")
    copied = set()
    if os.path.exists(prov_path):
        for per_model in json.load(open(prov_path, encoding="utf-8")).values():
            copied |= set(per_model.get("copied", []))
    paper_names = {r["filename"] for r in paper}
    n_copied_paper = len(paper_names & copied)
    independent = n_copied_paper == 0

    report["worst_abs_delta_macro_recall_pp"] = round(worst, 3)
    report["tolerance_pp"] = args.tol
    report["n_comparisons"] = n_compared
    report["n_real_embeddings_copied_from_published_cache"] = len(copied)
    report["n_paper_synthetic_embeddings_copied"] = n_copied_paper
    report["is_independent_reproduction"] = independent
    # A gate that reports PASS when nothing ran is worse than no gate: `worst` starts at
    # 0.0, so an empty comparison loop would otherwise sail through the tolerance test.
    #
    # Gate on flipped predictions, not percentage points. Macro-recall is quantised: on
    # the 68-image test split one image is worth 0.90-2.22pp depending on its class, so
    # a pp threshold cannot tell "the pipeline is wrong" apart from "one borderline
    # image fell the other way on different hardware". Counting images can.
    report["worst_images_changing_prediction"] = worst_flips
    report["max_allowed_flips"] = args.max_flips
    report["verdict"] = ("NO DATA" if n_compared == 0
                         else "PASS" if worst_flips <= args.max_flips else "FAIL")
    json.dump(report, open(out_path("verify_report.json"), "w"), indent=2)

    if n_compared == 0:
        raise SystemExit(
            "\n" + "=" * 70 + "\n"
            "NO DATA: zero comparisons ran, so nothing was verified. This is NOT a pass.\n"
            "Either the embedding caches are missing (run `embed` first and check it\n"
            "finished) or baseline/linear_probe_*_80pct_seed*.json was not unpacked.\n"
            + "=" * 70)

    print("\n" + "=" * 70)
    if worst_flips <= args.max_flips:
        print(f"PASS: at most {worst_flips} test image(s) changed prediction "
              f"(limit {args.max_flips}); worst macro-recall shift {worst:.2f}pp.")
        if independent and not copied:
            print("This is an INDEPENDENT reproduction: every embedding was computed by\n"
                  "this run, so the scene ablation is comparable to the published numbers.")
        else:
            print(f"CAUTION: {len(copied)} embeddings were copied from the published cache\n"
                  f"rather than computed here, so this PASS is partly circular and does\n"
                  f"NOT by itself show the new GPU pass agrees. `driftcheck` is the real\n"
                  f"gate in that setup -- run it and require PASS.")
    else:
        print(f"FAIL: {worst_flips} test images changed prediction (limit "
              f"{args.max_flips}); worst macro-recall shift {worst:.2f}pp.\n"
              f"The embedding pipeline does NOT reproduce the published probe. Do not\n"
              f"report ablation numbers against the published ones until this is\n"
              f"explained -- any 'more synthetic helps' effect could be this drift.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 2b: scene-level synthetic-quantity ablation
# ---------------------------------------------------------------------------
def scene_subset(items, n_scenes, seed):
    """Draw n_scenes whole scenes. Sampling whole scenes rather than images keeps the
    x-axis in units of independent scenes, and keeps a scene's three water levels
    from being split across the in/out boundary."""
    import random
    by_scene = defaultdict(list)
    for it in items:
        by_scene[it["scene"]].append(it)
    ids = sorted(by_scene)
    if n_scenes <= 0:
        return []
    chosen = ids if n_scenes >= len(ids) else random.Random(seed).sample(ids, n_scenes)
    out = []
    for s in sorted(chosen):
        out.extend(sorted(by_scene[s], key=lambda it: it["filename"]))
    return out


def image_subset(items, n, seed):
    """Draw n synthetic IMAGES keeping the class proportions of the pool.

    Copied verbatim from run_synthetic_quantity_ablation.stratified_subset, so the
    `--sample-by images` mode is the paper's Table VI protocol run on a bigger pool
    rather than a new design. Note this differs from scene_subset: it can take one
    water level of a scene and leave the other two, which is what the paper did."""
    import random
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


def cmd_ablate(args):
    import numpy as np

    manifest_path = out_path("embeddings", "synthetic_manifest.json")
    if os.path.exists(manifest_path):
        synth = json.load(open(manifest_path, encoding="utf-8"))
    elif args.synth_dir:
        synth, _ = list_synthetic(args.synth_dir)
        json.dump(synth, open(manifest_path, "w"), indent=2)
    else:
        raise SystemExit("need --synth-dir once so the synthetic manifest can be built")

    n_scenes_total = len({it["scene"] for it in synth})
    by_images = args.sample_by == "images"
    if by_images:
        # The paper's Table VI protocol: draw IMAGES stratified by class. 104 is its
        # exact setting, so that point is directly comparable to what is in print.
        cap = len(synth)
        grid = sorted({n for n in args.image_grid if n <= cap} | {cap})
        unit = "images"
    else:
        cap = n_scenes_total
        grid = sorted({n for n in args.scene_grid if n <= cap} | {cap})
        unit = "scenes"
    print(f"synthetic pool: {len(synth)} images / {n_scenes_total} scenes")
    print(f"sampling by {unit}; grid: {grid}")
    if by_images:
        print("  (104 images = the published Table VI setting, directly comparable)")
    else:
        print("  NOTE: sampling whole scenes is NOT the paper's protocol. The paper drew\n"
              "  104 images stratified by class from 44 scenes, taking only 1 or 2 water\n"
              "  levels from 18 of them; 44 scenes here means all 132 of their images.\n"
              "  Use --sample-by images for the directly comparable curve.")

    missing_caches = [m for m in args.models
                      if not os.path.exists(out_path("embeddings", f"{m}_cache.json"))]
    if len(missing_caches) == len(args.models):
        raise SystemExit(
            "\n" + "=" * 70 + "\n"
            f"NO EMBEDDINGS: none of {args.models} has a cache in "
            f"{out_path('embeddings')}.\n"
            "`embed` did not finish. Scroll up to its cell, read the error, and re-run\n"
            "it before this step. Writing an empty ablation would look like a result.\n"
            + "=" * 70)
    if missing_caches:
        print(f"WARNING: no cache for {missing_caches}; continuing with the rest")

    out = {}
    for model_name in args.models:
        cache_path = out_path("embeddings", f"{model_name}_cache.json")
        if not os.path.exists(cache_path):
            continue
        cache = json.load(open(cache_path, encoding="utf-8"))
        out[model_name] = {}
        for ratio in args.ratios:
            out[model_name][str(ratio)] = {}
            for n in grid:
                accs, mrs, n_imgs = [], [], []
                for seed in SEEDS:
                    tr = json.load(open(split_path(args.splits_dir, ratio, seed, "train"), encoding="utf-8"))
                    te = json.load(open(split_path(args.splits_dir, ratio, seed, "test"), encoding="utf-8"))
                    chosen = (image_subset(synth, n, seed) if by_images
                              else scene_subset(synth, n, seed))
                    Xtr, ytr, _ = get_matrix(tr, cache, np)
                    Xte, yte, kept = get_matrix(te, cache, np)
                    if len(Xte) != len(te) or len(Xtr) != len(tr):
                        raise SystemExit(f"missing real embeddings at ratio={ratio} seed={seed}")
                    if chosen:
                        Xs, ys, kept_s = get_matrix(chosen, cache, np)
                        if len(kept_s) != len(chosen):
                            raise SystemExit(f"missing {len(chosen)-len(kept_s)} synthetic "
                                             f"embeddings -- run embed first")
                        Xtr, ytr = np.concatenate([Xs, Xtr]), ys + ytr
                    a, m, _ = fit_probe(Xtr, ytr, Xte, yte, kept)
                    accs.append(a)
                    mrs.append(m)
                    n_imgs.append(len(chosen))
                out[model_name][str(ratio)][str(n)] = {
                    "sample_by": args.sample_by, "grid_point": n,
                    "n_scenes": len({it["scene"] for it in scene_subset(synth, n, seed)})
                                if not by_images else None,
                    "n_synth_images": n_imgs[0],
                    "acc_mean": round(st.mean(accs), 2), "acc_std": round(st.pstdev(accs), 2),
                    "mr_mean": round(st.mean(mrs), 2), "mr_std": round(st.pstdev(mrs), 2),
                    "mr_per_seed": [round(v, 2) for v in mrs],
                }
                print(f"{model_name:8s} ratio={ratio:2d}%  scenes={n:4d} "
                      f"imgs={n_imgs[0]:4d}  MR={st.mean(mrs):5.1f} +/- {st.pstdev(mrs):4.1f}"
                      f"  acc={st.mean(accs):5.1f}")
            print()
    name = "scene_ablation.json" if not by_images else "image_ablation.json"
    json.dump(out, open(out_path(name), "w"), indent=2)
    print("wrote " + out_path(name))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["audit", "zeroshot", "embed", "driftcheck",
                                        "verify", "ablate"])
    ap.add_argument("--synth-dir", help="folder of AI-generated synthetic images")
    ap.add_argument("--mask-dir", help="ground-truth mask folder (audit only)")
    ap.add_argument("--sturm-dir", default="sturm_flooddepth/upscaled_images")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--baseline-dir", default="baseline",
                    help="published linear_probe_*_80pct_seed*.json, read-only")
    ap.add_argument("--baseline-cache-dir",
                    help="original CPU float32 embedding caches, for drift measurement")
    ap.add_argument("--real-from-cache", metavar="DIR",
                    help="take the real-image embeddings from the published caches in "
                         "DIR instead of recomputing them, so the 340 real images do "
                         "not have to be moved. Makes `verify` partly circular -- run "
                         "`driftcheck` as the gate instead.")
    ap.add_argument("--n-probe", type=int, default=40,
                    help="driftcheck: how many overlapping images to re-embed")
    ap.add_argument("--min-cosine", type=float, default=0.999999,
                    help="driftcheck: minimum cosine between fresh and published "
                         "embeddings before the run is called drifted")
    ap.add_argument("--paper-manifest", default="paper_104_manifest.json")
    ap.add_argument("--models", nargs="+", default=["smolvlm", "qwen"],
                    choices=["smolvlm", "qwen", "ijepa"])
    ap.add_argument("--ratios", nargs="+", type=int, default=[30, 50, 80])
    ap.add_argument("--scene-grid", nargs="+", type=int,
                    default=[0, 11, 22, 44, 88, 150, 220, 300])
    ap.add_argument("--sample-by", choices=["scenes", "images"], default="scenes",
                    help="scenes: draw whole scenes, so the x-axis counts independent "
                         "views. images: the paper's Table VI protocol, drawing images "
                         "stratified by class, which makes 104 directly comparable.")
    ap.add_argument("--image-grid", nargs="+", type=int,
                    default=[0, 13, 26, 52, 78, 104, 250, 500, 750],
                    help="--sample-by images: the paper's grid, extended past 104")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="verify: macro-recall tolerance, reported but no longer gating")
    ap.add_argument("--max-flips", type=int, default=2,
                    help="verify: how many test images may change prediction before the "
                         "run is called a failure. Cross-device float differences flip a "
                         "borderline prediction or two; a pipeline error flips many.")
    ap.add_argument("--limit", type=int, help="debug: only the first N images")
    args = ap.parse_args()

    if args.command in ("audit", "zeroshot", "embed", "driftcheck") and not args.synth_dir:
        raise SystemExit(f"{args.command} needs --synth-dir")
    os.makedirs(OUT, exist_ok=True)
    {"audit": cmd_audit, "zeroshot": cmd_zeroshot, "embed": cmd_embed,
     "driftcheck": cmd_driftcheck, "verify": cmd_verify,
     "ablate": cmd_ablate}[args.command](args)


if __name__ == "__main__":
    main()
