"""
Free-text language evaluation for the CogMI paper (Colab GPU required).

WHY THIS EXISTS
---------------
Chuprov's review: "One of the main attractive properties of a VLM is not only its
visual representation, but also its language capabilities ... If those capabilities
are part of the motivation for choosing VLMs, I think they should be reflected
somewhere in the evaluation."

The paper currently argues that the VLM is worth keeping for its descriptions, and
that freezing preserves that ability while fine-tuning spends it, without measuring
either. This script measures both.

WHAT IT MEASURES
----------------
For one fixed set of real images, each model is run in two modes:

  label mode     the paper's own prompt, which asks for a bare class label
  language mode  a prompt asking for a one-or-two sentence description of the road
                 conditions followed by a severity call

Three models: SmolVLM2-2.2B zero-shot, Qwen2-VL-2B zero-shot, and the best
fine-tuned SmolVLM2-2.2B (the 30% blend adapter, the same checkpoint behind the
64.7% / 67.8% row in the paper). Six conditions in total, every one on the same
120 images, so every comparison is paired.

Three things come out of it:

  1. Does the severity claim inside a free-text description agree with ground truth
     as well as the bare label does? If language mode is much worse, the sentence an
     operator reads is less trustworthy than the label the system classified with,
     which matters for the two-stage deployment argument in Section V-C.
  2. Does fine-tuning on bare label strings destroy the ability to describe? The
     fine-tuning target is the class string alone, so the prediction is that the
     fine-tuned model emits a label even when asked for a description. Measured here
     as mean word count and share of degenerate (label-only) outputs.
  3. A blinded CSV for hand-scoring hallucinated detail, which no automatic metric
     can substitute for.

HOW TO RUN (Colab, T4 is enough)
--------------------------------
  1. Runtime > Change runtime type > GPU.
  2. Upload, or mount from Drive, so that these exist:
       /content/sturm_flooddepth/upscaled_images/Level*/...   (the real images)
       /content/splits/split_30pct_test.json
       /content/adapters/blend_30pct/                          (13 MB, from adapters/)
  3. !pip install -q "transformers==4.51.0" accelerate peft bitsandbytes qwen-vl-utils
  4. !python run_language_eval.py --base-dir /content

Roughly 20-30 minutes on a T4. Every condition checkpoints after each image, so an
interrupted run resumes where it stopped: just re-run the same command.

OUTPUTS (written to results/)
-----------------------------
  language_eval_raw.json          every generation, verbatim
  language_eval_summary.json      per-condition metrics, ready for the paper
  language_eval_for_scoring.csv   blinded descriptions for manual rubric scoring
  language_eval_scoring_key.json  blind_id -> condition/filename/ground truth
"""
import argparse
import csv
import json
import os
import random
import re
import statistics as st
import time
from collections import Counter, defaultdict

CLASSES = ["no flood", "small flood", "big flood"]
SEED = 42
PER_CLASS = 40

# The paper's label prompt, copied verbatim from flood_vlm_solutions.ipynb so label
# mode here is the same measurement reported in Table II.
PROMPT_LABEL = "Answer with one label only: no flood, small flood, big flood."

# Language mode. Asks for the description first so the model cannot simply emit a
# label and stop, then asks for the severity so the same parser can score it.
PROMPT_DESC = ("Describe the road conditions in this image in one or two sentences, "
               "then state the flood severity as no flood, small flood, or big flood.")

SMOL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
QWEN_ID = "Qwen/Qwen2-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# Label parsing. Identical rules to parse_prediction() in the original notebook,
# except that it also reports WHICH rule fired. That matters here: free-text output
# hits the fallback branches far more often than a bare label does, and a language
# mode score propped up by the "unparseable -> no flood" default would be an
# artifact of the parser rather than a property of the model.
# ---------------------------------------------------------------------------
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
                                 for k, v in per.items()}}


def is_degenerate(text):
    """True when the output is a bare label rather than a description."""
    stripped = text.strip().lower().rstrip(".")
    if stripped in CLASSES:
        return True
    return len(stripped.split()) <= 3


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def build_sample(base_dir, split_path=None, img_root=None):
    """Stratified sample from the 30% blend test split.

    The split is used because the fine-tuned condition must not be scored on images
    it trained on; these 238 are its held-out set. The zero-shot conditions have no
    such constraint but use the same images so all six conditions stay paired.
    """
    split_path = split_path or os.path.join(base_dir, "splits", "split_30pct_test.json")
    items = json.load(open(split_path, encoding="utf-8"))

    by_class = defaultdict(list)
    for r in items:
        by_class[r["ground_truth"]].append(r)
    for c in by_class:
        by_class[c].sort(key=lambda r: r["filename"])

    rng = random.Random(SEED)
    sample = []
    for c in CLASSES:
        pool = by_class[c]
        if len(pool) < PER_CLASS:
            raise SystemExit(f"only {len(pool)} images of class {c}, need {PER_CLASS}")
        sample.extend(rng.sample(pool, PER_CLASS))
    sample.sort(key=lambda r: r["filename"])

    # The split records carry the Colab paths from the original run; rebuild them
    # against whatever layout this run actually uses.
    img_root = img_root or os.path.join(base_dir, "sturm_flooddepth", "upscaled_images")
    for r in sample:
        level = re.search(r"Level(\d)", r["filename"]).group(0)
        r["path"] = os.path.join(img_root, level, r["filename"])
        if not os.path.exists(r["path"]):
            raise SystemExit(f"image not found: {r['path']}\n"
                             f"check --base-dir and that the images are unpacked")
    return sample


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_ckpt(name):
    path = f"results/{name}.json"
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        print(f"  resuming {name}: {len(data)} already done")
        return data, {r["filename"] for r in data}
    return [], set()


def save_ckpt(name, data):
    os.makedirs("results", exist_ok=True)
    json.dump(data, open(f"results/{name}.json", "w", encoding="utf-8"), indent=2)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
def run_smolvlm(sample, prompt, max_new_tokens, condition, adapter=None):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    records, done = load_ckpt(condition)
    todo = [r for r in sample if r["filename"] not in done]
    if not todo:
        return records

    processor = AutoProcessor.from_pretrained(SMOL_ID)
    if adapter:
        # Same 4-bit NF4 load path used for the fine-tuned evaluations in the paper,
        # so this is the same checkpoint behaving the same way.
        from transformers import BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForImageTextToText.from_pretrained(
            SMOL_ID, quantization_config=bnb, device_map="auto")
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            SMOL_ID, dtype=torch.float16, device_map="auto")
    model.eval()
    model.config.use_cache = True

    t0 = time.time()
    for i, item in enumerate(todo):
        messages = [{"role": "user", "content": [
            {"type": "image", "path": item["path"]},
            {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", do_image_splitting=False,
        ).to(model.device, dtype=torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        text = text.split("Assistant:")[-1].strip()
        records.append(_record(item, text, condition))
        save_ckpt(condition, records)
        _progress(condition, i, len(todo), t0)

    del model
    _free()
    return records


def run_qwen(sample, prompt, max_new_tokens, condition):
    import torch
    from transformers import AutoProcessor
    try:
        # transformers >= 4.49 renamed / removed AutoModelForVision2Seq in favor of
        # the unified image-text-to-text class; try the modern name first.
        from transformers import AutoModelForImageTextToText as QwenAutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as QwenAutoModel

    records, done = load_ckpt(condition)
    todo = [r for r in sample if r["filename"] not in done]
    if not todo:
        return records

    processor = AutoProcessor.from_pretrained(
        QWEN_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = QwenAutoModel.from_pretrained(
        QWEN_ID, dtype=torch.float16, device_map="auto")
    model.eval()

    from qwen_vl_utils import process_vision_info
    t0 = time.time()
    for i, item in enumerate(todo):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": item["path"]},
            {"type": "text", "text": prompt}]}]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_in], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        text = processor.decode(trimmed, skip_special_tokens=True).strip()
        records.append(_record(item, text, condition))
        save_ckpt(condition, records)
        _progress(condition, i, len(todo), t0)

    del model
    _free()
    return records


def _record(item, text, condition):
    pred, rule = parse_prediction(text)
    return {"filename": item["filename"], "ground_truth": item["ground_truth"],
            "condition": condition, "raw_output": text,
            "prediction": pred, "parse_rule": rule,
            "word_count": len(text.split()), "degenerate": is_degenerate(text)}


def _progress(condition, i, total, t0):
    if (i + 1) % 10 == 0 or (i + 1) == total:
        rate = (time.time() - t0) / (i + 1)
        eta = rate * (total - i - 1) / 60
        print(f"  [{condition}] {i+1}/{total}  {rate:.1f}s/img  ETA {eta:.1f} min")


def _free():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(all_records):
    summary = {}
    by_cond = defaultdict(list)
    for r in all_records:
        by_cond[r["condition"]].append(r)

    for cond, recs in by_cond.items():
        m = metrics(recs)
        m["mean_word_count"] = round(st.mean(r["word_count"] for r in recs), 1)
        m["median_word_count"] = st.median(r["word_count"] for r in recs)
        m["degenerate_share"] = round(
            sum(r["degenerate"] for r in recs) / len(recs) * 100, 1)
        m["parse_rules"] = dict(Counter(r["parse_rule"] for r in recs))
        m["fallback_share"] = round(
            sum(r["parse_rule"].startswith("fallback") for r in recs) / len(recs) * 100, 1)
        summary[cond] = m

    # Paired label-vs-language contrasts, the numbers the paper would quote.
    summary["_paired_contrasts"] = {}
    for model in ["smolvlm_zs", "qwen_zs", "smolvlm_ft"]:
        lab, desc = f"{model}_label", f"{model}_desc"
        if lab in summary and desc in summary:
            summary["_paired_contrasts"][model] = {
                "macro_recall_label": summary[lab]["macro_recall"],
                "macro_recall_language": summary[desc]["macro_recall"],
                "delta_macro_recall": round(
                    summary[desc]["macro_recall"] - summary[lab]["macro_recall"], 2),
                "mean_words_label": summary[lab]["mean_word_count"],
                "mean_words_language": summary[desc]["mean_word_count"],
                "degenerate_share_language": summary[desc]["degenerate_share"],
            }
    return summary


def write_scoring_files(all_records):
    """Blinded CSV for manual rubric scoring, plus the key to unblind it afterwards.

    Only language-mode outputs are scored. Rows are shuffled and the condition is
    withheld so the scorer cannot tell which model produced which description; the
    key file restores that mapping once scoring is finished.
    """
    desc = [r for r in all_records if r["condition"].endswith("_desc")]
    rng = random.Random(SEED)
    rng.shuffle(desc)

    key = {}
    os.makedirs("results", exist_ok=True)
    with open("results/language_eval_for_scoring.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "description",
                    "water_state_correct_0_1", "severity_correct_0_1",
                    "hallucinated_detail_0_1", "notes"])
        for i, r in enumerate(desc):
            bid = f"D{i:04d}"
            key[bid] = {"condition": r["condition"], "filename": r["filename"],
                        "ground_truth": r["ground_truth"]}
            w.writerow([bid, r["raw_output"].replace("\n", " "), "", "", "", ""])

    json.dump(key, open("results/language_eval_scoring_key.json", "w",
                        encoding="utf-8"), indent=2)
    print(f"\nwrote results/language_eval_for_scoring.csv ({len(desc)} rows, blinded)")
    print("  score each row 0/1 in the three blank columns:")
    print("    water_state_correct  - does it describe the water level actually present?")
    print("    severity_correct     - does the severity it names match ground truth?")
    print("    hallucinated_detail  - does it assert anything not visible in the image?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/content",
                    help="directory holding sturm_flooddepth/, splits/, adapters/")
    ap.add_argument("--adapter", default=None,
                    help="path to the 30%% blend adapter (default: <base-dir>/adapters/blend_30pct)")
    ap.add_argument("--skip-finetuned", action="store_true",
                    help="run only the two zero-shot models")
    ap.add_argument("--split-path", default=None,
                    help="override path to split_30pct_test.json")
    ap.add_argument("--img-root", default=None,
                    help="override path to upscaled_images/ (containing Level0..Level4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the sample and exit without loading any model")
    args = ap.parse_args()

    adapter = args.adapter or os.path.join(args.base_dir, "adapters", "blend_30pct")

    sample = build_sample(args.base_dir, args.split_path, args.img_root)
    if args.dry_run:
        from collections import Counter as _C
        print(f"dry run OK: {len(sample)} images resolved, all paths exist")
        print("  class balance:", dict(_C(r["ground_truth"] for r in sample)))
        print("  first three:", [r["filename"] for r in sample[:3]])
        return
    print(f"sample: {len(sample)} images, {PER_CLASS} per class, seed {SEED}\n")

    all_records = []
    print("SmolVLM2-2.2B zero-shot")
    all_records += run_smolvlm(sample, PROMPT_LABEL, 10, "smolvlm_zs_label")
    all_records += run_smolvlm(sample, PROMPT_DESC, 96, "smolvlm_zs_desc")

    print("\nQwen2-VL-2B zero-shot")
    all_records += run_qwen(sample, PROMPT_LABEL, 10, "qwen_zs_label")
    all_records += run_qwen(sample, PROMPT_DESC, 96, "qwen_zs_desc")

    if not args.skip_finetuned:
        if not os.path.isdir(adapter):
            raise SystemExit(f"adapter not found at {adapter}; pass --adapter or --skip-finetuned")
        print("\nSmolVLM2-2.2B fine-tuned (30% blend adapter)")
        all_records += run_smolvlm(sample, PROMPT_LABEL, 10, "smolvlm_ft_label", adapter=adapter)
        all_records += run_smolvlm(sample, PROMPT_DESC, 96, "smolvlm_ft_desc", adapter=adapter)

    json.dump(all_records, open("results/language_eval_raw.json", "w",
                                encoding="utf-8"), indent=2)
    summary = summarize(all_records)
    json.dump(summary, open("results/language_eval_summary.json", "w",
                            encoding="utf-8"), indent=2)

    print("\n" + "=" * 78)
    print(f"{'condition':22s} {'n':>4} {'acc':>7} {'macroR':>8} "
          f"{'words':>7} {'degen%':>8} {'fallbk%':>8}")
    for cond in ["smolvlm_zs_label", "smolvlm_zs_desc", "qwen_zs_label",
                 "qwen_zs_desc", "smolvlm_ft_label", "smolvlm_ft_desc"]:
        if cond not in summary:
            continue
        m = summary[cond]
        print(f"{cond:22s} {m['n']:>4} {m['accuracy']:>7.1f} {m['macro_recall']:>8.1f} "
              f"{m['mean_word_count']:>7.1f} {m['degenerate_share']:>8.1f} "
              f"{m['fallback_share']:>8.1f}")

    print("\npaired label vs language contrasts:")
    for model, c in summary["_paired_contrasts"].items():
        print(f"  {model:12s} macro-recall {c['macro_recall_label']:.1f} (label) -> "
              f"{c['macro_recall_language']:.1f} (language), "
              f"delta {c['delta_macro_recall']:+.1f}; "
              f"{c['mean_words_language']:.0f} words, "
              f"{c['degenerate_share_language']:.0f}% label-only")

    write_scoring_files(all_records)
    print("\nwrote results/language_eval_summary.json")


if __name__ == "__main__":
    main()
