import json, os, re, sys

RESULTS_DIR = "results"
KEYWORDS = ["split", "ijepa", "smolvlm", "qwen", "variance", "comparison", "summary"]

def is_relevant(fname):
    low = fname.lower()
    return any(k in low for k in KEYWORDS)

def summarize(obj, depth=0, max_items=5):
    """Return a short structural summary string."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return f"dict with {len(keys)} keys: {keys[:20]}"
    elif isinstance(obj, list):
        return f"list with {len(obj)} items; first item: {type(obj[0]).__name__ if obj else 'N/A'}"
    else:
        return f"{type(obj).__name__}: {obj}"

files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".json") and is_relevant(f))
print(f"Found {len(files)} relevant files out of {len(os.listdir(RESULTS_DIR))} total JSON files.\n")

for fname in files:
    path = os.path.join(RESULTS_DIR, fname)
    print("=" * 100)
    print(f"FILE: {fname}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue

    print(f"  Top-level type: {type(data).__name__}")
    print(f"  {summarize(data)}")

    # If dict, show a bit of each top-level key's value structure
    if isinstance(data, dict):
        for k, v in data.items():
            vs = summarize(v) if isinstance(v, (dict, list)) else repr(v)
            print(f"    - {k!r}: {vs}")
    print()
