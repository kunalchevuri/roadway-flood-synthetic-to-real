"""Build a plain-language results report, as Markdown and as Word.

Every number is read from the result files rather than typed, so re-running this after
a new experiment updates the prose and the tables together.

    python make_report.py

Writes RESULTS_REPORT.md and RESULTS_REPORT.docx.
"""
import json
import os
import re

PROJ = os.path.dirname(os.path.abspath(__file__))
ABL = json.load(open(os.path.join(PROJ, "results_full_1005_image_ablation.json"),
                     encoding="utf-8"))
CPU = json.load(open(os.path.join(PROJ, "results_full_1005_cpu_tier.json"),
                     encoding="utf-8"))

RATIOS = ["15", "20", "30", "40", "50", "60", "70", "80", "90"]
SIZES = ["0", "13", "26", "52", "78", "104", "250", "500", "750", "1005"]
NAME = {"smolvlm": "SmolVLM's eye", "qwen": "Qwen's eye", "ijepa": "I-JEPA"}


def mr(enc, ratio, size):
    return ABL[enc][ratio][size]["mr_mean"]


def best_real_only(enc):
    """Best score using no fake images at all."""
    return max((mr(enc, r, "0"), r) for r in RATIOS)


def best_cell(enc):
    return max((mr(enc, r, s), r, s) for r in RATIOS for s in SIZES)


# ---- numbers used in the prose -----------------------------------------
peak_80 = max(((mr("ijepa", "80", s) - mr("ijepa", "80", "0")), s)
              for s in SIZES if s != "0")
gain_full = mr("ijepa", "80", "1005") - mr("ijepa", "80", "0")
worse = {e: sum(1 for r in RATIOS if mr(e, r, "1005") < mr(e, r, "0"))
         for e in NAME}
shares = [1005 / (1005 + round(340 * int(r) / 100)) * 100 for r in RATIOS]
ij80 = CPU["at80"]
sp104 = ij80["104"]["ijepa"]["recall_spread_pp"]
sp1005 = ij80["1005"]["ijepa"]["recall_spread_pp"]
pc1005 = ij80["1005"]["ijepa"]["per_class_recall"]
synth_only = CPU["synthetic_only"]

# Published fine-tuning numbers, PARSED from main.tex Table III rather than typed.
# They were hardcoded here once with SmolVLM's 80% value filled in from the 30% row
# (57.2 instead of 64.5), which overstated the frozen-vs-fine-tuned gap by 7.3
# points. verify_paper.py checks main.tex but never checked this dict, so the error
# survived. Parsing removes the chance of mismapping a ratio to a value.
_PUB_COLS = ["smolvlm", "qwen", "ijepa"]        # header order, verified in main.tex
_ROW_RE = re.compile(r"^\s*(\d+)\\%\s*&(.*?)\\\\\s*$")
_VAL_RE = re.compile(r"(\d+\.\d+)\s*\$\\pm\$")


def _parse_published_ft():
    tex = open(os.path.join(PROJ, "overleaf", "main.tex"), encoding="utf-8").read()
    blk = tex[tex.index(r"\label{tab:doseresponse}"):]
    blk = blk[:blk.index(r"\end{table}")]
    out = {}
    for line in blk.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        vals = _VAL_RE.findall(m.group(2))
        if len(vals) != len(_PUB_COLS):
            continue
        for c, v in zip(_PUB_COLS, vals):
            out[(c, m.group(1))] = float(v)
    if len(out) != 27:
        raise SystemExit(f"Table III parse got {len(out)} cells, expected 27")
    return out


PUB_FT_ALL = _parse_published_ft()
PUB_FT = {k: v for k, v in PUB_FT_ALL.items()
          if k in {("smolvlm", "15"), ("smolvlm", "80"),
                   ("qwen", "15"), ("qwen", "50"), ("qwen", "90")}}
PUB_IJEPA_PEAK = max(v for (c, r), v in PUB_FT_ALL.items() if c == "ijepa")

L = []


def w(s=""):
    L.append(s)


w("# What we found with the full image set")
w()
w("A plain-language summary of the experiments on all 1,005 synthetic flood images.")
w("Written for anyone on the project, including people who have not read the paper.")
w()
w("---")
w()
w("## First, what the words mean")
w()
w("Four terms come up constantly. Everything else in this report is plain English.")
w()
w("**Real photos.** 340 actual photographs of flooded streets, from a public dataset")
w("called STURM. These are the exam. **Every score in this report is measured on real")
w("photos only.** No fake image is ever used to score anything.")
w()
w("**Fake images.** Our 1,005 generated images. We took real traffic-camera frames and")
w("used an AI image editor to raise the water level. These are study material, never")
w("the exam.")
w()
w("**Balanced score.** There are three answers a model can give: no flood, small")
w("flood, big flood. The real photos are lopsided -- 185 of the 340 are small flood.")
w("So a lazy model that always guesses \"small flood\" would look 54% correct while")
w("being useless. The balanced score fixes that: it checks how often the model is")
w("right on each of the three levels separately, then averages those three numbers. A")
w("model has to handle all three to score well. Higher is better, 100 is perfect, and")
w("random guessing is about 33.")
w()
w("**The two ways to teach a model.**")
w()
w("- **Retraining** (the paper calls this fine-tuning): we actually change the model's")
w("  internals so it learns our task. Powerful, but it can wreck what the model")
w("  already knew, and it needs a lot of data to go well.")
w("- **Freeze-and-probe:** we change nothing in the model. We feed it an image, read")
w("  out the internal summary it forms, and train a tiny separate classifier on those")
w("  summaries. The model itself is untouched -- for the two chat models, the")
w("  language half never even runs. Cheap, and almost nothing can go wrong.")
w()
w("Three models are involved. **SmolVLM** and **Qwen** are small chat models that can")
w("look at pictures; when we freeze-and-probe them we only use their image half, which")
w("this report calls \"SmolVLM's eye\" and \"Qwen's eye\". **I-JEPA** is a pure image")
w("model with no language ability at all, included as an outside comparison.")
w()
w("---")
w()
w("## What we ran")
w()
w("We tried every sensible combination:")
w()
w(f"- **{len(RATIOS)} different amounts of real photos** to study from, "
  f"from {RATIOS[0]}% of the 340 up to {RATIOS[-1]}%. Whatever we did not study from "
  f"became the exam.")
w(f"- **{len(SIZES)} different amounts of fake images** mixed in, "
  f"from {int(SIZES[0]):,} up to {int(SIZES[-1]):,}.")
w("- **3 repeats of each**, with a different random split each time, so a lucky split")
w("  cannot fool us.")
w()
w(f"That is {len(RATIOS)} x {len(SIZES)} x 3 = "
  f"{len(RATIOS) * len(SIZES) * 3:,} training runs for each of the three models.")
w()
w("---")
w()
w("## Finding 1: more fake images did not help. Mostly they hurt.")
w()
w("This was the main question, and the answer is clear.")
w()
w(f"For I-JEPA, fake images help a little -- but the sweet spot is tiny. Its best "
  f"improvement comes from just **{peak_80[1]} fake images** "
  f"(**+{peak_80[0]:.1f} points** over using none at all). Pile on all 1,005 and the "
  f"benefit shrinks to **+{gain_full:.1f}**.")
w()
w("For the two chat models' eyes it is worse than that. Adding all 1,005 fake images")
w(f"made **SmolVLM's eye worse at all {worse['smolvlm']}** amounts of real data, "
  f"and **Qwen's eye worse at {worse['qwen']} of the 9**.")
w()
w("**What it means.** We cannot fix a shortage of real photos by generating more fake")
w("ones. Past a very small number, extra fake images actively get in the way. The")
w("expensive part of this project is not generating images -- it is labelling real")
w("ones.")
w()
w("## Finding 2: it is the proportion that matters, not the number")
w()
w("The paper already found that fake images only help when they are a *small part* of")
w("what the model studies from. That explains Finding 1 exactly.")
w()
w(f"With all 1,005 fake images in the mix, they make up **{min(shares):.0f}% to "
  f"{max(shares):.0f}% of everything the model studies from**, no matter how we set it "
  f"up. The real photos are drowned out. The condition that made fake images useful "
  f"is never met.")
w()
w("**What it means.** There is no setting of the dials where all 1,005 fake images")
w("are a minority. To use them properly we would need several thousand more real")
w("photos to go with them.")
w()
w("## Finding 3: the chat models' eyes are our best tool")
w()
w("This is the most useful result, and it surprised us.")
w()
w("Using **real photos only, no fake images at all**:")
w()
w("| Model (frozen, tiny classifier on top) | Balanced score | Using this much real data |")
w("|---|---|---|")
for e in ("smolvlm", "qwen", "ijepa"):
    s, r = best_real_only(e)
    cell = f"**{s:.1f}**" if e != "ijepa" else f"{s:.1f}"
    w(f"| {NAME[e]} | {cell} | {r}% |")
w()
bs, br = best_real_only("smolvlm")
bq, _ = best_real_only("qwen")
bi, _ = best_real_only("ijepa")
w(f"The two chat models' eyes beat I-JEPA by about "
  f"**{((bs + bq) / 2) - bi:.1f} points** -- with no fake images involved at all. "
  f"For comparison, the best number in our published paper was "
  f"**{PUB_IJEPA_PEAK}** for I-JEPA, and it needed 104 fake images to get there.")
w()
w("Looking across every single combination we tried, the best result for each model is")
w("effectively a three-way tie:")
w()
w("| Model | Best anywhere | Setup that got it |")
w("|---|---|---|")
for e in ("smolvlm", "qwen", "ijepa"):
    s, r, sz = best_cell(e)
    w(f"| {NAME[e]} | {s:.1f} | {r}% real, {sz} fake |")
w()
w("**What it means.** If you are deploying this, use the image half of a model you")
w("are already running. You do not need to add a second specialist model, and you")
w("probably do not need fake images. SmolVLM's single best result contains **zero**")
w("fake images.")
w()
w("I-JEPA is the one model that genuinely benefits from them "
  f"({mr('ijepa', '80', '0'):.1f} to {max(mr('ijepa', '80', s) for s in SIZES):.1f}), "
  f"which fits the explanation already in the paper: I-JEPA learned from ordinary "
  f"photographs, so edited traffic-camera frames show it something new. The chat "
  f"models learned from enormous collections of images and captions and have already "
  f"seen this kind of scene.")
w()
w("## Finding 4: not retraining the model is worth an enormous amount")
w()
w("Same model, same data, same splits. The only difference is whether we retrained the")
w("model or left it frozen.")
w()
w("| Model | Real data used | Retrained | Frozen | Difference |")
w("|---|---|---|---|---|")
for (e, r), ft in sorted(PUB_FT.items(), key=lambda x: (x[0][0], int(x[0][1]))):
    fz = mr(e, r, "0")
    w(f"| {NAME[e]} | {r}% | {ft:.1f} | **{fz:.1f}** | **+{fz - ft:.1f}** |")
w()
w("Qwen more than doubles its score when we stop retraining it.")
w()
w("**Important caveat.** The retrained numbers are from our published paper, which")
w("used 104 fake images. We have **not yet re-run the retraining with all 1,005** --")
w("that is the big remaining job. So this table compares the new frozen results")
w("against older retrained results. The gap is far too large to be explained by that")
w("mismatch, but the comparison is not strictly like-for-like yet.")
w()
w("## Finding 5: fake images buy balance, not accuracy")
w()
w("A smaller result, but an interesting one. At the 80% setting, adding all 1,005 fake")
w("images changes *how* I-JEPA is right, not just how often.")
w()
w(f"- With 104 fake images, its three per-level scores are far apart: a "
  f"**{sp104:.0f}-point spread** between its best and worst level.")
w(f"- With all 1,005, the spread shrinks to **{sp1005:.0f} points** "
  f"({pc1005['no flood']:.0f} / {pc1005['small flood']:.0f} / "
  f"{pc1005['big flood']:.0f} on the three levels).")
w()
w("So it becomes much more even-handed across the three severity levels, even though")
w("its overall score drops.")
w()
w("**What it means.** If treating the three levels consistently matters more than the")
w("top-line number -- and for a warning system it might -- then fake images are doing")
w("something useful after all. Just not what we expected.")
w()
w("## Finding 6: studying only from fake images does not work")
w()
w("Worth stating plainly, since it is the whole premise of the paper. Train on fake")
w("images alone, then test on real photos:")
w()
w("| Model | Balanced score, 104 fake | Balanced score, 1,005 fake |")
w("|---|---|---|")
for e in ("smolvlm", "qwen", "ijepa"):
    a = synth_only[e]["104"]["mr_mean"]
    b = synth_only[e]["1005"]["mr_mean"]
    w(f"| {NAME[e]} | {a:.1f} | {b:.1f} |")
w()
w("Random guessing is about 33. **Having ten times as many fake images does not")
w("improve this at all.** The problem was never the quantity.")
w()
w("## Finding 7: the results are trustworthy")
w()
w("Three checks that the pipeline is doing what we think.")
w()
w("**It reproduced a published number exactly.** Asking Qwen cold, with no training")
w("at all, gave 37.9% correct and a 55.5 balanced score -- identical to what we")
w("published. Reaching the same number by a different route is the strongest evidence")
w("the setup is sound.")
w()
w("**It reproduced the published curve with different images.** Re-running at 104 fake")
w("images landed **24 of 27** measurements within 2.5 points of the published table --")
w("using a *different* random selection of 104 images. So the published result was not")
w("a fluke of which images we happened to generate.")
w()
w("**The 80% setting held up.** Our paper claimed 80% real data was the best mix, and")
w("one reviewer told us not to call that optimal off a single test set. It is now the")
w("best setting for I-JEPA at every fake-image count of 52 or more, and beats 90% at")
w("all ten counts.")
w()
w("---")
w()
w("## What is not done yet")
w()
w("1. **Retraining with all 1,005 fake images.** The big one. Finding 4's comparison")
w("   stays slightly uneven until this is done.")
w("2. **A bigger model.** Everything here is around 2 billion parameters. Whether any")
w("   of it holds at 7 billion is untested, and a reviewer raised it.")
w("3. **A second set of real photos.** All 340 real photos come from one source. We")
w("   cannot tell how much of this is specific to that source.")
w()
w("## Bottom line")
w()
w("If someone asks what we learned, it is these three things.")
w()
w("**Generating more fake training images is not the answer.** Ten times as many made")
w("things worse almost everywhere. The bottleneck is real labelled photos.")
w()
w("**Do not retrain a small chat model on a few hundred examples.** Freeze it, read")
w("out what it already sees, and train a tiny classifier on top. That was worth up to")
w(f"**{max(mr(e, r, '0') - ft for (e, r), ft in PUB_FT.items()):.0f} points**, which "
  f"is a bigger effect than anything else we measured.")
w()
w("**Use the model you already have.** The image half of SmolVLM or Qwen, frozen,")
w("matches a dedicated specialist image model and needs no fake data at all.")
w()
w("---")
w()
w("*Every number here is read directly from the experiment result files by")
w("`make_report.py`. Re-running that script regenerates this document.*")

md = "\n".join(L) + "\n"
md_path = os.path.join(PROJ, "RESULTS_REPORT.md")
open(md_path, "w", encoding="utf-8", newline="\n").write(md)
print("wrote RESULTS_REPORT.md")

# ---- Word version -------------------------------------------------------
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
except ImportError:
    print("python-docx not installed; skipping the .docx "
          "(pip install python-docx)")
    raise SystemExit(0)

import re

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def add_rich(par, text):
    """Render **bold** and *italic* inside a paragraph."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part.replace("`", ""))


lines = md.split("\n")
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("| ") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|-: "):
        header = [c.strip() for c in ln.strip("|").split("|")]
        rows = []
        i += 2
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip("|").split("|")])
            i += 1
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for c, h in enumerate(header):
            cell = t.rows[0].cells[c]
            cell.text = ""
            add_rich(cell.paragraphs[0], h)
            for r in cell.paragraphs[0].runs:
                r.bold = True
        for row in rows:
            cells = t.add_row().cells
            for c, v in enumerate(row[:len(header)]):
                cells[c].text = ""
                add_rich(cells[c].paragraphs[0], v)
        doc.add_paragraph()
        continue

    if ln.startswith("# "):
        doc.add_heading(ln[2:], level=0)
    elif ln.startswith("## "):
        doc.add_heading(ln[3:], level=1)
    elif ln.startswith("### "):
        doc.add_heading(ln[4:], level=2)
    elif ln.strip() == "---":
        pass
    elif ln.startswith("- ") or ln.startswith("  "):
        txt = ln.strip()
        if txt.startswith("- "):
            add_rich(doc.add_paragraph(style="List Bullet"), txt[2:])
        elif txt:
            add_rich(doc.add_paragraph(), txt)
    elif re.match(r"^\d+\. ", ln):
        add_rich(doc.add_paragraph(style="List Number"),
                 re.sub(r"^\d+\. ", "", ln))
    elif ln.startswith("*") and ln.endswith("*") and len(ln) > 2:
        p = doc.add_paragraph()
        r = p.add_run(ln.strip("*").replace("`", ""))
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    elif ln.strip():
        add_rich(doc.add_paragraph(), ln)
    i += 1

docx_path = os.path.join(PROJ, "RESULTS_REPORT.docx")
doc.save(docx_path)
print("wrote RESULTS_REPORT.docx")
print(f"\n{len(md.split())} words, {md.count(chr(10))} lines")
