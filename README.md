# Diagnosing the synthetic-to-real gap in small VLMs for deployable flood monitoring

Code, results, and verification for the IEEE CogMI 2026 paper *Diagnosing the
Synthetic-to-Real Gap in Small VLMs for Deployable Flood Monitoring*.

Kunal Chevuri, Angel Peredo, Sergei Chuprov.

> **Before making this repository public.** `overleaf/main.tex` is the accepted
> manuscript. IEEE's policy lets authors post the accepted version on a personal
> site with the IEEE copyright notice attached, but that notice is not in the
> file yet and the conference has not taken place. Either add the notice or
> remove `overleaf/` before changing visibility. `verify_paper.py` parses that
> file, so removing it disables the handful of checks that read the paper text;
> the rest still run.

## What the study does

Labeled roadway-flood photographs are scarce, so training sets get padded with
synthetic imagery. A model fitted to synthetic images often loses accuracy on
real ones. This paper measures that gap for small, edge-deployable VLMs and
locates where it comes from.

We build a synthetic flood dataset by generative editing of real CCTV frames,
then evaluate SmolVLM2-2.2B, Qwen2-VL-2B, and the self-supervised encoder
I-JEPA on three-class flood severity against 340 real photographs.

Main findings:

- Fine-tuning on synthetic data alone transfers poorly, and **each model fails
  toward a different severity class**, so the failure mode has to be measured
  per model rather than assumed.
- Blending real images in improves macro-recall at every mixture tested, but
  seed-to-seed standard deviation reaches **8.4 points**, and a recipe validated
  on one VLM does not transfer to the other.
- Freezing the pretrained vision encoder and training only a three-class
  logistic regression on its output is stronger and far steadier, peaking at
  **80.8% macro-recall**.
- A parameter-matched control separates the pretraining objective from the
  adaptation procedure. Applying the same small classifier to each VLM's *own*
  frozen vision tower recovers the advantage at all nine mixtures, so the gain
  does not come from I-JEPA's self-supervised objective. How few parameters the
  adaptation exposes to a few hundred real labels is the most likely
  explanation, though this comparison cannot isolate it.

## Verification

Every quantitative claim in the paper is recomputed from source data rather than
transcribed:

```bash
python verify_paper.py      # exit 0 if all checks pass
```

This reads `overleaf/main.tex`, extracts each number as printed, recomputes it
from the result files, and reports any disagreement. The current state is
**278 checks, all passing, no camera-ready blockers**. Run it after editing any
number in the paper.

It needs the synthetic images (see below) for the dataset-composition checks.

## Dataset

The 1,005 synthetic flood images are released separately under CC BY 4.0:

**[10.5281/zenodo.23251071](https://doi.org/10.5281/zenodo.23251071)**

`dataset_release/` holds the record's README and the label manifest. To run the
scripts that touch pixels, download the archive and extract the 104-image
experiment subset to `data/synthetic_flood_images/`.

Real photographs come from **STURM-FloodDepth**, a public dataset by its original
authors, cited in the paper and not redistributed here. Scripts expect it at
`data/sturm_flooddepth_real/upscaled_images/Level*/`.

## Layout

```
verify_paper.py                   278 checks of the paper against the data
generate_tables.py                regenerates the LaTeX tables
significance_test.py              paired tests across seeds
run_linear_probe_control.py       parameter-matched frozen-tower control
run_linear_probe_sweep.py         probe across the nine blend ratios
run_synthetic_quantity_ablation.py  varies synthetic-set size at fixed real data
run_language_eval.py              label-mode vs. free-text generation
colab_full_synth/                 the full 1,005-image sweep (GPU)
plot_*.py                         paper figures
results/                          per-image predictions and summaries
splits/                           train/test partitions, 9 ratios x 3 seeds
reference_baseline_results/       zero-shot baselines and the 104-image manifest
overleaf/                         paper source (see the notice above)
figures/                          figures as they appear in the paper
dataset_release/                  Zenodo record README and label manifest
```

## Reproducing

Most of the analysis runs on CPU from the committed results:

```bash
pip install -r requirements.txt
python verify_paper.py
python generate_tables.py
python significance_test.py
```

The probe sweeps re-fit classifiers on cached embeddings and need those caches,
which are derived artifacts rather than committed files. The fine-tuning and
embedding-extraction steps need a GPU.

## Scope

The paper states its limits and they apply here. The real benchmark is a single
dataset, and the test split shrinks as the blend ratio rises — from 289 images at
15% to 68 at 80% and 34 at 90% — which adds evaluation noise exactly where the
curves peak, so the 80% peak is where this curve happens to peak rather than a
recommended mixture. Results are scoped to encoders near the 2B parameter scale.
The three encoders run at different native input resolutions, so the residual
gap between the VLM towers and I-JEPA is not a clean test of pretraining
objective alone. The synthetic images are generative edits that vary in more
than water level, so a probe could in principle key on lighting or sky rather
than flooding.

## Citation

```bibtex
@inproceedings{chevuri2026synthreal,
  author    = {Chevuri, Kunal and Peredo, Angel and Chuprov, Sergei},
  title     = {Diagnosing the Synthetic-to-Real Gap in Small VLMs
               for Deployable Flood Monitoring},
  booktitle = {IEEE International Conference on Cognitive Machine
               Intelligence (CogMI)},
  year      = {2026}
}
```

## License

Code is MIT licensed (`LICENSE`). The synthetic dataset is CC BY 4.0 via Zenodo.
STURM-FloodDepth remains under its original authors' terms, and the manuscript
under `overleaf/` is subject to IEEE copyright.
