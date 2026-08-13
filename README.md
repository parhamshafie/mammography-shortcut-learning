# Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI

Code for training and evaluating an EfficientNet-B5 (Mammo-CLIP pretrained) linear probe on NLBSD,
with and without added positives from CBIS-DDSM and CMMD, plus the dataset-origin diagnostic and the
statistical comparison behind Table 2 / Figure 2 / Figure 3.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need:
- The NLBSD manifest CSV and preprocessed PNGs (see the paper's Data Availability section for NLBSD,
  CBIS-DDSM, and CMMD access).
- The Mammo-CLIP `b5-model-best-epoch-7.tar` checkpoint.

None of the scripts ship with real paths -- every `paths` / `image_root` / `mammo_clip_checkpoint`
field is a placeholder you need to fill in before running.

## Layout

```
mammo_pipeline/       shared library -- data loading, model, losses, metrics, training loop, plots
run_baseline.py        NLBSD -> NLBSD
run_nlbsd_cmmd.py       NLBSD + CMMD+
run_nlbsd_ddsm.py       NLBSD + CBIS-DDSM+
run_nlbsd_ddsm_cmmd.py  NLBSD + CBIS-DDSM+ + CMMD+
run_dataset_origin_3class.py   dataset-of-origin diagnostic (Sec. 2.6 / Fig. 3)
run_compare_delong_holm.py     DeLong test + Holm correction across the four runs above (Sec. 2.7)
```

Each `run_*.py` is a config dict plus one function call -- edit the paths at the top, then:

```bash
python run_baseline.py              # run this first -- it sets the shared decision threshold
python run_nlbsd_cmmd.py
python run_nlbsd_ddsm.py
python run_nlbsd_ddsm_cmmd.py
python run_dataset_origin_3class.py
python run_compare_delong_holm.py   # after the four training runs above have finished
```

`run_baseline.py` has to run before the other three: it picks the decision threshold (Sec. 2.7) on
its own validation split and saves it to `baseline_threshold.json`, and the other three configs load
that file (`CONFIG["shared_threshold"]["baseline_threshold_json"]`) instead of picking their own, so
sensitivity/specificity/accuracy are compared at one common threshold rather than a separately
optimized point per config.

`CONFIG["mode"]` in each script switches between `"train"` (single split, train + evaluate), `"eval"`
(reload checkpoints, evaluate only), and `"plot"` (regenerate plots from an existing
`per_sample_predictions.csv`).

## What each run produces

Every training script writes to `<output_root>/<run_name>/`: `metrics_image.csv` / `metrics_case.csv`,
a `per_sample_predictions.csv`, training curves, and PDFs (ROC/PR, or the confusion matrix for the
origin diagnostic). `run_compare_delong_holm.py` reads four of those CSVs and writes the AUC comparison
table + forest plot.

`run_nlbsd_cmmd.py` / `run_nlbsd_ddsm.py` / `run_nlbsd_ddsm_cmmd.py` each report two rows in one run:
the fixed-NLBSD-test result (Phase A) and a mixed-domain result (Phase B), controlled by
`phase_b.external_holdout_frac` in their CONFIG.

## Notes

- CBIS-DDSM and CMMD have no normal/suspicious manifest -- both are abnormal-enriched only, so every
  image under their `image_root` is simply treated as a positive case (or, in the origin diagnostic,
  labeled by dataset). Negatives always come from NLBSD.
- Splits are patient-level throughout: a single stratified 70/10/20 train/val/test split by `case_id`
  (Sec. 2.4), not cross-validation, so no patient's images cross between train/val/test.
- AUC 95% CIs use stratified bootstrap resampling (positives and negatives resampled separately,
  2,000 replicates for the DeLong comparison, 1,000 within each training run), per Sec. 2.7.
- `mammo_pipeline/stats.py` has no torch dependency, so `run_compare_delong_holm.py` can run anywhere
  pandas/scipy/scikit-learn are installed, no GPU required.
