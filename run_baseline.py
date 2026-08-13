# -*- coding: utf-8 -*-
"""NLBSD -> NLBSD baseline: train + evaluate on the screening cohort alone, no external data.
Single patient-level 70/10/20 train/val/test split, Mammo-CLIP EfficientNet-B5 frozen linear probe.
This is Table 2's top row.

This run is also where the shared decision threshold (Sec. 2.7) comes from -- it picks its own
Youden threshold on its validation split and saves it to baseline_threshold.json. Run this script
before the three run_nlbsd_*.py scripts, since they load that file.

    python run_baseline.py
"""
from mammo_pipeline.engine import run

CONFIG = {
    "run_name": "nlbsd_baseline",
    "mode": "train",                       # train | eval | plot
    "smoke_test": {
        "enabled": False,
        "train_pos": 15, "train_neg": 15, "val_pos": 5, "val_neg": 5, "test_pos": 5, "test_neg": 5,
        "epochs": 2, "bootstrap_n": 50, "image_size": [256, 160],
    },
    # ---- edit these ----
    "paths": {
        "manifest_csv": "/path/to/nlbs_training_manifest.csv",
        "image_root": "/path/to/nlbsd_png_folder",
        "output_root": "/path/to/outputs",
    },
    "columns": {"image_path": "File Path", "label": "Cancer", "case_id": "case_id"},
    "manifest": {
        "normalize_backslashes": True,
        "swap_dcm_to_png": True,
        "derive_case_id_from_path": True,
        "case_id_path_index": 1,
        "lefted_suffix_for_right": True,
        "laterality_col": "Image Laterality",
        "right_value": "R",
        "lefted_suffix": "_lefted",
    },
    "data": {
        "image_size": [1520, 912],
        "norm_mean": [0.3089279, 0.3089279, 0.3089279],
        "norm_std": [0.25053555408335154, 0.25053555408335154, 0.25053555408335154],
        "num_workers": 8, "pin_memory": True,
    },
    "split": {"test_frac": 0.2, "val_frac": 0.1, "seed": 42},
    "aggregation": {"unit": "exam", "rule": "max"},
    "backbone": {
        "mammo_clip_checkpoint": "/path/to/b5-model-best-epoch-7.tar",   # None -> plain ImageNet init
        "imagenet_fallback": True,
        "head_hidden": None, "head_dropout": 0.3,
        "train_encoder": False,
        "last_blocks": None,
    },
    "train": {
        "num_epochs": 30, "batch_size": 80,
        "head_lr": 3e-5, "weight_decay": 1e-4,
        "amp": True, "grad_clip": 5.0, "seed": 42,
        "early_stopping_patience": 10,
    },
    "control": {"resume": True, "use_best_epoch": True},
    "sampler": {"positives_per_batch": 20},
    "augment": {
        "rotation_deg": 10, "translate_frac": 0.05, "scale_min": 0.9, "scale_max": 1.1,
        "brightness": 0.05, "contrast_low": 0.90, "contrast_high": 1.10,
    },
    "logging": {"heartbeat_batches": 50},
    "outputs": {"per_sample_csv": True, "repro_snapshot": True, "generate_plots": True},
    "plot_from_csv": None,
}

if __name__ == "__main__":
    run(CONFIG)
