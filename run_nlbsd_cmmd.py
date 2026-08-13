# -*- coding: utf-8 -*-
"""NLBSD + CMMD+: adds CMMD as extra training positives. CMMD has no normal/suspicious manifest, so
every image under external_datasets.cmmd.image_root is just treated as positive.

Produces both Table 2 rows for this config in one run: the fixed-NLBSD-test result (Phase A) and the
NLBSD-u-CMMD mixed-domain result (Phase B), via phase_b.external_holdout_frac. Set phase_b.enabled to
False to use 100% of CMMD for training and skip Phase B.

Reuses the decision threshold saved by run_baseline.py (Sec. 2.7) -- run that script first.

    python run_nlbsd_cmmd.py
"""
from mammo_pipeline.engine import run

CONFIG = {
    "run_name": "nlbsd_plus_cmmd",
    "mode": "train",
    "smoke_test": {
        "enabled": False,
        "train_pos": 15, "train_neg": 15, "val_pos": 5, "val_neg": 5, "test_pos": 5, "test_neg": 5,
        "epochs": 2, "bootstrap_n": 50, "image_size": [256, 160],
        "ext_train_pos": 15, "ext_holdout_pos": 5,
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
    "external_datasets": {
        "cmmd": {
            "enabled": True, "name": "cmmd",
            "image_root": "/path/to/cmmd_png_folder",       # ---- edit this ----
            "glob_pattern": "**/*.png",
            "case_id_path_index": 0,
        },
    },
    "phase_b": {"enabled": True, "external_holdout_frac": 0.2, "seed": 123},
    "data": {
        "image_size": [1520, 912],
        "norm_mean": [0.3089279, 0.3089279, 0.3089279],
        "norm_std": [0.25053555408335154, 0.25053555408335154, 0.25053555408335154],
        "num_workers": 8, "pin_memory": True,
    },
    "split": {"test_frac": 0.2, "val_frac": 0.1, "seed": 42},
    "shared_threshold": {
        "baseline_threshold_json": "/path/to/outputs/nlbsd_baseline/baseline_threshold.json",  # ---- edit this ----
    },
    "aggregation": {"unit": "exam", "rule": "max"},
    "backbone": {
        "mammo_clip_checkpoint": "/path/to/b5-model-best-epoch-7.tar",
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
