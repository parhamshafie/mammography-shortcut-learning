# -*- coding: utf-8 -*-
"""Section 2.6 / Fig. 3 diagnostic: relabel every image by which dataset it came from (NLBSD=0,
CBIS-DDSM=1, CMMD=2) instead of the clinical label, and see how separable the three sources are.
Not a screening model -- a probe of whether the encoder is picking up dataset-specific shortcuts.

    python run_dataset_origin_3class.py
"""
from mammo_pipeline.engine_origin import run

CONFIG = {
    "run_name": "dataset_origin_3class",
    "mode": "train",
    "smoke_test": {
        "enabled": False,
        "per_class_train": 15, "per_class_val": 5, "per_class_test": 5,
        "epochs": 2, "image_size": [256, 160],
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
        "ddsm": {"name": "ddsm", "image_root": "/path/to/cbis_ddsm_png_folder",   # ---- edit this ----
                "glob_pattern": "**/*.png", "case_id_path_index": 0},
        "cmmd": {"name": "cmmd", "image_root": "/path/to/cmmd_png_folder",         # ---- edit this ----
                "glob_pattern": "**/*.png", "case_id_path_index": 0},
    },
    "classes": ["nlbsd", "ddsm", "cmmd"],   # fixed order -- matches Fig. 3's row/column order
    "data": {
        "image_size": [1520, 912],
        "norm_mean": [0.3089279, 0.3089279, 0.3089279],
        "norm_std": [0.25053555408335154, 0.25053555408335154, 0.25053555408335154],
        "num_workers": 8, "pin_memory": True,
    },
    "split": {"test_frac": 0.2, "val_frac": 0.1, "seed": 42},
    "aggregation": {"unit": "exam", "rule": "mean_prob"},
    "backbone": {
        "mammo_clip_checkpoint": "/path/to/b5-model-best-epoch-7.tar",
        "imagenet_fallback": True,
        "head_hidden": None, "head_dropout": 0.3,
        "train_encoder": False,
        "last_blocks": None,
    },
    "train": {
        "num_epochs": 30, "batch_size": 81,     # divisible by 3 classes -> exact 27/27/27 quota
        "head_lr": 3e-5, "weight_decay": 1e-4,
        "amp": True, "grad_clip": 5.0, "seed": 42,
        "early_stopping_patience": 10,
    },
    "control": {"resume": True, "use_best_epoch": True, "select_by": "macro_auc"},
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
