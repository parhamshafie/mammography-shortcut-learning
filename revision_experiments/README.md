# Revision experiments (JMI major revision)

These scripts reproduce the analyses added to the paper during the revision. Each script is standalone
and uses the same preprocessing, patient-level 70/10/20 split, training settings and evaluation as the
original experiments in this repository. Edit the data paths at the top of each script before running.

| Script | Paper section | What it does |
|---|---|---|
| `imagenet_control_experiment.py` | Sections 2.8 and 3.3, Table 4 | Repeats Experiment 1 and Experiment 2 with the same EfficientNet-B5 architecture initialized with ImageNet weights instead of Mammo-CLIP weights. Includes DeLong tests with Holm correction and bootstrap confidence intervals. |
| `run_imagenet_control.sh` | Sections 2.8 and 3.3 | Job script that runs the ImageNet control experiment on a GPU cluster (SLURM or LSF). |
| `origin_harmonized_mammoclip.py` | Sections 2.7 and 3.6 (Experiment 3) | Dataset-origin classification after histogram matching to a reference distribution built from NLBSD training images only. |
| `tsne_plot_standalone.py` | Sections 2.7 and 3.6, Figure 6 | t-SNE plot of the frozen Mammo-CLIP encoder features of 300 test images per dataset, before and after histogram matching. |
| `error_analysis.py` | Sections 2.9 and 3.4, Figure 4 | Rank-based error analysis comparing the NLBSD-only model with the NLBSD + CBIS-DDSM+ + CMMD+ model (ImageNet-initialized encoder), with the Wilcoxon signed-rank test. |

## Usage

```
python imagenet_control_experiment.py all
python origin_harmonized_mammoclip.py
python tsne_plot_standalone.py
python error_analysis.py predictions.csv --out error_analysis_out
```

`error_analysis.py` expects one CSV with one row per NLBSD test image and the columns
`key, label, prob_base, prob_mix, group, path`, where `prob_base` and `prob_mix` are the predicted cancer
probabilities of the two models.
