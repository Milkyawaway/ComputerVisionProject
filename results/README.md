# Results Summary

This directory stores lightweight result files that are small enough to commit and are used directly by the report.

Full inference JSON files, visualizations, downloaded COCO images, and generated evaluation folders are ignored by git. Recreate them with the commands in the main `README.md`.

## 100 Image Experiment

Files under `100_image_experiment/` summarize the main COCO val2017 20-class subset experiment:

- `baseline_metrics_summary.csv/json`: multi-class prompt baseline.
- `perclass_nms_metrics_summary.csv/json`: per-class prompt + per-label NMS.
- `perclass_nms_per_class_ap.csv`: AP/AP50/AP75 for each of the 20 classes.
- `threshold_comparison.csv`: baseline, per-class prompt + NMS, and post-score-cutoff variants.

The report figures generated from these runs are stored under `report/figures/`.
