# Project 4: Open-Vocabulary Detection

This repository implements Project 4 from the Computer Vision 2026 final project: **Open-Vocabulary Object Detection and Visual Grounding**.

It covers the three required technical components:

1. Reproduce a Grounding DINO open-vocabulary detection pipeline.
2. Prepare a COCO val2017 subset for later evaluation.
3. Evaluate predictions on the COCO subset with COCO-style metrics.

The implementation uses HuggingFace `transformers` instead of compiling the original GroundingDINO repository. This is more reliable with the current Python 3.13 environment.

## Grading Policy Coverage

- **Report**: see `report/report_draft_zh.pdf`. It follows the provided template and includes introduction, related work, method, experiments, conclusion, references, and member contribution placeholders.
- **Grounding DINO paper**: see `papers/grounding_dino_arxiv_2303.05499.pdf`.
- **Current pipeline explanation**: see `docs/pipeline_zh.md`.
- **Experimental results**: see `results/100_image_experiment/` for lightweight CSV/JSON summaries and `report/figures/` for report-ready figures.
- **Presentation**: see `presentation/presentation_outline_zh.md` for a 15-minute Chinese presentation outline and likely Q&A points.
- **Supplementary/code submission README**: see `SUBMISSION_README.md`.
- **Topic bonus**: this is Topic 4, so it is eligible for the frontier-topic bonus if the implementation quality is accepted.

## Reproduction Scope

This project does not train Grounding DINO from scratch. The model backbone, pretrained weights, HuggingFace processor, and Grounding DINO post-processing API are reused from the open-source model interface. The project contribution is the complete open-vocabulary detection experiment pipeline around that model: prompt normalization, single-image and batch inference, per-class prompt strategy, NMS and score cutoff, COCO subset preparation, COCOeval evaluation, per-class AP, threshold comparison, and failure-case visualization.

## Main 100-Image Results

The main experiment uses 100 COCO val2017 subset images, 20 open-vocabulary categories, and Grounding DINO tiny.

| Method | AP | AP50 | AP75 | AR100 | Valid predictions | Unmatched predictions |
|---|---:|---:|---:|---:|---:|---:|
| multi-class prompt | 0.1970 | 0.2681 | 0.2081 | 0.3142 | 816 | 118 |
| per-class prompt + NMS | 0.2355 | 0.2832 | 0.2669 | 0.5229 | 2711 | 0 |
| per-class + NMS + score cutoff 0.30 | 0.2272 | 0.2705 | 0.2572 | 0.4741 | 1878 | 0 |
| per-class + NMS + score cutoff 0.35 | 0.2194 | 0.2578 | 0.2484 | 0.4309 | 1334 | 0 |
| per-class + NMS + score cutoff 0.40 | 0.2116 | 0.2464 | 0.2398 | 0.3766 | 918 | 0 |

The best current setting is `per-class prompt + NMS` with thresholds `box=0.25`, `text=0.25`, and `nms_iou=0.5`. It improves AP from `0.1970` to `0.2355` and AR100 from `0.3142` to `0.5229`, mainly by reducing mixed text labels and missed detections.

## Environment

```bash
cd ComputerVisionProject  # or the repository root after cloning
python -m pip install -r requirements.txt
```

The default model is `IDEA-Research/grounding-dino-base`. If download time or memory is a concern, use `--model-id IDEA-Research/grounding-dino-tiny`.

## Single Image Demo

```bash
python scripts/infer.py \
  --image-url http://images.cocodataset.org/val2017/000000039769.jpg \
  --prompt "a cat. a remote control." \
  --output-dir outputs/demo
```

Each image produces:

- `<stem>_ovd.jpg`: visualization with boxes, labels, and scores
- `<stem>.json`: structured result with `boxes_xyxy`, `scores`, and `text_labels`

## Prepare COCO Subset

```bash
python scripts/prepare_coco_subset.py \
  --max-images 500 \
  --download-images \
  --output-dir data/coco_subset
```

Outputs:

- `data/coco_subset/annotations/instances_val2017_subset.json`
- `data/coco_subset/metadata/subset_manifest.csv`
- `data/coco_subset/prompts/coco_20_classes.txt`
- `data/coco_subset/images/` when `--download-images` is enabled

## Batch Inference on the COCO Subset

```bash
python scripts/infer.py \
  --image-dir data/coco_subset/images \
  --prompt-file data/coco_subset/prompts/coco_20_classes.txt \
  --output-dir outputs/coco_subset
```

To run one class at a time and suppress duplicate boxes with NMS:

```bash
python scripts/infer.py \
  --image-dir data/coco_subset/images \
  --prompt-file data/coco_subset/prompts/coco_20_classes.txt \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --output-dir outputs/coco_subset_per_class_nms
```

To reproduce the score-cutoff variants used in the report:

```bash
python scripts/infer.py \
  --image-dir data/coco_subset_100/images \
  --prompt-file data/coco_subset_100/prompts/coco_20_classes.txt \
  --model-id IDEA-Research/grounding-dino-tiny \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --post-score-cutoff 0.30 \
  --output-dir outputs/coco_subset_100_tiny_perclass_nms_t025_score030
```

## COCO Evaluation

```bash
python scripts/eval_coco.py \
  --annotations data/coco_subset/annotations/instances_val2017_subset.json \
  --pred-dir outputs/coco_subset \
  --output-dir outputs/eval_coco \
  --experiment-name grounding_dino_base_coco20
```

Outputs:

- `outputs/eval_coco/predictions_coco.json`: predictions converted to COCO detection format
- `outputs/eval_coco/metrics_summary.json`: AP, AP50, AP75, AR, thresholds, and prediction counts
- `outputs/eval_coco/metrics_summary.csv`: one-row summary table for reports
- `outputs/eval_coco/per_class_ap.csv`: AP/AP50/AP75 for each category
- `outputs/eval_coco/per_class_ap.png`: report-ready per-class AP bar chart
- `outputs/eval_coco/unmatched_predictions.csv`: predictions skipped because labels did not match COCO categories

## Threshold Comparison

Run inference into separate directories with different thresholds, then evaluate each directory:

```bash
python scripts/infer.py \
  --image-dir data/coco_subset/images \
  --prompt-file data/coco_subset/prompts/coco_20_classes.txt \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --output-dir outputs/coco_subset_t025

python scripts/eval_coco.py \
  --annotations data/coco_subset/annotations/instances_val2017_subset.json \
  --pred-dir outputs/coco_subset_t025 \
  --output-dir outputs/eval_t025 \
  --experiment-name t025
```

Compare multiple runs:

```bash
python scripts/eval_coco.py \
  --compare outputs/eval_t025/metrics_summary.json outputs/eval_t035/metrics_summary.json outputs/eval_t045/metrics_summary.json \
  --output-dir outputs/eval_compare
```

## Failure Case Visualization

```bash
python scripts/export_failure_cases.py \
  --annotations data/coco_subset_100/annotations/instances_val2017_subset.json \
  --predictions outputs/eval_100_tiny_perclass_nms_t025/predictions_coco.json \
  --image-dir data/coco_subset_100/images \
  --output-dir outputs/failure_cases_100_perclass \
  --top-k 12
```

The exported images use green for ground truth, blue for true positives, red for false positives, and orange for false negatives.

## Quick Checks

Run lightweight utility tests:

```bash
python -m unittest discover -s tests
```

Check CUDA and key imports:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection"
```

## Custom Image Demo

Put self-collected classroom, campus, or lab images in `data/custom_demo/images/`, then run:

```bash
python scripts/infer.py \
  --image-dir data/custom_demo/images \
  --prompt-file configs/custom_demo_prompt.txt \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --output-dir outputs/custom_demo
```

## Notes

- Prompts are normalized to lower-case English phrases separated by periods.
- Bare labels such as `person` are converted to `a person`.
- Evaluation maps predicted labels to COCO categories with exact matching after removing `a`, `an`, or `the`.
- Visual grounding metrics such as RefCOCO accuracy are not included in this COCO detection evaluation.
- Downloaded COCO images and full inference outputs are intentionally ignored by git. Use the commands above to regenerate them. Lightweight result summaries and report figures are committed under `results/` and `report/figures/`.

References:

- HuggingFace Grounding DINO docs: https://huggingface.co/docs/transformers/model_doc/grounding-dino
- Grounding DINO base model: https://huggingface.co/IDEA-Research/grounding-dino-base
