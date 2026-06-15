# Submission README

This repository contains code, report drafts, lightweight results, and presentation preparation material for Project 4: Open-Vocabulary Object Detection and Visual Grounding.

## Recommended Blackboard Files

- Final report PDF: `report/report_draft_zh.pdf`
- Code zip: create from this repository root after replacing group information:

```bash
cd ..
zip -r groupid_code.zip project4_ovd \
  -x "project4_ovd/data/coco_subset*" \
  -x "project4_ovd/outputs/*" \
  -x "project4_ovd/.git/*" \
  -x "project4_ovd/__pycache__/*" \
  -x "project4_ovd/**/__pycache__/*"
```

- Supplementary material: optional. If included, use the report figures and failure cases already summarized under `report/figures/` and `results/`.

## Repository Contents

- `scripts/infer.py`: single-image, URL, and batch inference with Grounding DINO.
- `scripts/prepare_coco_subset.py`: COCO val2017 subset builder.
- `scripts/eval_coco.py`: COCO-style detection evaluation and threshold comparison.
- `scripts/export_failure_cases.py`: false positive / false negative visualization.
- `src/ovd/`: Grounding DINO wrapper and visualization utilities.
- `configs/`: COCO 20-class prompts and custom demo prompt.
- `report/`: Chinese report draft, LaTeX source, PDF, and embedded figures.
- `results/100_image_experiment/`: lightweight metric summaries used in the report.
- `presentation/`: presentation outline and Q&A preparation.
- `tests/`: lightweight post-processing and evaluation utility tests.

## Data Policy

Downloaded COCO images, annotations, model caches, and full inference visualizations are intentionally not committed. They are reproducible from the commands in `README.md`.

## Minimum Reproduction Commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Prepare a 100-image subset:

```bash
python scripts/prepare_coco_subset.py \
  --max-images 100 \
  --download-images \
  --output-dir data/coco_subset_100
```

Run the best reported inference setting:

```bash
python scripts/infer.py \
  --image-dir data/coco_subset_100/images \
  --prompt-file data/coco_subset_100/prompts/coco_20_classes.txt \
  --model-id IDEA-Research/grounding-dino-tiny \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --output-dir outputs/coco_subset_100_tiny_perclass_nms_t025
```

Evaluate:

```bash
python scripts/eval_coco.py \
  --annotations data/coco_subset_100/annotations/instances_val2017_subset.json \
  --pred-dir outputs/coco_subset_100_tiny_perclass_nms_t025 \
  --output-dir outputs/eval_100_tiny_perclass_nms_t025 \
  --experiment-name tiny_perclass_nms_t025
```
