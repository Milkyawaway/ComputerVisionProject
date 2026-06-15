#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import numpy as np


ARTICLE_RE = re.compile(r"^(a|an|the)\s+")
METRIC_KEYS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO predictions on a COCO-format subset.")
    parser.add_argument("--annotations", type=Path, help="COCO-format ground-truth annotation JSON.")
    parser.add_argument("--pred-dir", type=Path, help="Directory containing JSON files produced by infer.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_coco"))
    parser.add_argument("--experiment-name", help="Name written to the metrics summary.")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        help="Compare multiple metrics_summary.json files and generate threshold_comparison outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare_summaries(args.compare, args.output_dir)
        return

    if args.annotations is None or args.pred_dir is None:
        raise SystemExit("--annotations and --pred-dir are required unless --compare is used.")

    run_evaluation(
        annotations_path=args.annotations,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name or args.pred_dir.name,
    )


def run_evaluation(
    annotations_path: Path,
    pred_dir: Path,
    output_dir: Path,
    experiment_name: str,
) -> None:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit("pycocotools is required. Run: python -m pip install -r requirements.txt") from exc

    if not annotations_path.exists():
        raise FileNotFoundError(annotations_path)
    if not pred_dir.exists():
        raise FileNotFoundError(pred_dir)

    annotations = load_json(annotations_path)
    image_stem_to_id = build_image_stem_map(annotations)
    category_name_to_id = {
        normalize_label(category["name"]): int(category["id"]) for category in annotations["categories"]
    }
    category_id_to_name = {int(category["id"]): category["name"] for category in annotations["categories"]}

    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        raise ValueError(f"No prediction JSON files found in {pred_dir}")

    coco_predictions, unmatched_rows, metadata_rows = convert_predictions(
        pred_files=pred_files,
        image_stem_to_id=image_stem_to_id,
        category_name_to_id=category_name_to_id,
    )

    predictions_path = output_dir / "predictions_coco.json"
    unmatched_path = output_dir / "unmatched_predictions.csv"
    summary_json_path = output_dir / "metrics_summary.json"
    summary_csv_path = output_dir / "metrics_summary.csv"
    per_class_csv_path = output_dir / "per_class_ap.csv"
    per_class_plot_path = output_dir / "per_class_ap.png"

    write_json(predictions_path, coco_predictions)
    write_unmatched_csv(unmatched_path, unmatched_rows)

    if not coco_predictions:
        raise ValueError(
            "No valid predictions could be mapped to COCO categories. "
            f"See {unmatched_path} for skipped predictions."
        )

    coco_gt = COCO(str(annotations_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(int(image["id"]) for image in annotations["images"])
    coco_eval.params.catIds = sorted(category_id_to_name)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    per_class_rows = per_class_ap(coco_eval, coco_gt, category_id_to_name)
    write_csv(per_class_csv_path, per_class_rows)
    plot_per_class_ap(per_class_rows, per_class_plot_path)

    summary = build_summary(
        coco_eval=coco_eval,
        experiment_name=experiment_name,
        annotations=annotations,
        pred_dir=pred_dir,
        metadata_rows=metadata_rows,
        valid_predictions=len(coco_predictions),
        unmatched_predictions=len(unmatched_rows),
    )
    write_json(summary_json_path, summary)
    write_csv(summary_csv_path, [summary])

    print(f"Valid predictions: {len(coco_predictions)}")
    print(f"Unmatched predictions: {len(unmatched_rows)}")
    print(f"Metrics summary: {summary_json_path.resolve()}")
    print(f"Per-class AP: {per_class_csv_path.resolve()}")
    print(f"COCO predictions: {predictions_path.resolve()}")


def convert_predictions(
    pred_files: list[Path],
    image_stem_to_id: dict[str, int],
    category_name_to_id: dict[str, int],
) -> tuple[list[dict], list[dict], list[dict]]:
    coco_predictions = []
    unmatched_rows = []
    metadata_rows = []

    for pred_file in pred_files:
        payload = load_json(pred_file)
        metadata_rows.append(
            {
                "model_id": payload.get("model_id"),
                "box_threshold": payload.get("box_threshold"),
                "text_threshold": payload.get("text_threshold"),
                "inference_mode": payload.get("inference_mode"),
                "nms_iou_threshold": payload.get("nms_iou_threshold"),
                "post_score_cutoff": payload.get("post_score_cutoff"),
                "prompt": payload.get("prompt"),
            }
        )

        image_id = resolve_image_id(pred_file, payload, image_stem_to_id)
        if image_id is None:
            unmatched_rows.append(
                unmatched_row(pred_file, payload, "", "", None, "image_id_not_found")
            )
            continue

        boxes = payload.get("boxes_xyxy", [])
        scores = payload.get("scores", [])
        labels = payload.get("text_labels", [])
        if not boxes and not scores:
            continue
        if len(boxes) != len(scores) or len(labels) != len(boxes):
            raise ValueError(
                f"Mismatched boxes/scores/text_labels lengths in {pred_file}: "
                f"{len(boxes)}, {len(scores)}, {len(labels)}"
            )

        for box, score, label in zip(boxes, scores, labels):
            normalized_label = normalize_label(str(label))
            category_id = category_name_to_id.get(normalized_label)
            if category_id is None:
                unmatched_rows.append(
                    unmatched_row(
                        pred_file,
                        payload,
                        str(label),
                        normalized_label,
                        score,
                        "category_not_found",
                    )
                )
                continue

            bbox_xywh = xyxy_to_xywh(box)
            if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                unmatched_rows.append(
                    unmatched_row(
                        pred_file,
                        payload,
                        str(label),
                        normalized_label,
                        score,
                        "invalid_bbox",
                    )
                )
                continue

            coco_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": round_list(bbox_xywh),
                    "score": float(score),
                }
            )

    return coco_predictions, unmatched_rows, metadata_rows


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = label.strip(".,;:!?\"'")
    label = re.sub(r"\s+", " ", label)
    while ARTICLE_RE.match(label):
        label = ARTICLE_RE.sub("", label)
    tokens = label.split()
    while tokens and tokens[-1] in {"a", "an", "the"}:
        tokens.pop()
    label = " ".join(tokens)
    return label.strip()


def xyxy_to_xywh(box: list[float]) -> list[float]:
    if len(box) != 4:
        raise ValueError(f"Expected xyxy box with 4 values, got {box}")
    x0, y0, x1, y1 = (float(value) for value in box)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def resolve_image_id(pred_file: Path, payload: dict, image_stem_to_id: dict[str, int]) -> int | None:
    candidates = [pred_file.stem]
    image_path = payload.get("image_path")
    if image_path:
        parsed_path = urlparse(str(image_path)).path
        candidates.append(Path(parsed_path).stem)
        candidates.append(Path(str(image_path)).stem)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in image_stem_to_id:
            return image_stem_to_id[candidate]
    return None


def build_image_stem_map(annotations: dict) -> dict[str, int]:
    mapping = {}
    for image in annotations["images"]:
        stem = Path(image["file_name"]).stem
        if stem in mapping:
            raise ValueError(f"Duplicate image stem in annotations: {stem}")
        mapping[stem] = int(image["id"])
    return mapping


def per_class_ap(coco_eval: object, coco_gt: object, category_id_to_name: dict[int, str]) -> list[dict]:
    precision = coco_eval.eval["precision"]
    params = coco_eval.params
    max_det_idx = params.maxDets.index(100) if 100 in params.maxDets else len(params.maxDets) - 1
    iou_50_idx = int(np.where(np.isclose(params.iouThrs, 0.50))[0][0])
    iou_75_idx = int(np.where(np.isclose(params.iouThrs, 0.75))[0][0])

    rows = []
    for cat_idx, category_id in enumerate(params.catIds):
        values = precision[:, :, cat_idx, 0, max_det_idx]
        values_50 = precision[iou_50_idx, :, cat_idx, 0, max_det_idx]
        values_75 = precision[iou_75_idx, :, cat_idx, 0, max_det_idx]
        gt_count = len(coco_gt.getAnnIds(catIds=[category_id], imgIds=params.imgIds))
        rows.append(
            {
                "category_id": int(category_id),
                "category_name": category_id_to_name[int(category_id)],
                "AP": safe_mean(values),
                "AP50": safe_mean(values_50),
                "AP75": safe_mean(values_75),
                "gt_count": int(gt_count),
            }
        )
    return rows


def build_summary(
    coco_eval: object,
    experiment_name: str,
    annotations: dict,
    pred_dir: Path,
    metadata_rows: list[dict],
    valid_predictions: int,
    unmatched_predictions: int,
) -> dict:
    stats = {key: float(value) for key, value in zip(METRIC_KEYS, coco_eval.stats)}
    summary = {
        "experiment_name": experiment_name,
        "pred_dir": str(pred_dir),
        "image_count": len(annotations["images"]),
        "category_count": len(annotations["categories"]),
        "valid_predictions": valid_predictions,
        "unmatched_predictions": unmatched_predictions,
        "model_id": unique_metadata_value(metadata_rows, "model_id"),
        "box_threshold": unique_metadata_value(metadata_rows, "box_threshold"),
        "text_threshold": unique_metadata_value(metadata_rows, "text_threshold"),
        "inference_mode": unique_metadata_value(metadata_rows, "inference_mode"),
        "nms_iou_threshold": unique_metadata_value(metadata_rows, "nms_iou_threshold"),
        "post_score_cutoff": unique_metadata_value(metadata_rows, "post_score_cutoff"),
    }
    summary.update(stats)
    return summary


def compare_summaries(summary_paths: list[Path], output_dir: Path) -> None:
    summaries = [load_json(path) for path in summary_paths]
    rows = []
    for path, summary in zip(summary_paths, summaries):
        rows.append(
            {
                "source": str(path),
                "experiment_name": summary.get("experiment_name", path.parent.name),
                "box_threshold": summary.get("box_threshold"),
                "text_threshold": summary.get("text_threshold"),
                "inference_mode": summary.get("inference_mode"),
                "nms_iou_threshold": summary.get("nms_iou_threshold"),
                "post_score_cutoff": summary.get("post_score_cutoff"),
                "AP": summary.get("AP"),
                "AP50": summary.get("AP50"),
                "AP75": summary.get("AP75"),
                "AR100": summary.get("AR100"),
                "valid_predictions": summary.get("valid_predictions"),
                "unmatched_predictions": summary.get("unmatched_predictions"),
            }
        )

    comparison_csv = output_dir / "threshold_comparison.csv"
    comparison_plot = output_dir / "threshold_comparison.png"
    write_csv(comparison_csv, rows)
    plot_threshold_comparison(rows, comparison_plot)
    print(f"Comparison CSV: {comparison_csv.resolve()}")
    print(f"Comparison plot: {comparison_plot.resolve()}")


def unique_metadata_value(rows: list[dict], key: str) -> object:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if value not in values:
            values.append(value)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def unmatched_row(
    pred_file: Path,
    payload: dict,
    label: str,
    normalized_label: str,
    score: object,
    reason: str,
) -> dict:
    return {
        "json_file": str(pred_file),
        "image_path": str(payload.get("image_path", "")),
        "label": label,
        "normalized_label": normalized_label,
        "score": "" if score is None else str(score),
        "reason": reason,
    }


def plot_per_class_ap(rows: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: finite_or_zero(row["AP"]), reverse=True)
    labels = [row["category_name"] for row in ordered]
    values = [finite_or_zero(row["AP"]) for row in ordered]

    fig_width = max(9, len(labels) * 0.55)
    plt.figure(figsize=(fig_width, 5))
    plt.bar(labels, values, color="#2a9d8f")
    plt.ylabel("AP")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_threshold_comparison(rows: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["experiment_name"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    for offset, metric in zip((-width, 0, width), ("AP", "AP50", "AP75")):
        values = [finite_or_zero(row.get(metric)) for row in rows]
        plt.bar(x + offset, values, width, label=metric)

    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def safe_mean(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def finite_or_zero(value: object) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return value


def round_list(values: list[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_unmatched_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["json_file", "image_path", "label", "normalized_label", "score", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
