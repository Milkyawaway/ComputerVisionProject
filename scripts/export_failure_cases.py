#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GT_COLOR = (42, 157, 143)
TP_COLOR = (38, 70, 83)
FP_COLOR = (230, 57, 70)
FN_COLOR = (244, 162, 97)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visual false-positive and false-negative cases.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="COCO detection JSON from eval_coco.py.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/failure_cases"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_json(args.annotations)
    predictions = [
        prediction
        for prediction in load_json(args.predictions)
        if float(prediction.get("score", 0.0)) >= args.score_threshold
    ]

    images_by_id = {int(image["id"]): image for image in annotations["images"]}
    categories = {int(category["id"]): category["name"] for category in annotations["categories"]}
    gt_by_image = group_by_image(annotations["annotations"])
    pred_by_image = group_by_image(predictions)

    case_rows = []
    detail_by_image = {}
    for image_id, image in images_by_id.items():
        details = analyze_image(
            image_id=image_id,
            gt_items=gt_by_image.get(image_id, []),
            pred_items=pred_by_image.get(image_id, []),
            iou_threshold=args.iou_threshold,
        )
        detail_by_image[image_id] = details
        case_rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "tp_count": len(details["tp_predictions"]),
                "fp_count": len(details["fp_predictions"]),
                "fn_count": len(details["fn_annotations"]),
                "gt_count": len(gt_by_image.get(image_id, [])),
                "prediction_count": len(pred_by_image.get(image_id, [])),
            }
        )

    write_csv(args.output_dir / "failure_cases.csv", case_rows)
    export_ranked_cases(
        output_dir=args.output_dir / "false_positive_cases",
        rows=sorted(case_rows, key=lambda row: (row["fp_count"], row["prediction_count"]), reverse=True),
        detail_by_image=detail_by_image,
        images_by_id=images_by_id,
        categories=categories,
        image_dir=args.image_dir,
        top_k=args.top_k,
    )
    export_ranked_cases(
        output_dir=args.output_dir / "false_negative_cases",
        rows=sorted(case_rows, key=lambda row: (row["fn_count"], row["gt_count"]), reverse=True),
        detail_by_image=detail_by_image,
        images_by_id=images_by_id,
        categories=categories,
        image_dir=args.image_dir,
        top_k=args.top_k,
    )

    print(f"Failure case CSV: {(args.output_dir / 'failure_cases.csv').resolve()}")
    print(f"False-positive visuals: {(args.output_dir / 'false_positive_cases').resolve()}")
    print(f"False-negative visuals: {(args.output_dir / 'false_negative_cases').resolve()}")


def analyze_image(
    image_id: int,
    gt_items: list[dict],
    pred_items: list[dict],
    iou_threshold: float,
) -> dict[str, list[dict]]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_predictions = []

    predictions = sorted(enumerate(pred_items), key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    for pred_idx, prediction in predictions:
        best_gt_idx = None
        best_iou = 0.0
        pred_category = int(prediction["category_id"])
        pred_box = xywh_to_xyxy(prediction["bbox"])
        for gt_idx, annotation in enumerate(gt_items):
            if gt_idx in matched_gt or int(annotation["category_id"]) != pred_category:
                continue
            iou = box_iou(pred_box, xywh_to_xyxy(annotation["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            tp_item = dict(prediction)
            tp_item["matched_iou"] = best_iou
            tp_predictions.append(tp_item)

    fp_predictions = [prediction for idx, prediction in enumerate(pred_items) if idx not in matched_pred]
    fn_annotations = [annotation for idx, annotation in enumerate(gt_items) if idx not in matched_gt]
    return {
        "tp_predictions": tp_predictions,
        "fp_predictions": fp_predictions,
        "fn_annotations": fn_annotations,
        "gt_annotations": gt_items,
    }


def export_ranked_cases(
    output_dir: Path,
    rows: list[dict],
    detail_by_image: dict[int, dict[str, list[dict]]],
    images_by_id: dict[int, dict],
    categories: dict[int, str],
    image_dir: Path,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for row in rows:
        if row["fp_count"] == 0 and row["fn_count"] == 0:
            continue
        image_id = int(row["image_id"])
        image_info = images_by_id[image_id]
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            continue

        visual = draw_case(Image.open(image_path).convert("RGB"), detail_by_image[image_id], categories)
        out_name = (
            f"{exported + 1:02d}_img{image_id}_"
            f"fp{row['fp_count']}_fn{row['fn_count']}_{Path(image_info['file_name']).stem}.jpg"
        )
        visual.save(output_dir / out_name, quality=95)
        exported += 1
        if exported >= top_k:
            break


def draw_case(image: Image.Image, details: dict[str, list[dict]], categories: dict[int, str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(canvas.size) / 350))

    for annotation in details["gt_annotations"]:
        draw_labeled_box(
            draw,
            xywh_to_xyxy(annotation["bbox"]),
            f"GT {categories[int(annotation['category_id'])]}",
            GT_COLOR,
            font,
            line_width,
        )
    for prediction in details["tp_predictions"]:
        draw_labeled_box(
            draw,
            xywh_to_xyxy(prediction["bbox"]),
            f"TP {categories[int(prediction['category_id'])]} {prediction['score']:.2f}",
            TP_COLOR,
            font,
            line_width,
        )
    for prediction in details["fp_predictions"]:
        draw_labeled_box(
            draw,
            xywh_to_xyxy(prediction["bbox"]),
            f"FP {categories[int(prediction['category_id'])]} {prediction['score']:.2f}",
            FP_COLOR,
            font,
            line_width,
        )
    for annotation in details["fn_annotations"]:
        draw_labeled_box(
            draw,
            xywh_to_xyxy(annotation["bbox"]),
            f"FN {categories[int(annotation['category_id'])]}",
            FN_COLOR,
            font,
            line_width,
        )
    return canvas


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    x0, y0, x1, y1 = [round(value) for value in box]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    pad = 3
    y_text = max(0, y0 - text_h - 2 * pad)
    draw.rectangle((x0, y_text, x0 + text_w + 2 * pad, y_text + text_h + 2 * pad), fill=color)
    draw.text((x0 + pad, y_text + pad), label, fill=(255, 255, 255), font=font)


def group_by_image(items: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for item in items:
        grouped.setdefault(int(item["image_id"]), []).append(item)
    return grouped


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
