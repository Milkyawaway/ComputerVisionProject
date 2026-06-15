#!/usr/bin/env python
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

from PIL import Image
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ovd import DetectionResult, GroundingDinoDetector, normalize_prompt  # noqa: E402
from ovd.visualize import draw_detections  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grounding DINO open-vocabulary detection.")

    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=Path, help="Path to one image.")
    image_group.add_argument("--image-dir", type=Path, help="Directory with images.")
    image_group.add_argument("--image-url", help="URL of one image.")

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Text prompt, e.g. 'a cat. a remote control.'")
    prompt_group.add_argument("--prompt-file", type=Path, help="File containing class names or a prompt.")

    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.4)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/infer"))
    parser.add_argument(
        "--per-class-prompts",
        action="store_true",
        help="Run one prompt per class and merge predictions. Slower, but reduces mixed text labels.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        help="Apply per-label NMS with this IoU threshold after prediction, e.g. 0.5.",
    )
    parser.add_argument(
        "--post-score-cutoff",
        type=float,
        help="Drop predictions below this score after optional NMS. Useful for precision/recall sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt if args.prompt is not None else read_prompt_file(args.prompt_file)
    inputs = collect_images(args)
    detector = GroundingDinoDetector(model_id=args.model_id, device=args.device)

    for image_ref, image, stem in tqdm(inputs, desc="Detecting", unit="image"):
        result = predict_image(
            detector=detector,
            image=image,
            prompt=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            per_class_prompts=args.per_class_prompts,
            nms_iou_threshold=args.nms_iou_threshold,
            post_score_cutoff=args.post_score_cutoff,
        )

        visualization = draw_detections(image, result.boxes_xyxy, result.scores, result.text_labels)
        vis_path = output_dir / f"{stem}_ovd.jpg"
        json_path = output_dir / f"{stem}.json"

        visualization.save(vis_path, quality=95)
        with json_path.open("w", encoding="utf-8") as f:
            payload = result.to_json_dict(image_ref)
            payload["inference_mode"] = "per_class" if args.per_class_prompts else "multi_class"
            payload["nms_iou_threshold"] = args.nms_iou_threshold
            payload["post_score_cutoff"] = args.post_score_cutoff
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_dir.resolve()}")


def read_prompt_file(path: Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        raise ValueError(f"Prompt file is empty: {path}")
    return ". ".join(lines)


def predict_image(
    detector: GroundingDinoDetector,
    image: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    per_class_prompts: bool,
    nms_iou_threshold: float | None,
    post_score_cutoff: float | None,
) -> DetectionResult:
    if not per_class_prompts:
        result = detector.predict(
            image=image,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        boxes, scores, labels = apply_nms(
            result.boxes_xyxy,
            result.scores,
            result.text_labels,
            nms_iou_threshold,
        )
        boxes, scores, labels = apply_score_cutoff(boxes, scores, labels, post_score_cutoff)
        result.boxes_xyxy = boxes
        result.scores = scores
        result.text_labels = labels
        return result

    normalized_prompt, class_prompts = normalize_prompt(prompt)
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_labels: list[str] = []

    for class_prompt in class_prompts:
        result = detector.predict(
            image=image,
            prompt=class_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        all_boxes.extend(result.boxes_xyxy)
        all_scores.extend(result.scores)
        all_labels.extend([class_prompt] * len(result.boxes_xyxy))

    boxes, scores, labels = apply_nms(all_boxes, all_scores, all_labels, nms_iou_threshold)
    boxes, scores, labels = apply_score_cutoff(boxes, scores, labels, post_score_cutoff)
    return DetectionResult(
        boxes_xyxy=boxes,
        scores=scores,
        text_labels=labels,
        prompt=normalized_prompt,
        model_id=detector.model_id,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        image_size=image.size,
        device=str(detector.device),
    )


def apply_score_cutoff(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    score_cutoff: float | None,
) -> tuple[list[list[float]], list[float], list[str]]:
    if score_cutoff is None:
        return boxes, scores, labels

    keep_indices = [idx for idx, score in enumerate(scores) if score >= score_cutoff]
    return (
        [boxes[idx] for idx in keep_indices],
        [scores[idx] for idx in keep_indices],
        [labels[idx] for idx in keep_indices],
    )


def apply_nms(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float | None,
) -> tuple[list[list[float]], list[float], list[str]]:
    if iou_threshold is None or not boxes:
        return boxes, scores, labels

    keep_indices: list[int] = []
    for label in sorted(set(labels)):
        indices = [idx for idx, candidate in enumerate(labels) if candidate == label]
        indices.sort(key=lambda idx: scores[idx], reverse=True)
        label_keep: list[int] = []
        for idx in indices:
            if all(box_iou_xyxy(boxes[idx], boxes[kept]) <= iou_threshold for kept in label_keep):
                label_keep.append(idx)
        keep_indices.extend(label_keep)

    keep_indices.sort(key=lambda idx: scores[idx], reverse=True)
    return (
        [boxes[idx] for idx in keep_indices],
        [scores[idx] for idx in keep_indices],
        [labels[idx] for idx in keep_indices],
    )


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
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


def collect_images(args: argparse.Namespace) -> list[tuple[str, Image.Image, str]]:
    if args.image is not None:
        image = load_local_image(args.image)
        return [(str(args.image), image, safe_stem(args.image.stem))]

    if args.image_url is not None:
        image = load_url_image(args.image_url)
        return [(args.image_url, image, url_stem(args.image_url))]

    image_paths = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.image_dir}")

    return [(str(path), load_local_image(path), safe_stem(path.stem)) for path in image_paths]


def load_local_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def load_url_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def url_stem(url: str) -> str:
    name = Path(urlparse(url).path).stem
    return safe_stem(name or "url_image")


def safe_stem(stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe or "image"


if __name__ == "__main__":
    main()
