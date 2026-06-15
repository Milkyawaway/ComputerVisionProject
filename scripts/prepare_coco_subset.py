#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys
from urllib.parse import urljoin
from zipfile import ZipFile

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ovd import normalize_prompt  # noqa: E402


DEFAULT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "dog",
    "cat",
    "horse",
    "chair",
    "couch",
    "dining table",
    "bottle",
    "cup",
    "laptop",
    "book",
    "backpack",
    "umbrella",
    "cell phone",
]

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a COCO val2017 subset for open-vocabulary detection.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/coco_subset"))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--annotations-json", type=Path, help="Existing instances_val2017.json path.")
    parser.add_argument("--annotations-zip-url", default=ANNOTATIONS_ZIP_URL)
    parser.add_argument("--image-base-url", default=VAL_IMAGE_BASE_URL)
    parser.add_argument("--source-image-dir", type=Path, help="Optional local val2017 image directory to copy from.")
    parser.add_argument("--class-file", type=Path, help="Optional class list, one class per line.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be at least 1")

    output_dir = args.output_dir
    annotations_dir = output_dir / "annotations"
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    prompts_dir = output_dir / "prompts"
    downloads_dir = output_dir / "downloads"

    for directory in (annotations_dir, metadata_dir, prompts_dir, downloads_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.download_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(args.class_file) if args.class_file else DEFAULT_CLASSES
    annotations_json = args.annotations_json or ensure_coco_annotations(downloads_dir, args.annotations_zip_url)

    with annotations_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    subset, manifest_rows = build_subset(
        coco=coco,
        class_names=class_names,
        max_images=args.max_images,
        seed=args.seed,
        image_base_url=args.image_base_url,
        images_dir=images_dir,
        download_images=args.download_images,
        source_image_dir=args.source_image_dir,
    )

    subset_path = annotations_dir / "instances_val2017_subset.json"
    manifest_path = metadata_dir / "subset_manifest.csv"
    prompt_path = prompts_dir / "coco_20_classes.txt"

    with subset_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    write_manifest(manifest_path, manifest_rows)
    normalized_prompt, _ = normalize_prompt(". ".join(class_names))
    prompt_path.write_text(normalized_prompt + "\n", encoding="utf-8")

    print(f"Images selected: {len(subset['images'])}")
    print(f"Annotations kept: {len(subset['annotations'])}")
    print(f"Subset annotation: {subset_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Prompt file: {prompt_path.resolve()}")


def read_class_names(path: Path) -> list[str]:
    classes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                classes.append(line)
    if not classes:
        raise ValueError(f"Class file is empty: {path}")
    return classes


def ensure_coco_annotations(downloads_dir: Path, annotations_zip_url: str) -> Path:
    annotations_json = downloads_dir / "annotations" / "instances_val2017.json"
    if annotations_json.exists():
        return annotations_json

    zip_path = downloads_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download_file(annotations_zip_url, zip_path)

    with ZipFile(zip_path) as archive:
        archive.extractall(downloads_dir)

    if not annotations_json.exists():
        raise FileNotFoundError(f"instances_val2017.json not found after extracting {zip_path}")
    return annotations_json


def build_subset(
    coco: dict,
    class_names: list[str],
    max_images: int,
    seed: int,
    image_base_url: str,
    images_dir: Path,
    download_images: bool,
    source_image_dir: Path | None,
) -> tuple[dict, list[dict[str, str]]]:
    rng = random.Random(seed)

    categories_by_name = {category["name"]: category for category in coco["categories"]}
    missing = [name for name in class_names if name not in categories_by_name]
    if missing:
        raise ValueError(f"Classes not found in COCO categories: {', '.join(missing)}")

    selected_categories = [categories_by_name[name] for name in class_names]
    selected_category_ids = {category["id"] for category in selected_categories}
    category_name_by_id = {category["id"]: category["name"] for category in selected_categories}

    annotations_by_category: dict[int, set[int]] = {category_id: set() for category_id in selected_category_ids}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        if category_id not in selected_category_ids:
            continue
        image_id = annotation["image_id"]
        annotations_by_category[category_id].add(image_id)
        annotations_by_image.setdefault(image_id, []).append(annotation)

    selected_image_ids = choose_image_ids(annotations_by_category, annotations_by_image, max_images, rng)
    images_by_id = {image["id"]: image for image in coco["images"]}
    selected_images = [images_by_id[image_id] for image_id in selected_image_ids if image_id in images_by_id]
    selected_annotations = [
        annotation
        for image in selected_images
        for annotation in annotations_by_image.get(image["id"], [])
    ]

    manifest_rows = []
    for image in selected_images:
        file_name = image["file_name"]
        image_url = urljoin(image_base_url, file_name)
        local_path = images_dir / file_name

        if download_images:
            copy_or_download_image(
                file_name=file_name,
                image_url=image_url,
                local_path=local_path,
                source_image_dir=source_image_dir,
            )

        image_category_names = sorted(
            {
                category_name_by_id[annotation["category_id"]]
                for annotation in annotations_by_image.get(image["id"], [])
            }
        )
        manifest_rows.append(
            {
                "image_id": str(image["id"]),
                "file_name": file_name,
                "width": str(image["width"]),
                "height": str(image["height"]),
                "coco_url": image_url,
                "local_path": str(local_path if download_images else ""),
                "categories": ";".join(image_category_names),
            }
        )

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
    }
    return subset, manifest_rows


def choose_image_ids(
    annotations_by_category: dict[int, set[int]],
    annotations_by_image: dict[int, list[dict]],
    max_images: int,
    rng: random.Random,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()

    category_ids = list(annotations_by_category)
    rng.shuffle(category_ids)
    grouped_image_ids = {
        category_id: rng.sample(sorted(image_ids), k=len(image_ids))
        for category_id, image_ids in annotations_by_category.items()
    }

    # First pass: try to cover as many selected classes as possible.
    for category_id in category_ids:
        if len(selected) >= max_images:
            break
        for image_id in grouped_image_ids[category_id]:
            if image_id not in selected_set:
                selected.append(image_id)
                selected_set.add(image_id)
                break

    # Fill remaining slots with random candidate images.
    all_candidates = list(annotations_by_image)
    rng.shuffle(all_candidates)
    for image_id in all_candidates:
        if len(selected) >= max_images:
            break
        if image_id in selected_set:
            continue
        selected.append(image_id)
        selected_set.add(image_id)

    return selected


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_id", "file_name", "width", "height", "coco_url", "local_path", "categories"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_or_download_image(
    file_name: str,
    image_url: str,
    local_path: Path,
    source_image_dir: Path | None,
) -> None:
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    if source_image_dir is not None:
        source_path = source_image_dir / file_name
        if source_path.exists() and source_path.stat().st_size > 0:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, local_path)
            return
    download_file(image_url, local_path, quiet=True)


def download_file(url: str, output_path: Path, quiet: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
            disable=quiet,
        )
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()

    shutil.move(str(tmp_path), output_path)


if __name__ == "__main__":
    main()
