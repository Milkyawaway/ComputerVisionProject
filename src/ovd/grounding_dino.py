from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ARTICLE_PREFIXES = ("a ", "an ", "the ")
VOWEL_INITIALS = tuple("aeiou")


@dataclass
class DetectionResult:
    boxes_xyxy: list[list[float]]
    scores: list[float]
    text_labels: list[str]
    prompt: str
    model_id: str
    box_threshold: float
    text_threshold: float
    image_size: tuple[int, int]
    device: str

    def to_json_dict(self, image_path: str) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_path"] = image_path
        payload["image_size"] = {"width": self.image_size[0], "height": self.image_size[1]}
        return payload


def split_prompt(prompt: str) -> list[str]:
    """Split class prompts from periods, newlines, semicolons, or commas."""
    text = prompt.strip().lower()
    if not text:
        raise ValueError("Prompt is empty.")

    if "." in text or "\n" in text or ";" in text:
        parts = re.split(r"[.;\n]+", text)
    else:
        parts = text.split(",")

    labels = []
    for part in parts:
        label = part.strip().strip(".")
        label = re.sub(r"\s+", " ", label)
        if label:
            labels.append(label)

    if not labels:
        raise ValueError("Prompt does not contain any valid labels.")
    return labels


def normalize_prompt(prompt: str) -> tuple[str, list[str]]:
    """Return a Grounding DINO friendly prompt and label list.

    Grounding DINO is most stable with lower-case English phrases separated by
    periods. Bare class names are converted from "person" to "a person".
    """
    labels = []
    for label in split_prompt(prompt):
        if not label.startswith(ARTICLE_PREFIXES):
            article = "an" if label.startswith(VOWEL_INITIALS) else "a"
            label = f"{article} {label}"
        labels.append(label)

    normalized = ". ".join(labels) + "."
    return normalized, labels


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.model_id = model_id
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        image = image.convert("RGB")
        normalized_prompt, labels = normalize_prompt(prompt)

        inputs = self._prepare_inputs(image, normalized_prompt, labels)
        outputs = self.model(**inputs)
        result = self._post_process(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            image=image,
            labels=labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        text_labels = result.get("text_labels", result.get("labels", []))

        return DetectionResult(
            boxes_xyxy=_tensor_to_nested_float_list(boxes),
            scores=_tensor_to_float_list(scores),
            text_labels=[str(label) for label in text_labels],
            prompt=normalized_prompt,
            model_id=self.model_id,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_size=image.size,
            device=str(self.device),
        )

    def _prepare_inputs(
        self,
        image: Image.Image,
        normalized_prompt: str,
        labels: list[str],
    ) -> dict[str, Any]:
        try:
            inputs = self.processor(images=image, text=[labels], return_tensors="pt")
        except Exception:
            inputs = self.processor(images=image, text=normalized_prompt, return_tensors="pt")

        return inputs.to(self.device)

    def _post_process(
        self,
        outputs: Any,
        input_ids: torch.Tensor | None,
        image: Image.Image,
        labels: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> dict[str, Any]:
        target_sizes = [(image.height, image.width)]

        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
                text_labels=[labels],
            )
        except TypeError:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

        return processed[0]


def _tensor_to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def _tensor_to_nested_float_list(values: Any) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [[float(coord) for coord in box] for box in values]
