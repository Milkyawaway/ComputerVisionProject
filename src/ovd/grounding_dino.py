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
    """Grounding DINO 单张图片的统一输出格式。"""

    # boxes_xyxy 使用 [x0, y0, x1, y1]，这是 HuggingFace 后处理返回的检测框格式。
    boxes_xyxy: list[list[float]]

    # 每个检测框对应的置信度。
    scores: list[float]

    # 每个检测框对应的文本标签，例如 "a person" 或 "a traffic light"。
    text_labels: list[str]

    # 实际送入模型的规范化 prompt。
    prompt: str

    # 模型和推理参数会一起写入 JSON，方便后续复现实验。
    model_id: str
    box_threshold: float
    text_threshold: float
    image_size: tuple[int, int]
    device: str

    def to_json_dict(self, image_path: str) -> dict[str, Any]:
        """转换为 infer.py 保存到磁盘的 JSON 结构。"""
        payload = asdict(self)
        payload["image_path"] = image_path
        payload["image_size"] = {"width": self.image_size[0], "height": self.image_size[1]}
        return payload


def split_prompt(prompt: str) -> list[str]:
    """把原始 prompt 拆成类别短语列表。"""
    text = prompt.strip().lower()
    if not text:
        raise ValueError("Prompt is empty.")

    # prompt 文件通常是一行一个类别，或者用英文句号分隔；
    # 手动输入时也可能用逗号分隔，所以这里统一兼容几种写法。
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
    """把 prompt 规范化成 Grounding DINO 更稳定的英文格式。"""
    labels = []
    for label in split_prompt(prompt):
        # Grounding DINO 推荐使用短语式 prompt。裸类别名如 "person"
        # 会被补成 "a person"，元音开头则补成 "an umbrella"。
        if not label.startswith(ARTICLE_PREFIXES):
            article = "an" if label.startswith(VOWEL_INITIALS) else "a"
            label = f"{article} {label}"
        labels.append(label)

    # 多个类别之间用英文句号分隔，例如：
    # "a person. a car. a traffic light."
    normalized = ". ".join(labels) + "."
    return normalized, labels


class GroundingDinoDetector:
    """HuggingFace Grounding DINO 的轻量封装。"""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
    ) -> None:
        # 如果用户指定 cuda 但当前环境没有 CUDA，就自动回退到 CPU，
        # 避免脚本在无 GPU 环境下直接崩溃。
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.model_id = model_id
        self.device = torch.device(device)

        # 这里是本项目直接调用的 HuggingFace 接口：
        # processor 负责图像/文本预处理和后处理，model 负责前向推理。
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        # 本项目只做推理和评估，不训练模型，所以切到 eval 模式。
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        """对单张图片执行开放词汇检测。"""
        image = image.convert("RGB")
        normalized_prompt, labels = normalize_prompt(prompt)

        # 1. 把 PIL 图片和文本 prompt 转换成模型输入 tensor。
        inputs = self._prepare_inputs(image, normalized_prompt, labels)

        # 2. 前向推理。这里输出仍是模型原始结果，不是最终 bbox。
        outputs = self.model(**inputs)

        # 3. HuggingFace 后处理：根据 threshold 过滤预测，
        # 并把框缩放回原图尺寸。
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

        # 4. 统一转换为纯 Python list，便于 JSON 序列化和后续评估。
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
        """准备模型输入，兼容不同 transformers 版本的文本输入格式。"""
        try:
            # 新版本 transformers 对 Grounding DINO 支持 list-of-labels 格式，
            # 可以更准确地保留每个文本短语的边界。
            inputs = self.processor(images=image, text=[labels], return_tensors="pt")
        except Exception:
            # 旧版本或接口不兼容时，回退到普通字符串 prompt。
            inputs = self.processor(images=image, text=normalized_prompt, return_tensors="pt")

        # processor 返回的是 BatchEncoding，需要整体移动到 GPU/CPU。
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
        """把模型原始输出转换为检测框、分数和文本标签。"""
        target_sizes = [(image.height, image.width)]

        try:
            # 新版 transformers 使用 threshold 参数，并支持 text_labels。
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
                text_labels=[labels],
            )
        except TypeError:
            # 兼容旧版 transformers：参数名可能是 box_threshold，
            # 且不支持显式传入 text_labels。
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

        # 本项目一次只处理一张图，所以取 batch 中的第 0 个结果。
        return processed[0]


def _tensor_to_float_list(values: Any) -> list[float]:
    """把 torch tensor 或普通序列转换成 float list。"""
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def _tensor_to_nested_float_list(values: Any) -> list[list[float]]:
    """把 Nx4 的 tensor/list 转换成 list[list[float]]。"""
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [[float(coord) for coord in box] for box in values]
