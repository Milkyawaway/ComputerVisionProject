"""Open-vocabulary detection helpers for Project 4."""

from .grounding_dino import DetectionResult, GroundingDinoDetector, normalize_prompt

__all__ = ["DetectionResult", "GroundingDinoDetector", "normalize_prompt"]
