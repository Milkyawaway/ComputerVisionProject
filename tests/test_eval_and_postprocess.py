from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_coco import normalize_label, xyxy_to_xywh  # noqa: E402
from infer import apply_nms, apply_score_cutoff  # noqa: E402
from ovd import normalize_prompt  # noqa: E402


class EvaluationUtilityTests(unittest.TestCase):
    def test_normalize_label_removes_articles(self) -> None:
        self.assertEqual(normalize_label("a cat"), "cat")
        self.assertEqual(normalize_label("an umbrella"), "umbrella")
        self.assertEqual(normalize_label("the traffic light."), "traffic light")

    def test_xyxy_to_xywh(self) -> None:
        self.assertEqual(xyxy_to_xywh([10, 20, 35, 50]), [10.0, 20.0, 25.0, 30.0])


class InferencePostprocessTests(unittest.TestCase):
    def test_normalize_prompt_adds_articles_and_periods(self) -> None:
        prompt, labels = normalize_prompt("person, umbrella, traffic light")
        self.assertEqual(prompt, "a person. an umbrella. a traffic light.")
        self.assertEqual(labels, ["a person", "an umbrella", "a traffic light"])

    def test_apply_nms_is_per_label(self) -> None:
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]]
        scores = [0.9, 0.8, 0.7]
        labels = ["a cat", "a cat", "a dog"]

        kept_boxes, kept_scores, kept_labels = apply_nms(boxes, scores, labels, 0.5)

        self.assertEqual(kept_boxes, [[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(kept_scores, [0.9, 0.7])
        self.assertEqual(kept_labels, ["a cat", "a dog"])

    def test_apply_score_cutoff_filters_low_scores(self) -> None:
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        scores = [0.29, 0.31]
        labels = ["a cat", "a dog"]

        kept_boxes, kept_scores, kept_labels = apply_score_cutoff(boxes, scores, labels, 0.30)

        self.assertEqual(kept_boxes, [[20, 20, 30, 30]])
        self.assertEqual(kept_scores, [0.31])
        self.assertEqual(kept_labels, ["a dog"])


if __name__ == "__main__":
    unittest.main()
