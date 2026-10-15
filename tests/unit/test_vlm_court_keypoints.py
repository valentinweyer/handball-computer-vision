"""The VLM court-keypoint harness: prompt wording, answer parsing, and scoring.

The scoring tests guard the failure the harness exists to catch: a labelling
that is internally consistent but mirrored fits its own homography perfectly,
so it must still come out as gross point errors and a wrong court position.
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from scripts.vlm_court_keypoints import (
    KEYPOINT_SCHEMA,
    LabellerStopped,
    DESCRIPTIONS,
    answer_points,
    frame_prompt,
    label_frames,
    mistral_payload,
    camera_view,
    court_vertices,
    middle_court,
    parse_predictions,
    score_frame,
)

VERTICES = court_vertices()
# Court cm -> image px, upright like the template: x=0 on the left, y=2000 (far) on top.
CAMERA = np.array([[0.3, 0.02, 0.0], [0.0, -0.25, 620.0], [0.0, 0.00004, 1.0]])


def project(matrix: np.ndarray, ids) -> dict[int, np.ndarray]:
    points = cv2.perspectiveTransform(VERTICES[list(ids)].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return dict(zip(ids, points))


class DescriptionTests(unittest.TestCase):
    def test_every_vertex_is_described(self):
        self.assertEqual(len(DESCRIPTIONS), len(VERTICES))

    def test_side_words_match_the_camera_view(self):
        for index, text in enumerate(DESCRIPTIONS):
            x, y = VERTICES[index]
            with self.subTest(kp=index + 1, text=text):
                self.assertFalse("left" in text and "right" in text)
                self.assertFalse("near" in text and "far" in text)
                if "left" in text:
                    self.assertLess(x, 2000)
                if "right" in text:
                    self.assertGreater(x, 2000)
                if "near" in text:
                    self.assertLess(y, 1000)
                if "far" in text:
                    self.assertGreater(y, 1000)


class ParsingTests(unittest.TestCase):
    def test_accepts_the_key_and_value_spellings(self):
        parsed = parse_predictions({"f.jpg": {"KP17": [1, 2], "05": {"x": 3, "y": 4},
                                              "kp9-L-6m-Lower": [5.5, 6]}})
        self.assertEqual(sorted(parsed["f.jpg"]), [4, 8, 16])
        np.testing.assert_allclose(parsed["f.jpg"][4], [3, 4])

    def test_rejects_rather_than_drops(self):
        for bad in ({"KP38": [1, 2]}, {"KP17": [1]}, {"KP17": [1, None]},
                    {"KP17": [1, 2], "17": [3, 4]}, {"centre": [1, 2]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_predictions({"f.jpg": bad})


class CodexStepTests(unittest.TestCase):
    def test_schema_allows_exactly_the_template_landmarks(self):
        allowed = KEYPOINT_SCHEMA["properties"]["keypoints"]["items"]["properties"]["kp"]["enum"]
        self.assertEqual(allowed, [f"KP{i:02d}" for i in range(1, len(DESCRIPTIONS) + 1)])

    def test_frame_prompt_names_the_frame_and_drops_the_folder_output(self):
        prompt = frame_prompt("# Task\n\nbody\n\n## Output\n\nwrite predictions.json\n", "frame_03.jpg")
        self.assertIn("image 2 is the frame `frame_03.jpg`", prompt)
        self.assertIn("body", prompt)
        self.assertNotIn("## Output", prompt)

    def test_answer_becomes_a_predictions_entry(self):
        answer = {"keypoints": [{"kp": "KP17", "x": 1, "y": 2}, {"kp": "KP14", "x": 3.5, "y": 4}]}
        self.assertEqual(answer_points(answer), {"KP17": [1.0, 2.0], "KP14": [3.5, 4.0]})
        self.assertEqual(parse_predictions({"f.jpg": answer_points(answer)})["f.jpg"].keys(), {13, 16})

    def test_a_landmark_answered_twice_is_rejected(self):
        with self.assertRaises(ValueError):
            answer_points({"keypoints": [{"kp": "KP17", "x": 1, "y": 2}, {"kp": "KP17", "x": 5, "y": 6}]})


class CoordinateTests(unittest.TestCase):
    def test_normalised_answers_come_back_as_pixels(self):
        answer = {"keypoints": [{"kp": "KP17", "x": 500, "y": 500}, {"kp": "KP01", "x": 1000, "y": 0}]}
        points = answer_points(answer, "norm1000")
        self.assertEqual(points["KP17"], [640.0, 360.0])
        self.assertEqual(points["KP01"], [1280.0, 0.0])

    def test_prompt_says_which_coordinates_to_use(self):
        self.assertIn("pixel coordinates", frame_prompt("body", "f.jpg", "pixel"))
        self.assertIn("0..1000", frame_prompt("body", "f.jpg", "norm1000"))


class MistralPayloadTests(unittest.TestCase):
    def test_sends_prompt_then_both_images_and_the_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for name in ("template.png", "frame_01.jpg"):
                path = Path(tmp) / name
                path.write_bytes(b"x")
                images.append(path)
            payload = mistral_payload("m", "the prompt", images, {"temperature": 0})
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "the prompt"})
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(content[2]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertIs(payload["response_format"]["json_schema"]["schema"], KEYPOINT_SCHEMA)
        self.assertEqual(payload["temperature"], 0)


class LabelFramesTests(unittest.TestCase):
    """The per-frame loop: saves as it goes, resumes, keeps malformed answers apart."""

    def make_task(self, root: Path, frames: int) -> None:
        task = root / "run" / "task"
        (task / "frames").mkdir(parents=True)
        (task / "PROMPT.md").write_text("# Task\n\nbody\n\n## Output\n\nx\n")
        for i in range(1, frames + 1):
            (task / "frames" / f"frame_{i:02d}.jpg").write_bytes(b"x")

    def run_loop(self, root: Path, ask) -> dict:
        with mock.patch("scripts.vlm_court_keypoints.RUNS_DIR", root), \
                mock.patch("builtins.print"):
            label_frames(argparse.Namespace(name="run", coords="pixel"), "fake", ask)
        return json.loads((root / "run" / "task" / "predictions.json").read_text())

    def test_stops_on_a_failed_request_and_resumes_without_asking_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_task(root, 3)
            asked = []

            def flaky(task_dir, frame, prompt):
                asked.append(frame)
                if frame == "frame_02.jpg" and asked.count(frame) == 1:
                    raise LabellerStopped("limit")
                return {"raw": json.dumps({"keypoints": [{"kp": "KP17", "x": 1, "y": 2}]})}

            self.assertEqual(sorted(self.run_loop(root, flaky)), ["frame_01.jpg"])
            self.assertEqual(sorted(self.run_loop(root, flaky)),
                             ["frame_01.jpg", "frame_02.jpg", "frame_03.jpg"])
            self.assertEqual(asked.count("frame_01.jpg"), 1)

    def test_malformed_answer_is_kept_but_not_predicted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_task(root, 1)
            predictions = self.run_loop(root, lambda *a: {"raw": "not json"})
            self.assertEqual(predictions, {})
            record = json.loads((root / "run" / "fake" / "frame_01.jpg.json").read_text())
            self.assertIn("malformed", record)


class ViewTests(unittest.TestCase):
    def test_upright_view_is_accepted_and_a_mirror_is_not(self):
        ids = [13, 14, 15, 16, 17, 18, 19, 35, 36]
        self.assertTrue(camera_view(project(CAMERA, ids), VERTICES))
        mirror = np.diag([-1.0, 1.0, 1.0]) @ CAMERA
        self.assertFalse(camera_view(project(mirror, ids), VERTICES))

    def test_middle_court_means_no_goal_line_landmark(self):
        self.assertTrue(middle_court(project(CAMERA, [13, 16, 19, 35]), VERTICES))
        self.assertFalse(middle_court(project(CAMERA, [13, 16, 29]), VERTICES))


class ScoringTests(unittest.TestCase):
    # Symmetric about the centre line, so a mirror stays inside the labelled set.
    truth_ids = [13, 14, 15, 16, 17, 18, 19, 33, 34, 35, 36, 6, 9, 23, 26]

    def test_perfect_labels_score_zero(self):
        truth = project(CAMERA, self.truth_ids)
        result = score_frame(truth, dict(truth), VERTICES)
        self.assertEqual(result["recall"], 1.0)
        self.assertLess(max(p["error_px"] for p in result["points"]), 1e-3)
        self.assertLess(result["court_error_cm"], 1.0)

    def test_landmark_the_annotator_skipped_is_reported_not_scored(self):
        truth = project(CAMERA, self.truth_ids)
        extra = {22: np.array([5000.0, 5000.0])}  # would be gross if it were scored
        result = score_frame(truth, {**truth, **extra}, VERTICES)
        self.assertEqual([p["kp"] for p in result["unlabelled"]], [23])
        self.assertNotIn(23, [p["kp"] for p in result["points"]])

    def test_mirrored_labelling_is_gross_despite_fitting_itself(self):
        truth = project(CAMERA, self.truth_ids)
        mirrored_x = VERTICES[:, 0].max() - VERTICES[:, 0]
        swap = {
            i: int(np.argmin(np.hypot(VERTICES[:, 0] - mirrored_x[i], VERTICES[:, 1] - VERTICES[i, 1])))
            for i in self.truth_ids
        }
        predicted = {swap[i]: truth[i] for i in self.truth_ids}
        result = score_frame(truth, predicted, VERTICES)
        # A centre view keeps its centre line under the mirror, so the median
        # court error is metres, not the court length -- still far off the floor.
        self.assertGreater(result["court_error_cm"], 200)
        self.assertGreater(sum(p["error_px"] > 40 for p in result["points"]), 5)


if __name__ == "__main__":
    unittest.main()
