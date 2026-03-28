"""
Real-time sign detection. Exposes SignLanguageDetector for reuse (e.g. Google Meet).
Landmark normalization lives in hand_mediapipe (shared with collect_data).
"""

import collections
import os
import pickle

import cv2
import numpy as np

from hand_mediapipe import (
    HandLandmarkerSession,
    draw_hand_skeleton_bgr,
    landmarks_to_normalized_features,
)


class SignLanguageDetector:
    """Loadable detector: predict(frame) -> (label, confidence); draw_overlay for UI."""

    def __init__(self, model_path: str = "model.pkl"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        self._hands = HandLandmarkerSession()
        self._last_landmarks = None
        self._last_hand_detected = False
        self._last_proba = None

    def close(self):
        self._hands.close()

    def confidence_for_class(self, class_name: str) -> float:
        """Confidence [0,100] for a class from the last predict() with a hand."""
        if not class_name or self._last_proba is None:
            return 0.0
        classes = list(self._model.classes_)
        if class_name not in classes:
            return 0.0
        idx = classes.index(class_name)
        return float(self._last_proba[idx] * 100.0)

    def predict(self, frame):
        """
        Run detection and classification on a BGR frame (numpy array).
        Returns (label, confidence) with confidence in [0, 100].
        If no hand: ("", 0.0).
        """
        if frame is None or frame.size == 0:
            self._last_hand_detected = False
            self._last_landmarks = None
            self._last_proba = None
            return "", 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms = self._hands.process_rgb(rgb)

        if lms is None:
            self._last_hand_detected = False
            self._last_landmarks = None
            self._last_proba = None
            return "", 0.0

        self._last_hand_detected = True
        self._last_landmarks = lms

        feats = landmarks_to_normalized_features(lms).reshape(1, -1)
        pred = self._model.predict(feats)[0]
        proba = self._model.predict_proba(feats)[0]
        self._last_proba = proba
        classes = getattr(self._model, "classes_", None)
        if classes is not None:
            idx = list(classes).index(pred)
            conf = float(proba[idx] * 100.0)
        else:
            conf = float(np.max(proba) * 100.0)
        return str(pred), conf

    def draw_overlay(self, frame, label: str, confidence: float):
        """Draw hand skeleton (if last predict had a hand) and label/confidence text."""
        out = frame.copy()
        h, w = out.shape[:2]

        if self._last_hand_detected and self._last_landmarks is not None:
            draw_hand_skeleton_bgr(out, self._last_landmarks)

        if label:
            text = label
            conf_text = f"{confidence:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.4
            thickness = 3
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            (cw, ch), _ = cv2.getTextSize(conf_text, font, 0.7, 2)
            cx = (w - max(tw, cw)) // 2
            y_base = h - 50
            cv2.putText(
                out, text, (cx, y_base), font, scale, (0, 255, 0), thickness, cv2.LINE_AA
            )
            ccx = (w - cw) // 2
            cv2.putText(
                out,
                conf_text,
                (ccx, y_base + th + 8),
                font,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out


def _majority_vote(labels):
    if not labels:
        return ""
    counts = collections.Counter(labels)
    best = counts.most_common(1)[0][0]
    return best


def main():
    detector = SignLanguageDetector("model.pkl")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        detector.close()
        return

    history = collections.deque(maxlen=10)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        label, conf = detector.predict(frame)

        if not label:
            history.clear()
            out = detector.draw_overlay(frame, "", 0.0)
            cv2.putText(
                out,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            history.append(label)
            smooth_label = _majority_vote(list(history))
            display_conf = detector.confidence_for_class(smooth_label)
            out = detector.draw_overlay(frame, smooth_label, display_conf)

        cv2.imshow("Sign Language Interpreter", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
