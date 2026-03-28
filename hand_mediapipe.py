"""
MediaPipe Hand Landmarker using the Tasks API (required on Python 3.13+; mp.solutions removed).
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL_FILE = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Same topology as classic MediaPipe Hands drawing
HAND_CONNECTIONS = [
    (0, 1),
    (0, 5),
    (9, 13),
    (13, 17),
    (5, 9),
    (0, 17),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _default_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILE)


def ensure_hand_landmarker_model(path: str | None = None) -> str:
    path = path or _default_model_path()
    if not os.path.isfile(path):
        print("Downloading hand landmarker model (one-time, ~10 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def landmarks_to_normalized_features(landmarks) -> np.ndarray:
    wrist = landmarks[0]
    features = []
    for i in range(21):
        features.append(landmarks[i].x - wrist.x)
        features.append(landmarks[i].y - wrist.y)
    return np.array(features, dtype=np.float64)


class HandLandmarkerSession:
    """VIDEO-mode landmarker; timestamps must increase each frame."""

    def __init__(self, model_path: str | None = None):
        asset = ensure_hand_landmarker_model(model_path)
        base_options = python.BaseOptions(model_asset_path=asset)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process_rgb(self, rgb: np.ndarray):
        """
        rgb: HxWx3 uint8 RGB. Returns list of 21 landmarks (.x, .y, .z) or None.
        """
        self._ts_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._lm.detect_for_video(mp_image, self._ts_ms)
        if not result.hand_landmarks:
            return None
        hl = result.hand_landmarks[0]
        return [hl[i] for i in range(21)]

    def close(self):
        self._lm.close()


def draw_hand_skeleton_bgr(
    frame,
    landmarks_seq,
    line_color=(0, 255, 0),
    point_color=(255, 0, 0),
):
    """Draw 21-landmark hand on BGR frame in-place."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_seq]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, point_color, -1, cv2.LINE_AA)
