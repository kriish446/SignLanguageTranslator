"""
Capture normalized hand landmark samples per sign for training.
Landmark normalization lives in hand_mediapipe (shared with inference).
"""

import math
import os
import time

import cv2
import numpy as np

from hand_mediapipe import (
    HandLandmarkerSession,
    draw_hand_skeleton_bgr,
    landmarks_to_normalized_features,
)

SIGNS = [
    "Hello",
    "Excuse Me",
    "No",
    "Time Now",
    "Goodbye",
    "Sorry",
    "Help",
    "Yes",
    "Thank You",
]

SAMPLES_PER_SIGN = 200
COUNTDOWN_SEC = 3


def sign_to_filename(sign: str) -> str:
    return sign.replace(" ", "_") + ".npy"


def main():
    os.makedirs("dataset", exist_ok=True)

    session = HandLandmarkerSession()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        session.close()
        return

    for sign in SIGNS:
        t0 = time.time()
        while time.time() - t0 < COUNTDOWN_SEC:
            elapsed = time.time() - t0
            secs_left = max(1, int(math.ceil(COUNTDOWN_SEC - elapsed)))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = session.process_rgb(rgb)
            if lms is not None:
                draw_hand_skeleton_bgr(frame, lms)
            msg = f"Next sign: {sign} — Get ready... ({secs_left}s)"
            cv2.putText(
                frame,
                msg,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect Sign Data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                session.close()
                return

        samples = []
        collected = 0
        while collected < SAMPLES_PER_SIGN:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = session.process_rgb(rgb)

            if lms is not None:
                draw_hand_skeleton_bgr(frame, lms)
                feats = landmarks_to_normalized_features(lms)
                samples.append(feats)
                collected = len(samples)

            progress = f"Collecting: {sign} | Sample {collected}/{SAMPLES_PER_SIGN}"
            cv2.putText(
                frame,
                progress,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect Sign Data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                session.close()
                return

        out_path = os.path.join("dataset", sign_to_filename(sign))
        np.save(out_path, np.stack(samples, axis=0))
        print(f"Saved {out_path} shape {np.load(out_path).shape}")

    cap.release()
    cv2.destroyAllWindows()
    session.close()
    print("Dataset complete! Run train_model.py next.")


if __name__ == "__main__":
    main()
