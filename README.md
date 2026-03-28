# Sign Language Interpreter

Desktop app that detects hand signs in real time using MediaPipe Hands, OpenCV, and scikit-learn.

## Installation

```bash
pip install mediapipe opencv-python scikit-learn numpy
```

## How to Run

From the `sign_language_interpreter` folder:

**Step 1:** `python collect_data.py` — sit in good lighting, show each sign clearly

**Step 2:** `python train_model.py` — trains model, prints accuracy and saves `model.pkl`

**Step 3:** `python inference.py` — opens webcam (requires `model.pkl` from step 2)

### Python 3.13 and newer MediaPipe

Recent `mediapipe` builds no longer ship `mp.solutions`. This project uses the **Hand Landmarker Tasks** API instead. The first run downloads `hand_landmarker.task` (~10 MB) into this folder.

## Signs Supported

Hello, Excuse Me, No, Time Now, Goodbye, Sorry, Help, Yes, Thank You

## How to Add a New Sign Later

1. Add sign name to the list in `collect_data.py`
2. Re-run `collect_data.py` (only needs new sign samples)
3. Re-run `train_model.py` to retrain
4. `inference.py` auto-picks up new sign

## Google Meet Integration Note

This project is built to be modular. The `SignLanguageDetector` class in `inference.py` can be directly imported into a Chrome Extension background script via a local Python WebSocket server.
