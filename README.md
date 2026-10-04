<div align="center">

# ✋ Hand Gesture Control

**Control your computer without touching it — mouse, slides and an air-drawing whiteboard, driven by real-time hand tracking and a gesture classifier trained on your own data.**

[![CI](https://github.com/haroonkhan02/hand-gesture-control/actions/workflows/ci.yml/badge.svg)](https://github.com/haroonkhan02/hand-gesture-control/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

<!-- Demo: record with `gesture-control` and save as assets/demo.gif -->
<!-- <p align="center"><img src="assets/demo.gif" width="720" alt="demo"></p> -->

---

## Highlights

- **Three modes in one app** — gesture mouse (move / click / drag / scroll / right-click), presentation remote (swipe to change slides), and an air-drawing whiteboard.
- **Custom-trained classifier** — record your own gestures, train an MLP on normalized 3D landmarks, and compare it against a rule-based baseline with an honest, *session-level* train/test split.
- **Smooth & stable** — One Euro filtering removes cursor jitter without adding lag; temporal debouncing stops single-frame misclassifications from firing clicks.
- **Fast** — ~8 ms per frame for hand tracking on an Apple M5 CPU (>100 FPS processing throughput); classification adds < 0.1 ms.
- **Tested** — geometry, filtering, controllers and the training pipeline are unit-tested with synthetic hand poses and a fake input backend; CI runs on every push.

## Gestures

| Gesture | 🖱️ Mouse mode | 📊 Slides mode | 🎨 Draw mode |
|---|---|---|---|
| ☝️ `point` | Move cursor | — | Draw |
| 🤏 `pinch` | Hold left button (click / drag) | — | — |
| ✌️ `peace` | Scroll (move hand up/down) | — | Pen up (move without drawing) |
| 👍 `thumbs_up` | Right click | — | — |
| ✋ `open_palm` | Pause cursor | Swipe ← next · Swipe → previous | Erase |
| ✊ `fist` | Pause cursor (re-centre your hand) | Black screen on/off | — |

**Keyboard:** `M` switch mode · `P` pause · `C` clear canvas · `K` next colour · `S` save drawing · `Q` quit

## How it works

```mermaid
flowchart LR
    A[Webcam frame] --> B[MediaPipe<br/>Hand Landmarker]
    B -->|21 × 3D landmarks| C[Normalize<br/>translate · mirror · scale]
    C --> D{Classifier}
    D -->|rules baseline| E[Debouncer<br/>N stable frames]
    D -->|trained MLP| E
    E --> F[Mode controller]
    F --> G[Mouse<br/>One Euro filter]
    F --> H[Slides<br/>swipe detector]
    F --> I[Whiteboard<br/>canvas overlay]
```

1. **Tracking** — MediaPipe's Hand Landmarker finds the hand and regresses 21 keypoints per frame; in video mode it tracks between frames instead of re-detecting.
2. **Normalization** — landmarks are shifted so the wrist is the origin, left hands are mirrored onto right hands, and everything is divided by palm size. The classifier sees the hand's *shape*, not where it is or how far away.
3. **Classification** — either geometry rules (finger extended/curled tests) or a trained MLP with a confidence threshold that returns `none` when unsure.
4. **Debouncing** — a gesture must persist for N frames before it becomes active.
5. **Control** — the cursor follows the **index knuckle** rather than the fingertip, so pinching to click doesn't knock the cursor off target. A One Euro filter smooths the position, heavily when still and lightly when moving fast.

## Repository structure

```text
hand-gesture-control/
├── src/gesture_control/
│   ├── app.py            # Main app: webcam loop, modes, HUD
│   ├── tracking.py       # MediaPipe Hand Landmarker wrapper
│   ├── features.py       # Landmark normalization & finger geometry
│   ├── gestures.py       # Rule-based + learned classifiers, debouncer
│   ├── filters.py        # One Euro filter
│   ├── controllers.py    # Mouse & presentation controllers, swipe detector
│   ├── whiteboard.py     # Air-drawing canvas
│   ├── hud.py            # On-screen overlay & FPS counter
│   ├── dataset.py        # Landmark CSV read/write
│   ├── collect.py        # Record labelled gesture samples
│   ├── train.py          # Train + evaluate classifiers, write report
│   ├── benchmark.py      # Per-stage latency benchmark
│   ├── download.py       # Fetch the MediaPipe model
│   └── config.py         # Paths, gesture names, landmark indices
├── tests/                # Unit tests (synthetic hands, fake input backend)
├── data/                 # Your recorded landmark dataset (gestures.csv)
├── models/               # Downloaded / trained models (git-ignored)
├── reports/              # Generated metrics, confusion matrix, benchmark
├── assets/               # Demo GIFs & images
├── .github/workflows/    # CI: lint + tests
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/haroonkhan02/hand-gesture-control.git
cd hand-gesture-control

# Python 3.10–3.12
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"

gesture-download-model              # fetches the MediaPipe hand model (~8 MB)
gesture-control --mode mouse        # start (uses rule-based gestures until you train)
```

Try it safely first with `--dry-run` (recognizes gestures but doesn't move the mouse). You can also pass a video file: `--source clip.mp4`.

> **macOS:** allow your terminal under *System Settings → Privacy & Security* for **Camera** and **Accessibility** (Accessibility is needed to move the mouse and press keys).

## Train your own gesture classifier

```bash
# 1. Record each gesture (SPACE to start). Run each one 2+ times on different
#    occasions so the evaluation can hold out a whole session.
gesture-collect point --seconds 20
gesture-collect pinch --seconds 20
# ... open_palm, fist, peace, thumbs_up

# 2. Train + evaluate (rules baseline vs. logistic regression vs. MLP)
gesture-train

# 3. Run with the trained model (picked up automatically)
gesture-control

# 4. Measure latency on your machine
gesture-benchmark --frames 300
```

`gesture-train` writes `reports/metrics.md` (accuracy, macro-F1, per-class report, latency) and `reports/confusion_matrix.png`.

**Why a session-level split?** Consecutive video frames are nearly identical. A random split puts near-duplicates of training frames into the test set and inflates accuracy. Holding out each gesture's most recent recording session measures how well the model generalizes to a *new* recording.

## Results

`gesture-train` and `gesture-benchmark` write their results to `reports/` (`metrics.md`, `confusion_matrix.png`, `benchmark.md`).

## Testing

```bash
pytest          # 47 tests, no camera needed
ruff check .
```

## Future work

- Dynamic gestures (e.g. circle to undo) with a small temporal model (GRU/TCN) over landmark sequences
- Two-hand gestures (zoom/rotate)
- Export the classifier to ONNX and run in the browser with MediaPipe JS

## License

[MIT](LICENSE) © Muhammad Haroon Khan
