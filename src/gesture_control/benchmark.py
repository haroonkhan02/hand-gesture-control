"""Measure per-stage latency and end-to-end FPS on a webcam or video file.

Example:
    gesture-benchmark --source 0 --frames 300
    gesture-benchmark --source demo.mp4
"""

from __future__ import annotations

import argparse
import platform
import time

import cv2
import numpy as np

from .config import CLASSIFIER_PATH, REPORTS_DIR
from .gestures import LandmarkClassifier, RuleBasedClassifier
from .tracking import HandTracker


def summarize(ms: list[float]) -> str:
    if not ms:
        return "n/a | n/a"
    a = np.asarray(ms)
    return f"{a.mean():.2f} | {np.percentile(a, 95):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")

    classifiers = [RuleBasedClassifier()]
    if CLASSIFIER_PATH.exists():
        classifiers.append(LandmarkClassifier.load(CLASSIFIER_PATH))

    track_ms, total_ms = [], []
    cls_ms: dict[str, list[float]] = {c.name: [] for c in classifiers}
    detected = 0
    with HandTracker() as tracker:
        for i in range(args.frames + args.warmup):
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            hands = tracker.process(cv2.flip(frame, 1), int(i * 33))
            mid = time.perf_counter()
            for c in classifiers:
                if hands:
                    s = time.perf_counter()
                    c.predict(hands[0].points, hands[0].handedness)
                    if i >= args.warmup:
                        cls_ms[c.name].append((time.perf_counter() - s) * 1000)
            end = time.perf_counter()
            if i >= args.warmup:
                track_ms.append((mid - start) * 1000)
                total_ms.append((end - start) * 1000)
                detected += bool(hands)
    cap.release()

    n = len(total_ms)
    if n == 0:
        raise SystemExit("No frames processed")
    lines = [
        "# Latency benchmark",
        "",
        f"- Machine: {platform.machine()} / {platform.system()} {platform.release()}",
        f"- Frames: {n} (after {args.warmup} warm-up), hand detected in {detected / n:.0%}",
        f"- Processing throughput: {1000 / np.mean(total_ms):.0f} FPS "
        "(excluding camera capture and display)",
        "",
        "| Stage | Mean (ms) | p95 (ms) |",
        "|---|---|---|",
        f"| Hand tracking (MediaPipe) | {summarize(track_ms)} |",
        *(f"| Classifier: {name} | {summarize(v)} |" for name, v in cls_ms.items()),
        f"| Total per frame | {summarize(total_ms)} |",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
