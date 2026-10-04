"""Record labelled hand-landmark samples from the webcam.

Example:
    gesture-collect point --seconds 20

Press SPACE to start recording after the countdown, Q to quit. Vary distance,
angle, lighting and use both hands while recording - diversity beats quantity.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2

from .config import DATASET_PATH, GESTURES
from .dataset import append_samples
from .hud import draw_hand, draw_panel
from .tracking import HandTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", choices=GESTURES)
    parser.add_argument("--seconds", type=float, default=20.0, help="recording duration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--every", type=int, default=2, help="keep every n-th frame")
    parser.add_argument("--output", default=str(DATASET_PATH))
    args = parser.parse_args()

    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    samples = []
    recording_since: float | None = None
    frame_idx = 0
    t0 = time.perf_counter()
    with HandTracker() as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            hands = tracker.process(frame, int((now - t0) * 1000))
            frame_idx += 1

            if hands:
                draw_hand(frame, hands[0].points)

            if recording_since is None:
                status = "SPACE = start, Q = quit"
            else:
                left = args.seconds - (now - recording_since)
                if left <= 0:
                    break
                status = f"REC {left:4.1f}s  samples: {len(samples)}"
                if hands and frame_idx % args.every == 0:
                    samples.append((hands[0].points, hands[0].handedness))

            draw_panel(frame, [f"gesture: {args.label}", status])
            cv2.imshow("gesture-collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and recording_since is None:
                recording_since = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()
    if samples:
        append_samples(args.output, args.label, session, samples)
        print(f"Saved {len(samples)} '{args.label}' samples (session {session}) to {args.output}")
    else:
        print("No samples recorded.")


if __name__ == "__main__":
    main()
