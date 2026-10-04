"""Run gesture control from the webcam.

Modes (switch with M): mouse, slides, draw.
Keys: M mode, P pause, C clear canvas, K next colour, S save canvas, Q quit.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2

from .config import CLASSIFIER_PATH, NONE, OUTPUTS_DIR
from .controllers import HandState, MouseController, PresentationController, PynputBackend
from .gestures import GestureDebouncer, LandmarkClassifier, RuleBasedClassifier
from .hud import FPSCounter, draw_active_region, draw_hand, draw_panel
from .tracking import HandTracker
from .whiteboard import Whiteboard

MODES = ("mouse", "slides", "draw")


def load_classifier(kind: str, min_confidence: float):
    if kind in ("auto", "mlp") and CLASSIFIER_PATH.exists():
        return LandmarkClassifier.load(CLASSIFIER_PATH, min_confidence)
    if kind == "mlp":
        raise SystemExit(f"No trained model at {CLASSIFIER_PATH}. Run `gesture-train` first.")
    return RuleBasedClassifier()


def parse_screen(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="mouse")
    parser.add_argument("--source", default="0", help="camera index or path to a video file")
    parser.add_argument("--classifier", choices=("auto", "rules", "mlp"), default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--debounce", type=int, default=3, help="frames a gesture must persist")
    parser.add_argument("--margin", type=float, default=0.2, help="dead border of the frame")
    parser.add_argument("--screen", help="override screen size, e.g. 1512x982")
    parser.add_argument("--dry-run", action="store_true",
                        help="recognise gestures but don't move the mouse / press keys")
    args = parser.parse_args()

    classifier = load_classifier(args.classifier, args.min_confidence)
    print(f"Classifier: {classifier.name}")
    debouncer = GestureDebouncer(args.debounce)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera returned no frames")
    h, w = frame.shape[:2]

    backend = None if args.dry_run else PynputBackend(parse_screen(args.screen))
    controllers = {}
    if backend is not None:
        controllers = {"mouse": MouseController(backend, margin=args.margin),
                       "slides": PresentationController(backend)}
    board = Whiteboard(w, h)
    mode, paused = args.mode, False
    fps = FPSCounter()
    t0 = time.perf_counter()

    try:
        with HandTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror view feels natural
                t = time.perf_counter() - t0
                hands = tracker.process(frame, int(t * 1000))

                raw, conf = NONE, 0.0
                hand = hands[0] if hands else None
                if hand is not None:
                    raw, conf = classifier.predict(hand.points, hand.handedness)
                    draw_hand(frame, hand.points)
                gesture = debouncer.update(raw)

                state = HandState(gesture, hand.norm if hand else None, t)
                if not paused:
                    if mode == "draw":
                        board.update(gesture, hand.points if hand else None, t)
                    elif mode in controllers:
                        controllers[mode].update(state)

                if mode == "draw":
                    frame = board.overlay(frame)
                elif mode == "mouse":
                    draw_active_region(frame, args.margin)

                draw_panel(frame, [
                    f"mode: {mode}{'  [PAUSED]' if paused else ''}",
                    f"gesture: {gesture}  ({raw} {conf:.2f})",
                    f"fps: {fps.tick():.0f}   classifier: {classifier.name}",
                ])
                cv2.imshow("hand-gesture-control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    for c in controllers.values():
                        c.close()
                    mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                elif key == ord("p"):
                    paused = not paused
                    for c in controllers.values():
                        c.close()
                elif key == ord("c"):
                    board.clear()
                elif key == ord("k"):
                    board.next_color()
                elif key == ord("s"):
                    path = board.save(OUTPUTS_DIR / f"drawing-{datetime.now():%Y%m%d-%H%M%S}.png")
                    print(f"Saved {path}")
    finally:
        for c in controllers.values():
            c.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
