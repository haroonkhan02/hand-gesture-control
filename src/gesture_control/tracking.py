"""MediaPipe hand-landmark tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import HAND_MODEL_PATH


@dataclass
class Hand:
    norm: np.ndarray  # (21, 3) x, y in [0, 1] of the image; z relative depth
    points: np.ndarray  # (21, 3) same landmarks in pixel units (z scaled like x)
    handedness: str  # "Left" or "Right"
    score: float


class HandTracker:
    """Thin wrapper around MediaPipe's HandLandmarker task.

    ``video=True`` enables temporal tracking between frames (faster and steadier than
    running palm detection on every frame); timestamps must then increase.
    """

    def __init__(self, model_path: str | Path = HAND_MODEL_PATH, max_hands: int = 1,
                 video: bool = True, min_detection: float = 0.6, min_tracking: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Run `gesture-download-model` first."
            )
        self._mp = mp
        self.video = video
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self.video:
            ts = int(timestamp_ms if timestamp_ms is not None else self._last_ts + 33)
            ts = max(ts, self._last_ts + 1)  # MediaPipe requires strictly increasing
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness, strict=True):
            norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
            points = norm * np.array([w, h, w], dtype=float)
            hands.append(Hand(norm, points, handed[0].category_name, float(handed[0].score)))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
