"""On-screen overlay: skeleton, gesture label, mode and FPS."""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from .config import HAND_CONNECTIONS

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hand(frame: np.ndarray, points: np.ndarray, color=(0, 255, 160)) -> None:
    pts = points[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (230, 230, 230), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, color, -1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str], origin=(12, 12)) -> None:
    x, y = origin
    width = max(cv2.getTextSize(t, FONT, 0.6, 1)[0][0] for t in lines) + 20
    height = 26 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + 10, y + 26 * (i + 1)), FONT, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)


def draw_active_region(frame: np.ndarray, margin: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * margin), int(h * margin)),
                  (int(w * (1 - margin)), int(h * (1 - margin))), (255, 200, 0), 1)


class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
