"""Air-drawing whiteboard.

* point     -> draw with the index fingertip
* peace     -> move without drawing (pen up)
* open_palm -> erase around the palm centre
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import INDEX_TIP, MIDDLE_MCP
from .filters import OneEuroFilter

COLORS = [(255, 80, 80), (80, 220, 80), (80, 160, 255), (0, 220, 255), (255, 255, 255)]


class Whiteboard:
    def __init__(self, width: int, height: int, thickness: int = 6, eraser: int = 45):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.thickness = thickness
        self.eraser = eraser
        self.color_idx = 0
        self.filter = OneEuroFilter(min_cutoff=1.5, beta=0.05)
        self._last: tuple[int, int] | None = None

    @property
    def color(self) -> tuple[int, int, int]:
        return COLORS[self.color_idx]

    def next_color(self) -> None:
        self.color_idx = (self.color_idx + 1) % len(COLORS)

    def clear(self) -> None:
        self.canvas[:] = 0
        self._last = None

    def update(self, gesture: str, pixel_points: np.ndarray | None, t: float) -> None:
        if pixel_points is None or gesture not in ("point", "open_palm"):
            self._last = None
            self.filter.reset()
            return

        if gesture == "open_palm":
            cx, cy = pixel_points[MIDDLE_MCP, :2].astype(int)
            cv2.circle(self.canvas, (int(cx), int(cy)), self.eraser, (0, 0, 0), -1)
            self._last = None
            self.filter.reset()
            return

        x, y = self.filter(pixel_points[INDEX_TIP, :2], t)
        p = (int(x), int(y))
        if self._last is not None:
            cv2.line(self.canvas, self._last, p, self.color, self.thickness, cv2.LINE_AA)
        self._last = p

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        mask = self.canvas.any(axis=2)
        out = frame.copy()
        out[mask] = self.canvas[mask]
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.canvas)
        return path
