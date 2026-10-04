"""Synthetic hand poses for testing geometry without a camera.

Coordinates mimic MediaPipe pixel output for a right hand, palm facing the camera,
fingers pointing up (image y grows downwards).
"""

from __future__ import annotations

import numpy as np
import pytest

WRIST = (320.0, 400.0)
MCPS = {"index": (290.0, 310.0), "middle": (315.0, 305.0), "ring": (340.0, 310.0),
        "pinky": (362.0, 322.0)}
FINGER_START = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

EXTENDED = [(0, 0), (0, -30), (0, -50), (0, -65)]  # mcp, pip, dip, tip offsets
CURLED = [(0, 0), (0, -20), (0, -5), (0, 10)]  # tip folds back toward the palm

THUMB_EXTENDED = [(300, 380), (275, 365), (250, 345), (225, 330)]
THUMB_CURLED = [(300, 380), (290, 355), (305, 340), (318, 335)]
THUMB_UP = [(300, 380), (290, 340), (285, 300), (282, 250)]


def make_hand(index=False, middle=False, ring=False, pinky=False, thumb="curled",
              pinch=False) -> np.ndarray:
    pts = np.zeros((21, 3))
    pts[0, :2] = WRIST
    thumb_pts = {"curled": THUMB_CURLED, "extended": THUMB_EXTENDED, "up": THUMB_UP}[thumb]
    pts[1:5, :2] = thumb_pts
    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, start in FINGER_START.items():
        offsets = EXTENDED if states[name] else CURLED
        base = np.array(MCPS[name])
        for k, off in enumerate(offsets):
            pts[start + k, :2] = base + off
    if pinch:
        # index half-bent forward, thumb tip touching the index tip
        pts[6, :2], pts[7, :2], pts[8, :2] = (285, 285), (278, 270), (272, 262)
        pts[4, :2] = (268, 266)
        pts[3, :2] = (275, 300)
    return pts


@pytest.fixture
def hands() -> dict[str, np.ndarray]:
    return {
        "open_palm": make_hand(True, True, True, True, thumb="extended"),
        "fist": make_hand(thumb="curled"),
        "point": make_hand(index=True),
        "peace": make_hand(index=True, middle=True),
        "thumbs_up": make_hand(thumb="up"),
        "pinch": make_hand(middle=True, ring=True, pinky=True, pinch=True),
    }
