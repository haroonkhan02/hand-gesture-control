"""Turn raw hand landmarks into features a classifier can learn from.

Raw landmarks depend on where the hand is in the frame, how far it is from the
camera and which hand it is. ``normalize_landmarks`` removes those nuisance
factors so the classifier only sees the *shape* of the hand.
"""

from __future__ import annotations

import numpy as np

from .config import (
    FINGER_JOINTS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    PINKY_MCP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
)

FEATURE_VERSION = 1
FEATURE_DIM = NUM_LANDMARKS * 3


def _check(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (21, 3), got {points.shape}")
    return points


def palm_size(points: np.ndarray) -> float:
    """Wrist-to-middle-knuckle distance (2D). Used as the hand's unit of length."""
    points = _check(points)
    size = float(np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2]))
    return max(size, 1e-6)


def normalize_landmarks(points: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Return a 63-d feature vector invariant to translation, scale and handedness.

    1. Translate so the wrist is at the origin.
    2. Mirror left hands onto right hands (x -> -x) so one model serves both.
    3. Divide by palm size so near/far hands look the same.

    Rotation is deliberately *not* removed: "thumbs up" vs. "fist" differ mainly by
    orientation.
    """
    points = _check(points)
    centred = points - points[WRIST]
    if handedness.lower().startswith("l"):
        centred[:, 0] *= -1.0
    return (centred / palm_size(points)).reshape(-1)


def _dist(points: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(points[i, :2] - points[j, :2]))


def finger_states(points: np.ndarray) -> dict[str, bool]:
    """Heuristic extended/curled state for each finger (2D geometry only).

    A finger counts as extended when its tip is farther from the wrist than its
    PIP joint. The thumb is extended when its tip is far from the index knuckle.
    """
    points = _check(points)
    size = palm_size(points)
    names = ("index", "middle", "ring", "pinky")
    states = {
        name: _dist(points, WRIST, tip) > _dist(points, WRIST, pip)
        for name, (pip, tip) in zip(names, FINGER_JOINTS, strict=True)
    }
    thumb_reach = _dist(points, THUMB_TIP, INDEX_MCP) / size
    thumb_out = _dist(points, THUMB_TIP, PINKY_MCP) > _dist(points, THUMB_IP, PINKY_MCP)
    states["thumb"] = thumb_reach > 0.5 and thumb_out
    return states


def pinch_ratio(points: np.ndarray) -> float:
    """Thumb-tip to index-tip distance in palm units (small = pinching)."""
    points = _check(points)
    return _dist(points, THUMB_TIP, INDEX_TIP) / palm_size(points)


def index_reach(points: np.ndarray) -> float:
    """Wrist to index-tip distance in palm units (large = index pointing out)."""
    points = _check(points)
    return _dist(points, WRIST, INDEX_TIP) / palm_size(points)
