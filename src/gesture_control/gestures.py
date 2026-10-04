"""Gesture classifiers and temporal debouncing.

Two classifiers share one interface (``predict(points, handedness) -> (label, conf)``):

* ``RuleBasedClassifier`` - hand-written geometry rules. Works out of the box and
  serves as the baseline the learned model has to beat.
* ``LandmarkClassifier`` - a scikit-learn model trained on your own recordings
  (see ``gesture-collect`` and ``gesture-train``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from .config import NONE, THUMB_TIP
from .features import (
    FEATURE_VERSION,
    finger_states,
    index_reach,
    normalize_landmarks,
    pinch_ratio,
)


class GestureClassifier(Protocol):
    name: str

    def predict(self, points: np.ndarray, handedness: str = "Right") -> tuple[str, float]: ...


class RuleBasedClassifier:
    """Finger-state rules. Confidence is 1.0 for a matched rule, 0.0 otherwise."""

    name = "rules"

    def __init__(self, pinch_threshold: float = 0.3, pinch_min_reach: float = 1.1):
        self.pinch_threshold = pinch_threshold
        self.pinch_min_reach = pinch_min_reach

    def predict(self, points: np.ndarray, handedness: str = "Right") -> tuple[str, float]:
        points = np.asarray(points, dtype=float)
        pinching = pinch_ratio(points) < self.pinch_threshold
        if pinching and index_reach(points) > self.pinch_min_reach:
            return "pinch", 1.0

        s = finger_states(points)
        fingers = [s["index"], s["middle"], s["ring"], s["pinky"]]
        n_up = sum(fingers)

        if n_up >= 4 or (n_up == 3 and s["thumb"]):
            return "open_palm", 1.0
        if n_up == 0:
            thumb_highest = points[THUMB_TIP, 1] < np.delete(points[:, 1], THUMB_TIP).min()
            if s["thumb"] and thumb_highest:
                return "thumbs_up", 1.0
            return "fist", 1.0
        if fingers == [True, False, False, False]:
            return "point", 1.0
        if fingers == [True, True, False, False]:
            return "peace", 1.0
        return NONE, 0.0


class LandmarkClassifier:
    """Wraps a trained scikit-learn pipeline saved by ``gesture-train``."""

    name = "mlp"

    def __init__(self, model, labels: list[str], min_confidence: float = 0.6):
        self.model = model
        self.labels = list(labels)
        self.min_confidence = min_confidence

    @classmethod
    def load(cls, path: str | Path, min_confidence: float = 0.6) -> LandmarkClassifier:
        bundle = joblib.load(path)
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"{path} was trained with feature version {bundle.get('feature_version')}, "
                f"expected {FEATURE_VERSION}. Re-run gesture-train."
            )
        return cls(bundle["model"], bundle["labels"], min_confidence)

    def save(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "labels": self.labels, "feature_version": FEATURE_VERSION,
             **metadata},
            path,
        )

    def predict(self, points: np.ndarray, handedness: str = "Right") -> tuple[str, float]:
        x = normalize_landmarks(points, handedness)[None, :]
        proba = self.model.predict_proba(x)[0]
        best = int(np.argmax(proba))
        conf = float(proba[best])
        label = str(self.model.classes_[best])
        return (label, conf) if conf >= self.min_confidence else (NONE, conf)


class GestureDebouncer:
    """Only switch gesture after it has been seen ``frames`` times in a row.

    Single-frame misclassifications would otherwise fire spurious clicks.
    """

    def __init__(self, frames: int = 3):
        if frames < 1:
            raise ValueError("frames must be >= 1")
        self.frames = frames
        self.reset()

    def reset(self) -> None:
        self.current = NONE
        self._candidate = NONE
        self._count = 0

    def update(self, label: str) -> str:
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count >= self.frames:
            self.current = self._candidate
        return self.current
