"""Read/write the landmark dataset recorded with ``gesture-collect``.

Raw pixel-space landmarks are stored (not features), so feature engineering can
change later without re-recording. Each row also stores a ``session`` id: frames
from one recording session are near-duplicates, so the train/test split is done
by session to get an honest accuracy estimate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import NUM_LANDMARKS

COORD_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
HEADER = ["label", "handedness", "session", *COORD_COLUMNS]


@dataclass
class LandmarkDataset:
    points: np.ndarray  # (N, 21, 3)
    labels: np.ndarray  # (N,) str
    handedness: np.ndarray  # (N,) str
    sessions: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return len(self.labels)


def append_samples(path: str | Path, label: str, session: str,
                   samples: list[tuple[np.ndarray, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        for points, handedness in samples:
            flat = np.asarray(points, dtype=float).reshape(-1)
            writer.writerow([label, handedness, session, *(f"{v:.3f}" for v in flat)])


def load_dataset(path: str | Path) -> LandmarkDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Record one with `gesture-collect`.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    points = np.array([[float(r[c]) for c in COORD_COLUMNS] for r in rows]).reshape(
        -1, NUM_LANDMARKS, 3
    )
    return LandmarkDataset(
        points=points,
        labels=np.array([r["label"] for r in rows]),
        handedness=np.array([r["handedness"] for r in rows]),
        sessions=np.array([r["session"] for r in rows]),
    )
