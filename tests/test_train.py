"""End-to-end check of the dataset -> train -> report pipeline on synthetic data."""

import numpy as np

from gesture_control.dataset import append_samples, load_dataset
from gesture_control.gestures import LandmarkClassifier
from gesture_control.train import split_by_session, train_and_evaluate


def _write(path, hands, sessions=("s1", "s2"), n=25, seed=0):
    rng = np.random.default_rng(seed)
    for session in sessions:
        for name, pts in hands.items():
            samples = [(pts + rng.normal(0, 3, pts.shape), "Right") for _ in range(n)]
            append_samples(path, name, session, samples)


def test_dataset_roundtrip(hands, tmp_path):
    path = tmp_path / "g.csv"
    append_samples(path, "fist", "s1", [(hands["fist"], "Left")])
    ds = load_dataset(path)
    assert len(ds) == 1
    assert ds.labels[0] == "fist" and ds.handedness[0] == "Left"
    np.testing.assert_allclose(ds.points[0], hands["fist"], atol=1e-3)


def test_session_split_holds_out_latest_session(hands, tmp_path):
    path = tmp_path / "g.csv"
    _write(path, hands, sessions=("a", "b"), n=5)
    ds = load_dataset(path)
    train, test, note = split_by_session(ds)
    assert set(ds.sessions[test]) == {"b"} and set(ds.sessions[train]) == {"a"}
    assert "session" in note


def test_single_session_falls_back_to_random_split(hands, tmp_path):
    path = tmp_path / "g.csv"
    _write(path, hands, sessions=("a",), n=10)
    _, test, note = split_by_session(load_dataset(path))
    assert len(test) > 0 and "random" in note


def test_train_and_evaluate(hands, tmp_path):
    data, model, reports = tmp_path / "g.csv", tmp_path / "m.joblib", tmp_path / "reports"
    _write(data, hands)
    out = train_and_evaluate(data, model, reports)
    assert out["results"][out["best"]]["acc"] > 0.95
    assert (reports / "metrics.md").exists() and (reports / "confusion_matrix.png").exists()
    clf = LandmarkClassifier.load(model)
    assert clf.predict(hands["peace"])[0] == "peace"
