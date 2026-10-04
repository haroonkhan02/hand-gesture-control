import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_control.config import NONE
from gesture_control.features import normalize_landmarks
from gesture_control.gestures import GestureDebouncer, LandmarkClassifier, RuleBasedClassifier


@pytest.mark.parametrize("name", ["open_palm", "fist", "point", "peace", "thumbs_up", "pinch"])
def test_rule_classifier(hands, name):
    label, conf = RuleBasedClassifier().predict(hands[name])
    assert label == name
    assert conf == 1.0


def test_rule_classifier_works_for_left_hand(hands):
    left = hands["peace"].copy()
    left[:, 0] = 1000 - left[:, 0]
    assert RuleBasedClassifier().predict(left, "Left")[0] == "peace"


def test_debouncer_ignores_single_frame_glitch():
    d = GestureDebouncer(frames=3)
    for _ in range(3):
        d.update("point")
    assert d.update("pinch") == "point"  # one noisy frame
    assert d.update("point") == "point"


def test_debouncer_switches_after_n_frames():
    d = GestureDebouncer(frames=2)
    assert d.update("fist") == NONE
    assert d.update("fist") == "fist"


def test_debouncer_validates():
    with pytest.raises(ValueError):
        GestureDebouncer(0)


def _fit(hands):
    rng = np.random.default_rng(0)
    X, y = [], []
    for name, pts in hands.items():
        for _ in range(30):
            X.append(normalize_landmarks(pts + rng.normal(0, 2, pts.shape)))
            y.append(name)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return LandmarkClassifier(model, sorted(hands))


def test_landmark_classifier_roundtrip(hands, tmp_path):
    clf = _fit(hands)
    path = tmp_path / "m.joblib"
    clf.save(path)
    loaded = LandmarkClassifier.load(path)
    for name, pts in hands.items():
        assert loaded.predict(pts)[0] == name


def test_landmark_classifier_low_confidence_returns_none(hands):
    clf = _fit(hands)
    clf.min_confidence = 1.01
    assert clf.predict(hands["fist"])[0] == NONE
