import numpy as np
import pytest

from gesture_control.features import (
    FEATURE_DIM,
    finger_states,
    normalize_landmarks,
    palm_size,
    pinch_ratio,
)


def test_feature_shape(hands):
    assert normalize_landmarks(hands["point"]).shape == (FEATURE_DIM,)


def test_translation_invariance(hands):
    p = hands["peace"]
    shifted = p + np.array([123.0, -45.0, 0.0])
    np.testing.assert_allclose(normalize_landmarks(p), normalize_landmarks(shifted), atol=1e-9)


def test_scale_invariance(hands):
    p = hands["open_palm"]
    np.testing.assert_allclose(normalize_landmarks(p), normalize_landmarks(p * 2.5), atol=1e-9)


def test_left_hand_is_mirrored_onto_right(hands):
    right = hands["point"]
    left = right.copy()
    left[:, 0] = 1000.0 - left[:, 0]  # mirror image of the same pose
    np.testing.assert_allclose(normalize_landmarks(right, "Right"),
                               normalize_landmarks(left, "Left"), atol=1e-9)


def test_wrist_is_origin(hands):
    f = normalize_landmarks(hands["fist"]).reshape(21, 3)
    np.testing.assert_allclose(f[0], 0.0)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3)))


def test_palm_size_never_zero():
    assert palm_size(np.zeros((21, 3))) > 0


def test_finger_states(hands):
    assert finger_states(hands["open_palm"]) == dict(index=True, middle=True, ring=True,
                                                     pinky=True, thumb=True)
    assert finger_states(hands["fist"]) == dict(index=False, middle=False, ring=False,
                                                pinky=False, thumb=False)
    s = finger_states(hands["point"])
    assert s["index"] and not (s["middle"] or s["ring"] or s["pinky"])


def test_pinch_ratio(hands):
    assert pinch_ratio(hands["pinch"]) < 0.3
    assert pinch_ratio(hands["open_palm"]) > 0.3
