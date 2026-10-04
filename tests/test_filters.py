import numpy as np
import pytest

from gesture_control.filters import OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    np.testing.assert_allclose(f([3.0, 4.0], 0.0), [3.0, 4.0])


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    for i in range(50):
        out = f(5.0, i / 30)
    assert out == pytest.approx(5.0)


def test_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    noisy = 100 + rng.normal(0, 3, 300)
    smooth = np.array([f(x, i / 30) for i, x in enumerate(noisy)]).ravel()
    assert smooth[50:].std() < noisy[50:].std() / 2


def test_higher_beta_means_less_lag():
    t = np.arange(60) / 30
    ramp = 1000 * t  # fast linear motion
    lag = {}
    for beta in (0.0, 0.1):
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        out = [f(x, ti) for x, ti in zip(ramp, t, strict=True)]
        lag[beta] = ramp[-1] - float(out[-1])
    assert lag[0.1] < lag[0.0]


def test_non_increasing_timestamp_is_ignored():
    f = OneEuroFilter()
    f(1.0, 1.0)
    assert f(50.0, 1.0) == pytest.approx(1.0)


def test_reset():
    f = OneEuroFilter()
    f(1.0, 0.0)
    f.reset()
    assert f(9.0, 1.0) == pytest.approx(9.0)


def test_invalid_cutoff():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0)
