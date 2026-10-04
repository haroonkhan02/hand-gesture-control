"""Signal smoothing for jitter-free cursor control.

Implements the One Euro filter (Casiez et al., CHI 2012): a low-pass filter whose
cutoff frequency rises with speed. Slow movements get heavy smoothing (no jitter),
fast movements get light smoothing (no lag).
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter for a scalar or a fixed-size vector signal.

    Args:
        min_cutoff: Cutoff (Hz) at zero speed. Lower = smoother when still.
        beta: Speed coefficient. Higher = less lag on fast motion.
        d_cutoff: Cutoff (Hz) used to smooth the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()

        dt = t - self._t
        if dt <= 0:  # duplicate/out-of-order timestamp: keep previous estimate
            return self._x.copy()

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = np.array([_alpha(c, dt) for c in np.atleast_1d(cutoff)]).reshape(x.shape)
        x_hat = a * x + (1 - a) * self._x

        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()
