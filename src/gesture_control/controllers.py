"""Map gestures to computer actions.

Controllers talk to the OS only through an ``InputBackend``, so all the logic can be
unit-tested with a fake backend and no real mouse/keyboard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import INDEX_MCP, NONE
from .filters import OneEuroFilter


class InputBackend(Protocol):
    def screen_size(self) -> tuple[int, int]: ...
    def move(self, x: float, y: float) -> None: ...
    def press(self, button: str = "left") -> None: ...
    def release(self, button: str = "left") -> None: ...
    def click(self, button: str = "left") -> None: ...
    def scroll(self, dy: int) -> None: ...
    def key(self, name: str) -> None: ...


class PynputBackend:
    """Real mouse/keyboard via pynput (needs Accessibility permission on macOS)."""

    def __init__(self, screen: tuple[int, int] | None = None):
        from pynput import keyboard, mouse  # imported lazily: needs a display

        self._mouse = mouse.Controller()
        self._keyboard = keyboard.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right}
        self._keys = keyboard.Key
        self._screen = screen or _detect_screen_size()

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def move(self, x: float, y: float) -> None:
        self._mouse.position = (int(x), int(y))

    def press(self, button: str = "left") -> None:
        self._mouse.press(self._buttons[button])

    def release(self, button: str = "left") -> None:
        self._mouse.release(self._buttons[button])

    def click(self, button: str = "left") -> None:
        self._mouse.click(self._buttons[button])

    def scroll(self, dy: int) -> None:
        self._mouse.scroll(0, dy)

    def key(self, name: str) -> None:
        k = getattr(self._keys, name, name)
        self._keyboard.press(k)
        self._keyboard.release(k)


def _detect_screen_size() -> tuple[int, int]:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return (1920, 1080)


@dataclass
class HandState:
    """What a controller needs from one frame."""

    gesture: str
    norm_points: np.ndarray | None  # (21, 3) in [0, 1] image coordinates, None if no hand
    t: float


class MouseController:
    """Gesture mouse.

    * point      -> move cursor
    * pinch      -> hold left button (pinch + move = drag, quick pinch = click)
    * peace      -> scroll by moving the hand up/down
    * thumbs_up  -> right click
    * anything else (fist, open palm, no hand) -> cursor frozen ("clutch")

    The cursor follows the index *knuckle*, not the fingertip: the knuckle barely
    moves when you pinch, so clicking doesn't knock the cursor off target.
    """

    MOVE_GESTURES = ("point", "pinch")

    def __init__(self, backend: InputBackend, margin: float = 0.2, scroll_gain: float = 40.0,
                 min_cutoff: float = 1.0, beta: float = 0.02):
        self.backend = backend
        self.margin = margin
        self.scroll_gain = scroll_gain
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.prev_gesture = NONE
        self.pressed = False
        self._scroll_anchor: float | None = None

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        """Map the inner [margin, 1 - margin] box of the frame to the full screen."""
        w, h = self.backend.screen_size()
        span = 1.0 - 2.0 * self.margin
        sx = np.clip((nx - self.margin) / span, 0.0, 1.0) * (w - 1)
        sy = np.clip((ny - self.margin) / span, 0.0, 1.0) * (h - 1)
        return float(sx), float(sy)

    def update(self, state: HandState) -> None:
        g = state.gesture if state.norm_points is not None else NONE

        if g in self.MOVE_GESTURES:
            anchor = state.norm_points[INDEX_MCP, :2]
            x, y = self.filter(self.to_screen(*anchor), state.t)
            self.backend.move(x, y)
        else:
            self.filter.reset()

        if g == "pinch" and not self.pressed:
            self.backend.press("left")
            self.pressed = True
        elif g != "pinch" and self.pressed:
            self.backend.release("left")
            self.pressed = False

        if g == "thumbs_up" and self.prev_gesture != "thumbs_up":
            self.backend.click("right")

        if g == "peace":
            y = float(state.norm_points[INDEX_MCP, 1])
            if self._scroll_anchor is not None:
                steps = round((self._scroll_anchor - y) * self.scroll_gain)
                if steps:
                    self.backend.scroll(steps)
                    self._scroll_anchor = y
            else:
                self._scroll_anchor = y
        else:
            self._scroll_anchor = None

        self.prev_gesture = g

    def close(self) -> None:
        """Never leave the mouse button stuck down."""
        if self.pressed:
            self.backend.release("left")
            self.pressed = False


class SwipeDetector:
    """Detect horizontal swipes from a short history of x positions."""

    def __init__(self, window: float = 0.4, distance: float = 0.25, cooldown: float = 0.8):
        self.window = window
        self.distance = distance
        self.cooldown = cooldown
        self._history: deque[tuple[float, float]] = deque()
        self._last_fire = -float("inf")

    def reset(self) -> None:
        self._history.clear()

    def update(self, x: float, t: float) -> str | None:
        self._history.append((t, x))
        while self._history and t - self._history[0][0] > self.window:
            self._history.popleft()
        if t - self._last_fire < self.cooldown or len(self._history) < 2:
            return None
        dx = x - self._history[0][1]
        if abs(dx) >= self.distance:
            self._last_fire = t
            self._history.clear()
            return "right" if dx > 0 else "left"
        return None


class PresentationController:
    """Slide control.

    * open palm swipe left  -> next slide
    * open palm swipe right -> previous slide
    * fist                  -> toggle black screen ("b" in PowerPoint/Keynote)
    """

    def __init__(self, backend: InputBackend, swipe: SwipeDetector | None = None):
        self.backend = backend
        self.swipe = swipe or SwipeDetector()
        self.prev_gesture = NONE
        self.last_action: str | None = None

    def update(self, state: HandState) -> None:
        g = state.gesture if state.norm_points is not None else NONE
        if g == "open_palm":
            direction = self.swipe.update(float(state.norm_points[0, 0]), state.t)
            if direction == "left":
                self.backend.key("right")
                self.last_action = "next"
            elif direction == "right":
                self.backend.key("left")
                self.last_action = "previous"
        else:
            self.swipe.reset()

        if g == "fist" and self.prev_gesture != "fist":
            self.backend.key("b")
            self.last_action = "blank"
        self.prev_gesture = g

    def close(self) -> None:
        pass
