import numpy as np

from gesture_control.controllers import (
    HandState,
    MouseController,
    PresentationController,
    SwipeDetector,
)


class FakeBackend:
    def __init__(self):
        self.events = []

    def screen_size(self):
        return (1000, 500)

    def move(self, x, y):
        self.events.append(("move", round(x), round(y)))

    def press(self, button="left"):
        self.events.append(("press", button))

    def release(self, button="left"):
        self.events.append(("release", button))

    def click(self, button="left"):
        self.events.append(("click", button))

    def scroll(self, dy):
        self.events.append(("scroll", dy))

    def key(self, name):
        self.events.append(("key", name))

    def kinds(self):
        return [e[0] for e in self.events]


def hand_at(x, y):
    pts = np.zeros((21, 3))
    pts[:, 0], pts[:, 1] = x, y
    return pts


def test_to_screen_maps_inner_box_to_full_screen():
    m = MouseController(FakeBackend(), margin=0.2)
    assert m.to_screen(0.2, 0.2) == (0.0, 0.0)
    assert m.to_screen(0.8, 0.8) == (999.0, 499.0)
    assert m.to_screen(0.0, 1.0) == (0.0, 499.0)  # clamped


def test_point_moves_cursor_and_fist_freezes_it():
    b = FakeBackend()
    m = MouseController(b)
    m.update(HandState("point", hand_at(0.5, 0.5), 0.0))
    m.update(HandState("fist", hand_at(0.7, 0.7), 0.1))
    assert b.kinds() == ["move"]


def test_pinch_press_and_release():
    b = FakeBackend()
    m = MouseController(b)
    m.update(HandState("pinch", hand_at(0.5, 0.5), 0.0))
    m.update(HandState("pinch", hand_at(0.5, 0.5), 0.03))
    m.update(HandState("point", hand_at(0.5, 0.5), 0.06))
    kinds = b.kinds()
    assert kinds.count("press") == 1 and kinds.count("release") == 1


def test_losing_hand_releases_button():
    b = FakeBackend()
    m = MouseController(b)
    m.update(HandState("pinch", hand_at(0.5, 0.5), 0.0))
    m.update(HandState("pinch", None, 0.03))
    assert ("release", "left") in b.events


def test_close_releases_button():
    b = FakeBackend()
    m = MouseController(b)
    m.update(HandState("pinch", hand_at(0.5, 0.5), 0.0))
    m.close()
    assert b.events[-1] == ("release", "left")


def test_thumbs_up_right_clicks_once():
    b = FakeBackend()
    m = MouseController(b)
    for i in range(5):
        m.update(HandState("thumbs_up", hand_at(0.5, 0.5), i * 0.03))
    assert b.events.count(("click", "right")) == 1


def test_peace_scrolls_up_when_hand_moves_up():
    b = FakeBackend()
    m = MouseController(b, scroll_gain=40)
    m.update(HandState("peace", hand_at(0.5, 0.6), 0.0))
    m.update(HandState("peace", hand_at(0.5, 0.5), 0.03))
    assert b.events == [("scroll", 4)]


def test_swipe_detector():
    s = SwipeDetector(window=0.5, distance=0.25, cooldown=1.0)
    assert s.update(0.7, 0.0) is None
    assert s.update(0.55, 0.1) is None
    assert s.update(0.4, 0.2) == "left"
    assert s.update(0.1, 0.3) is None  # cooldown


def test_slow_drift_is_not_a_swipe():
    s = SwipeDetector(window=0.4, distance=0.25)
    out = [s.update(0.8 - 0.01 * i, i * 0.1) for i in range(40)]
    assert all(o is None for o in out)


def test_presentation_swipe_left_goes_next():
    b = FakeBackend()
    p = PresentationController(b)
    for i, x in enumerate([0.7, 0.55, 0.4]):
        p.update(HandState("open_palm", hand_at(x, 0.5), i * 0.1))
    assert b.events == [("key", "right")]
    assert p.last_action == "next"


def test_presentation_fist_blanks_once():
    b = FakeBackend()
    p = PresentationController(b)
    for i in range(4):
        p.update(HandState("fist", hand_at(0.5, 0.5), i * 0.1))
    assert b.events == [("key", "b")]
