import numpy as np

from gesture_control.whiteboard import Whiteboard


def _hand(x, y):
    pts = np.zeros((21, 3))
    pts[:, 0], pts[:, 1] = x, y
    return pts


def test_point_draws_a_stroke():
    wb = Whiteboard(200, 100)
    wb.update("point", _hand(20, 50), 0.0)
    wb.update("point", _hand(180, 50), 0.5)
    assert wb.canvas.any()


def test_pen_up_does_not_draw():
    wb = Whiteboard(200, 100)
    wb.update("peace", _hand(20, 50), 0.0)
    wb.update("peace", _hand(180, 50), 0.5)
    assert not wb.canvas.any()


def test_open_palm_erases():
    wb = Whiteboard(200, 100, eraser=200)
    wb.update("point", _hand(20, 50), 0.0)
    wb.update("point", _hand(180, 50), 0.5)
    wb.update("open_palm", _hand(100, 50), 1.0)
    assert not wb.canvas.any()


def test_overlay_and_clear():
    wb = Whiteboard(50, 50)
    wb.canvas[10, 10] = (255, 0, 0)
    frame = np.full((50, 50, 3), 7, np.uint8)
    out = wb.overlay(frame)
    assert tuple(out[10, 10]) == (255, 0, 0) and out[0, 0, 0] == 7
    wb.clear()
    assert not wb.canvas.any()
