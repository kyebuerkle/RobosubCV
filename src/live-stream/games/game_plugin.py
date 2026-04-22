#!/usr/bin/env python3
"""
game_plugin.py — Base class for streamer game plugins
------------------------------------------------------
Every game script must define a class called `Game` that inherits from
`GamePlugin`.  The streamer will import the script, instantiate Game(),
and call its methods each frame.

API contract
------------
    class Game(GamePlugin):
        def on_frame(self, frame, keypoints, detections, frame_w, frame_h) -> np.ndarray:
            ...
            return frame

Where `keypoints` is:
    list of instances, each a list of (x, y, conf) tuples

Where `detections` is:
    list of (label, conf, x1, y1, x2, y2) tuples

Pause / reset
-------------
The streamer calls game.toggle_pause() when SPACE is pressed.
GamePlugin.toggle_pause() handles the overlay automatically — subclasses
just implement reset() if they want a reset option in the pause menu.

Games that support reset should override:
    def reset(self): ...

Games may also define:
    def on_start(self, frame_w, frame_h, class_names): ...
    def on_stop(self): ...
"""

from __future__ import annotations
import time
import cv2
import numpy as np


class GamePlugin:

    def __init__(self):
        self._paused      = False
        self._pause_ts    = 0.0   # when pause started (for animation)

    # ── Pause (handled here so all games get it for free) ────

    def toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_ts = time.perf_counter()

    def reset(self):
        """Override in subclass to support reset from pause menu."""

    # ── Internal: draw pause overlay onto frame ──────────────

    def _draw_pause_overlay(self, frame: "np.ndarray") -> "np.ndarray":
        now   = time.perf_counter()
        t     = now - self._pause_ts
        alpha = min(0.62, t * 3.0)          # fade in over ~0.2 s
        h, w  = frame.shape[:2]
        FONT  = cv2.FONT_HERSHEY_DUPLEX
        FONTS = cv2.FONT_HERSHEY_SIMPLEX

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (10, 8, 22), -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

        # Title
        title = "PAUSED"
        (tw, th), _ = cv2.getTextSize(title, FONT, 2.2, 4)
        tx, ty = w // 2 - tw // 2, h // 2 - 40
        cv2.putText(frame, title, (tx+2, ty+2), FONT, 2.2, (0,0,0),      5, cv2.LINE_AA)
        cv2.putText(frame, title, (tx,   ty  ), FONT, 2.2, (200,200,255), 4, cv2.LINE_AA)

        # SPACE to resume
        sub = "SPACE  resume"
        (sw, _), _ = cv2.getTextSize(sub, FONTS, 0.7, 1)
        cv2.putText(frame, sub, (w//2 - sw//2, h//2 + 10), FONTS, 0.7, (150,150,200), 1, cv2.LINE_AA)

        # Reset option (only if subclass overrides reset)
        if type(self).reset is not GamePlugin.reset:
            rst = "R  reset"
            (rw, _), _ = cv2.getTextSize(rst, FONTS, 0.7, 1)
            cv2.putText(frame, rst, (w//2 - rw//2, h//2 + 38), FONTS, 0.7, (150,200,150), 1, cv2.LINE_AA)

        return frame

    # ── Wrapper called by streamer each frame ────────────────

    def _tick(self, frame, keypoints, detections, fw, fh):
        """
        Called by streamer instead of on_frame directly.
        Handles pause overlay; delegates to on_frame when not paused.
        """
        if self._paused:
            return self._draw_pause_overlay(frame)
        return self.on_frame(frame, keypoints, detections, fw, fh)

    # ── Subclass overrides ────────────────────────────────────

    def on_start(self, frame_w: int, frame_h: int, class_names=None) -> None:
        pass

    def on_frame(self, frame, keypoints, detections, frame_w, frame_h):
        return frame

    def on_stop(self) -> None:
        pass
