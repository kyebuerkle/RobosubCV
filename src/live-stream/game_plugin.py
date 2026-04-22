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
        def on_frame(self, frame, keypoints, frame_w, frame_h) -> np.ndarray:
            ...
            return frame   # modified BGR frame to display

Where `keypoints` is:
    list of instances, each a list of (x, y, conf) tuples
    — already scaled to full display-frame coordinates
    — one instance = one detected hand / body

The game overlay is composited *on top of* the camera frame + pose dots,
so games can draw anything they like on the frame.

Games may also define:
    def on_start(self, frame_w, frame_h): ...   # called once when stream starts
    def on_stop(self): ...                       # called when stream stops

See `game_bounce.py` for a full example.
"""

from __future__ import annotations
import numpy as np


class GamePlugin:
    """Abstract base — inherit and override `on_frame`."""

    def on_start(self, frame_w: int, frame_h: int) -> None:
        """Called once when the stream starts (or game is hot-swapped in)."""

    def on_frame(
        self,
        frame: np.ndarray,
        keypoints: list[list[tuple[float, float, float]]],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """
        Process one frame.

        Parameters
        ----------
        frame      : BGR image (full resolution) — draw on this
        keypoints  : list of hand/body instances; each is a list of (x,y,conf)
        frame_w/h  : display canvas dimensions
        Returns    : the (modified) frame
        """
        return frame

    def on_stop(self) -> None:
        """Called when the stream stops."""
