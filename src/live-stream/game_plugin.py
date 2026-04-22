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
            return frame   # modified BGR frame to display

Where `keypoints` is:
    list of instances, each a list of (x, y, conf) tuples
    — already scaled to full display-frame coordinates
    — one instance = one detected hand / body
    — empty list if the model has no pose output

Where `detections` is:
    list of (label, conf, x1, y1, x2, y2) tuples
    — same data shown as bounding boxes on screen
    — works with any model, pose or detection

The game overlay is composited *on top of* the camera frame + pose dots,
so games can draw anything they like on the frame.

Games may also define:
    def on_start(self, frame_w, frame_h, class_names): ...
        # called once when stream starts or game is hot-swapped in
        # class_names: list[str] of all labels the model knows
    def on_stop(self): ...

See game_bounce.py and game_showme.py for examples.
"""

from __future__ import annotations
import numpy as np


class GamePlugin:
    """Abstract base — inherit and override on_frame."""

    def on_start(
        self,
        frame_w: int,
        frame_h: int,
        class_names: list[str] | None = None,
    ) -> None:
        """Called once when the stream starts (or game is hot-swapped in).

        Parameters
        ----------
        frame_w/h   : display canvas size in pixels
        class_names : all label strings the loaded model can output.
                      May be None if no model is loaded yet.
        """

    def on_frame(
        self,
        frame: "np.ndarray",
        keypoints: "list[list[tuple[float, float, float]]]",
        detections: "list[tuple[str, float, int, int, int, int]]",
        frame_w: int,
        frame_h: int,
    ) -> "np.ndarray":
        """
        Process one frame.

        Parameters
        ----------
        frame      : BGR image (full resolution) — draw on this
        keypoints  : pose instances; each is [(x, y, conf), ...]
                     Empty list for non-pose models.
        detections : [(label, conf, x1, y1, x2, y2), ...]
                     Every tracked box the model found this frame.
        frame_w/h  : display canvas dimensions
        Returns    : the (modified) frame
        """
        return frame

    def on_stop(self) -> None:
        """Called when the stream stops."""
