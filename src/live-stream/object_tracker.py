#!/usr/bin/env python3
"""
object_tracker.py — Kalman-filter Object Tracker with Label Smoothing
----------------------------------------------------------------------
Replaces the old centroid/IoU tracker with one that predicts where each
object will be next frame, so fast-moving objects stay locked to a single
track instead of spawning a new one every time they leave their last box.

Tracking approach
-----------------
  - Per-track Kalman filter (constant-velocity model, 4-state: cx,cy,w,h)
  - Predicted position used for IoU matching, not last-seen position
  - Hungarian-style greedy matching (best-IoU-first)
  - Occlusion memory: track kept alive for `max_lost_frames` frames,
    coasting on Kalman prediction so the box keeps moving smoothly
  - Majority-vote label smoothing over last N frames

All parameters are plain Python values — the UI sliders write to them live.

Usage
-----
    from object_tracker import ObjectTracker

    tracker = ObjectTracker(
        iou_threshold       = 0.20,   # lower is fine — we match on predictions
        max_lost_frames     = 15,
        smooth_alpha        = 0.60,   # kept for API compatibility; Kalman handles smoothing
        label_smooth_frames = 5,
    )

    raw_boxes = [("hand", 0.91, x1, y1, x2, y2), ...]
    smoothed  = tracker.update(raw_boxes)

    tracker.reset()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Minimal Kalman filter — constant velocity, (cx, cy, w, h)
# ─────────────────────────────────────────────────────────────

class _KalmanBox:
    """
    4-state Kalman filter tracking a bounding box as (cx, cy, w, h)
    with a constant-velocity motion model.

    State vector  x = [cx, cy, w, h, vx, vy, vw, vh]  (8-dim)
    Observation   z = [cx, cy, w, h]                   (4-dim)
    """

    def __init__(self, cx: float, cy: float, w: float, h: float):
        dt = 1.0   # one frame timestep

        # ── Transition matrix F ─────────────────────────────
        #   pos(t+1) = pos(t) + vel(t)
        #   vel(t+1) = vel(t)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        # ── Observation matrix H ────────────────────────────
        self.H = np.eye(4, 8, dtype=np.float32)

        # ── Process noise Q ─────────────────────────────────
        # Allow position to drift a little; velocity more uncertain
        q_pos, q_vel = 1.0, 0.25
        self.Q = np.diag([q_pos, q_pos, q_pos, q_pos,
                          q_vel, q_vel, q_vel, q_vel]).astype(np.float32)

        # ── Measurement noise R ─────────────────────────────
        # How much we trust each new detection box
        r_pos, r_size = 1.0, 4.0
        self.R = np.diag([r_pos, r_pos, r_size, r_size]).astype(np.float32)

        # ── Initial state ────────────────────────────────────
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        # ── Initial covariance P ─────────────────────────────
        # Start uncertain about velocity
        self.P = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32)

    # ── Public API ────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Advance state one frame. Returns predicted (cx, cy, w, h)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Clamp width/height to stay positive
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.x[:4].copy()

    def update(self, cx: float, cy: float, w: float, h: float) -> None:
        """Correct state with a new measurement."""
        z = np.array([cx, cy, w, h], dtype=np.float32)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])

    @property
    def box_xyxy(self) -> tuple[float, float, float, float]:
        """Current estimate as (x1, y1, x2, y2)."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# ─────────────────────────────────────────────────────────────
#  Internal track record
# ─────────────────────────────────────────────────────────────

@dataclass
class _Track:
    track_id:   int
    label:      str
    conf:       float
    kalman:     _KalmanBox

    lost_frames:   int = 0
    label_history: Deque[str] = field(default_factory=deque)


# ─────────────────────────────────────────────────────────────
#  Public tracker class
# ─────────────────────────────────────────────────────────────

class ObjectTracker:
    """
    Multi-object tracker with Kalman-filter motion prediction.

    Parameters
    ----------
    iou_threshold       : minimum IoU (against *predicted* box) to match
    max_lost_frames     : frames a hidden track coasts before deletion
    smooth_alpha        : kept for API compatibility; Kalman does smoothing
    label_smooth_frames : majority-vote window for class label
    """

    def __init__(
        self,
        iou_threshold:        float = 0.20,
        max_lost_frames:      int   = 15,
        smooth_alpha:         float = 0.60,   # unused internally, kept for UI compat
        label_smooth_frames:  int   = 5,
    ):
        self.iou_threshold        = iou_threshold
        self.max_lost_frames      = max_lost_frames
        self.smooth_alpha         = smooth_alpha
        self.label_smooth_frames  = label_smooth_frames

        self._tracks:  dict[int, _Track] = {}
        self._next_id: int = 0

    # ── Public API ────────────────────────────────────────────

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update(
        self,
        detections: list[tuple[str, float, int, int, int, int]],
    ) -> list[tuple[str, float, int, int, int, int]]:
        """
        Match detections → tracks, predict missing ones, age stale ones.

        Parameters
        ----------
        detections : list of (label, conf, x1, y1, x2, y2)

        Returns
        -------
        list of (label, conf, x1, y1, x2, y2)
            Smoothed / predicted boxes for every currently live track.
        """

        # ── 1. Predict every track one step forward ───────────
        for track in self._tracks.values():
            track.kalman.predict()

        # ── 2. IoU-match detections → predicted track boxes ───
        unmatched_det   = list(range(len(detections)))
        matched_ids: set[int] = set()

        if self._tracks and detections:
            track_ids  = list(self._tracks.keys())
            iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)

            for di, det in enumerate(detections):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = _iou_xyxy(
                        _det_to_xyxy(det),
                        self._tracks[tid].kalman.box_xyxy,
                    )

            # Greedy: match best IoU pairs first
            while True:
                idx = int(np.argmax(iou_matrix))
                di, ti = divmod(idx, len(track_ids))
                best   = iou_matrix[di, ti]
                if best < self.iou_threshold:
                    break
                tid   = track_ids[ti]
                label, conf, x1, y1, x2, y2 = detections[di]
                cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
                w  =  x2 - x1;       h  =  y2 - y1
                track = self._tracks[tid]
                track.kalman.update(cx, cy, w, h)
                track.conf        = conf
                track.lost_frames = 0
                _append_label(track, label, self.label_smooth_frames)
                matched_ids.add(tid)
                if di in unmatched_det:
                    unmatched_det.remove(di)
                iou_matrix[di, :] = -1.0
                iou_matrix[:, ti] = -1.0

        # ── 3. Create new tracks for unmatched detections ─────
        for di in unmatched_det:
            label, conf, x1, y1, x2, y2 = detections[di]
            cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
            w  =  x2 - x1;       h  =  y2 - y1
            kf = _KalmanBox(cx, cy, w, h)
            t  = _Track(
                track_id=self._next_id,
                label=label,
                conf=conf,
                kalman=kf,
            )
            t.label_history = deque([label], maxlen=max(1, self.label_smooth_frames))
            self._tracks[self._next_id] = t
            self._next_id += 1

        # ── 4. Age unmatched tracks; delete stale ones ────────
        stale = []
        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        # ── 5. Build output ───────────────────────────────────
        out = []
        for track in self._tracks.values():
            x1, y1, x2, y2 = track.kalman.box_xyxy
            label = _majority_label(track)
            out.append((
                label,
                track.conf,
                int(round(x1)), int(round(y1)),
                int(round(x2)), int(round(y2)),
            ))
        return out


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _det_to_xyxy(det: tuple) -> tuple[float, float, float, float]:
    _, _, x1, y1, x2, y2 = det
    return float(x1), float(y1), float(x2), float(y2)


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _append_label(track: _Track, label: str, window: int) -> None:
    win = max(1, window)
    if track.label_history.maxlen != win:
        track.label_history = deque(track.label_history, maxlen=win)
    track.label_history.append(label)


def _majority_label(track: _Track) -> str:
    if not track.label_history:
        return track.label
    counts: dict[str, int] = {}
    for lbl in track.label_history:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)   # type: ignore[arg-type]