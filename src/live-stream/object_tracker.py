#!/usr/bin/env python3
"""
object_tracker.py — Shared Object Tracking & Label Smoothing Module
---------------------------------------------------------------------
Provides a single ObjectTracker class consumed by both yolo_streamer.py
and gpu_yolo_streamer.py.

Tracking approach:
  - Centroid-based IoU matching (no external deps beyond NumPy)
  - Kalman-style exponential smoothing on bounding-box coordinates
  - Occlusion memory: keeps a track alive for `max_lost_frames` frames
    after it disappears (handles brief pass-behind-person situations)
  - Optional label smoothing: majority-votes the class label over the
    last `label_smooth_frames` frames so flickering labels stabilise

All parameters are plain Python values — the UI sliders in each streamer
just write new values to the tracker's attributes at any time.

Usage
-----
    from object_tracker import ObjectTracker

    tracker = ObjectTracker(
        iou_threshold       = 0.30,   # min IoU to match detection→track
        max_lost_frames     = 15,     # frames a hidden track is kept alive
        smooth_alpha        = 0.40,   # box-smoothing weight (0=frozen,1=raw)
        label_smooth_frames = 5,      # majority-vote window for class label
    )

    # Each inference frame:
    raw_boxes = [("person", 0.91, x1, y1, x2, y2), ...]
    smoothed  = tracker.update(raw_boxes)  # returns same format

    # On stream stop / restart:
    tracker.reset()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Internal track representation
# ─────────────────────────────────────────────────────────────

@dataclass
class _Track:
    track_id:   int
    label:      str
    conf:       float
    x1: float; y1: float; x2: float; y2: float  # smoothed box

    lost_frames: int = 0          # frames since last matched detection
    label_history: Deque[str] = field(default_factory=deque)


# ─────────────────────────────────────────────────────────────
#  Public tracker class
# ─────────────────────────────────────────────────────────────

class ObjectTracker:
    """
    Centroid/IoU multi-object tracker with:
      - exponential box smoothing
      - occlusion memory (max_lost_frames)
      - majority-vote label smoothing
    """

    def __init__(
        self,
        iou_threshold:        float = 0.30,
        max_lost_frames:      int   = 15,
        smooth_alpha:         float = 0.40,
        label_smooth_frames:  int   = 5,
    ):
        # ── tunable parameters (can be changed live from the UI) ──
        self.iou_threshold:        float = iou_threshold
        self.max_lost_frames:      int   = max_lost_frames
        self.smooth_alpha:         float = smooth_alpha   # 0 = frozen, 1 = raw
        self.label_smooth_frames:  int   = label_smooth_frames

        # ── internal state ──
        self._tracks:   dict[int, _Track] = {}
        self._next_id:  int               = 0

    # ── public API ───────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracks (call when stream restarts)."""
        self._tracks.clear()
        self._next_id = 0

    def update(
        self,
        detections: list[tuple[str, float, int, int, int, int]],
    ) -> list[tuple[str, float, int, int, int, int]]:
        """
        Match detections to existing tracks, create new ones, age lost ones.

        Parameters
        ----------
        detections : list of (label, conf, x1, y1, x2, y2)

        Returns
        -------
        list of (label, conf, x1, y1, x2, y2)
            Smoothed / extrapolated boxes for every currently visible track.
        """
        alpha = max(0.0, min(1.0, self.smooth_alpha))

        # ── 1. IoU match detections → tracks ──────────────────
        unmatched_det_indices  = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        if self._tracks and detections:
            track_ids   = list(self._tracks.keys())
            iou_matrix  = np.zeros((len(detections), len(track_ids)), dtype=np.float32)

            for di, det in enumerate(detections):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = _iou(det[2:6], self._tracks[tid])

            # Greedy match (best IoU first)
            while True:
                idx = np.argmax(iou_matrix)
                di, ti = divmod(int(idx), len(track_ids))
                best_iou = iou_matrix[di, ti]
                if best_iou < self.iou_threshold:
                    break
                tid = track_ids[ti]
                det = detections[di]
                self._update_track(self._tracks[tid], det, alpha)
                matched_track_ids.add(tid)
                unmatched_det_indices.remove(di)
                iou_matrix[di, :] = -1.0
                iou_matrix[:, ti] = -1.0

        # ── 2. Create new tracks for unmatched detections ─────
        for di in unmatched_det_indices:
            label, conf, x1, y1, x2, y2 = detections[di]
            t = _Track(
                track_id=self._next_id,
                label=label,
                conf=conf,
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
            )
            t.label_history = deque([label], maxlen=max(1, self.label_smooth_frames))
            self._tracks[self._next_id] = t
            self._next_id += 1

        # ── 3. Age unmatched tracks; remove stale ones ────────
        stale = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids and tid not in {
                self._tracks[list(self._tracks.keys())[i]].track_id
                for i in range(len(self._tracks))
                if list(self._tracks.keys())[i] not in matched_track_ids
                and self._tracks[list(self._tracks.keys())[i]].lost_frames == 0
            }:
                pass  # already counted above; handled below

        # Cleaner aging loop
        stale = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        # ── 4. Build output list ──────────────────────────────
        output = []
        for track in self._tracks.values():
            label = self._smoothed_label(track)
            output.append((
                label,
                track.conf,
                int(round(track.x1)), int(round(track.y1)),
                int(round(track.x2)), int(round(track.y2)),
            ))
        return output

    # ── internals ────────────────────────────────────────────

    def _update_track(self, track: _Track, det: tuple, alpha: float) -> None:
        label, conf, x1, y1, x2, y2 = det
        track.lost_frames = 0
        track.conf = conf

        # Exponential smoothing on box coordinates
        track.x1 = alpha * x1 + (1 - alpha) * track.x1
        track.y1 = alpha * y1 + (1 - alpha) * track.y1
        track.x2 = alpha * x2 + (1 - alpha) * track.x2
        track.y2 = alpha * y2 + (1 - alpha) * track.y2

        # Label history (for majority-vote smoothing)
        win = max(1, self.label_smooth_frames)
        if track.label_history.maxlen != win:
            track.label_history = deque(track.label_history, maxlen=win)
        track.label_history.append(label)

    @staticmethod
    def _smoothed_label(track: _Track) -> str:
        """Return the majority-vote label from the track's history."""
        if not track.label_history:
            return track.label
        counts: dict[str, int] = {}
        for lbl in track.label_history:
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────

def _iou(
    det_box: tuple[int, int, int, int],
    track: _Track,
) -> float:
    """Intersection-over-Union between a raw detection box and a track."""
    dx1, dy1, dx2, dy2 = det_box
    tx1, ty1, tx2, ty2 = track.x1, track.y1, track.x2, track.y2

    ix1 = max(dx1, tx1); iy1 = max(dy1, ty1)
    ix2 = min(dx2, tx2); iy2 = min(dy2, ty2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_d = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
    area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union   = area_d + area_t - inter
    return inter / union if union > 0 else 0.0
