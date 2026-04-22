#!/usr/bin/env python3
"""
pose_renderer.py — Shared Pose Keypoint Rendering Module
---------------------------------------------------------
Draws YOLO-pose keypoints and skeleton connections on frames.
Consumed by both yolo_streamer.py and gpu_yolo_streamer.py.

Hand keypoint layout (21 points — MediaPipe / Ultralytics convention):
  0  = Wrist
  1  = Thumb CMC       2  = Thumb MCP      3  = Thumb IP      4  = Thumb Tip
  5  = Index MCP       6  = Index PIP      7  = Index DIP     8  = Index Tip
  9  = Middle MCP     10  = Middle PIP    11  = Middle DIP    12  = Middle Tip
 13  = Ring MCP       14  = Ring PIP      15  = Ring DIP      16  = Ring Tip
 17  = Pinky MCP      18  = Pinky PIP     19  = Pinky DIP     20  = Pinky Tip

For generic body-pose (17-point COCO), a fallback skeleton is included.

Usage
-----
    from pose_renderer import PoseRenderer, PoseStyles

    styles  = PoseStyles()           # holds colours / radii
    renderer = PoseRenderer(styles)

    # In your inference loop, after getting YOLO results:
    renderer.draw(frame, results)    # modifies frame in-place

    # To update colours from a UI:
    styles.keypoint_color  = (0, 255, 128)   # BGR
    styles.skeleton_color  = (255, 200, 0)
    styles.keypoint_radius = 5
    styles.skeleton_thick  = 2
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  Skeleton definitions
# ─────────────────────────────────────────────────────────────

# 21-point hand skeleton (finger chains + palm)
HAND_SKELETON: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckle bar
    (5, 9), (9, 13), (13, 17),
]

# 17-point COCO body skeleton (fallback for non-hand models)
BODY_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def _choose_skeleton(n_kp: int) -> list[tuple[int, int]]:
    """Pick the right skeleton based on the keypoint count in the model."""
    if n_kp == 21:
        return HAND_SKELETON
    elif n_kp == 17:
        return BODY_SKELETON
    else:
        # Unknown model — connect consecutive points as chains
        return [(i, i + 1) for i in range(n_kp - 1)]


# ─────────────────────────────────────────────────────────────
#  Style container (mutate freely from the UI)
# ─────────────────────────────────────────────────────────────

class PoseStyles:
    """
    All pose rendering parameters in one place.
    The streamer UI reads / writes these attributes directly.
    """

    def __init__(self):
        # Keypoint dot
        self.keypoint_color:  tuple[int, int, int] = (0, 230, 255)   # BGR cyan-ish
        self.keypoint_radius: int                  = 5
        self.keypoint_thick:  int                  = -1               # -1 = filled

        # Skeleton line
        self.skeleton_color:  tuple[int, int, int] = (255, 180, 0)   # BGR amber
        self.skeleton_thick:  int                  = 2

        # Confidence gate — keypoints below this are skipped
        self.conf_threshold: float = 0.3

        # Whether to draw keypoints at all (can be toggled from UI)
        self.show_keypoints: bool = True
        self.show_skeleton:  bool = True


# ─────────────────────────────────────────────────────────────
#  Renderer
# ─────────────────────────────────────────────────────────────

class PoseRenderer:
    """
    Draws pose keypoints and skeleton lines on frames.

    Parameters
    ----------
    styles : PoseStyles
        Shared style object; mutate its fields live from the UI.
    """

    def __init__(self, styles: PoseStyles):
        self.styles   = styles
        self._skeleton: list[tuple[int, int]] | None = None

    def draw(
        self,
        frame: np.ndarray,
        results,
        scale_xy: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """
        Draw all pose detections onto `frame` in-place.

        Parameters
        ----------
        frame    : BGR numpy array (the full-res display frame)
        results  : ultralytics Results object list (output of model())
        scale_xy : (sx, sy) scale factors if inference was run at smaller size
        """
        sx, sy = scale_xy
        s = self.styles

        for result in results:
            if result.keypoints is None:
                continue

            kp_data = result.keypoints.data   # shape: (N, K, 3) — x, y, conf
            if kp_data.shape[0] == 0:         # no detections this frame
                continue
            n_instances, n_kp, _ = kp_data.shape

            # Choose / cache skeleton
            if self._skeleton is None or getattr(self, "_last_n_kp", -1) != n_kp:
                self._skeleton = _choose_skeleton(n_kp)
                self._last_n_kp = n_kp

            for inst in range(n_instances):
                pts = kp_data[inst]  # (K, 3)

                # -- Skeleton lines first (drawn under dots) --
                if s.show_skeleton:
                    for (a, b) in self._skeleton:
                        if a >= n_kp or b >= n_kp:
                            continue
                        xa, ya, ca = float(pts[a][0]), float(pts[a][1]), float(pts[a][2])
                        xb, yb, cb = float(pts[b][0]), float(pts[b][1]), float(pts[b][2])
                        if ca < s.conf_threshold or cb < s.conf_threshold:
                            continue
                        p1 = (int(xa * sx), int(ya * sy))
                        p2 = (int(xb * sx), int(yb * sy))
                        cv2.line(frame, p1, p2, s.skeleton_color, s.skeleton_thick,
                                 cv2.LINE_AA)

                # -- Keypoint dots --
                if s.show_keypoints:
                    for k in range(n_kp):
                        xk, yk, ck = float(pts[k][0]), float(pts[k][1]), float(pts[k][2])
                        if ck < s.conf_threshold:
                            continue
                        cx = int(xk * sx)
                        cy = int(yk * sy)
                        cv2.circle(frame, (cx, cy), s.keypoint_radius,
                                   s.keypoint_color, s.keypoint_thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  Convenience: extract keypoints as plain Python list
# ─────────────────────────────────────────────────────────────

def extract_keypoints(
    results,
    scale_xy: tuple[float, float] = (1.0, 1.0),
    conf_threshold: float = 0.3,
) -> list[list[tuple[float, float, float]]]:
    """
    Pull keypoints out of YOLO results into a simple Python structure.

    Returns
    -------
    List of instances, each a list of (x, y, conf) tuples in display-frame coords.
    Useful for game scripts that need to read hand positions.
    """
    sx, sy = scale_xy
    all_instances = []

    for result in results:
        if result.keypoints is None:
            continue
        kp_data = result.keypoints.data
        if kp_data.shape[0] == 0:
            continue
        n_instances, n_kp, _ = kp_data.shape

        for inst in range(n_instances):
            pts = kp_data[inst]
            instance_kps = []
            for k in range(n_kp):
                x, y, c = float(pts[k][0]), float(pts[k][1]), float(pts[k][2])
                instance_kps.append((x * sx, y * sy, c))
            all_instances.append(instance_kps)

    return all_instances