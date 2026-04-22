#!/usr/bin/env python3
"""
game_bounce.py — Hand-Slap Ball Bounce Game
--------------------------------------------
Load this from the streamer's "Game Script" browser.

How it works
------------
A glowing ball bounces around the frame.  Any hand keypoint that moves
close enough to the ball imparts a velocity impulse — direction and speed
based on the keypoint's movement between frames.  The ball slows down over
time (friction) so it remains manageable.

Requires a pose model (e.g. yolo11n-pose or your hand-keypoints model).
Works best with the 21-keypoint hand model, but falls back gracefully to
any number of keypoints.

Configuration constants at the top of the class — tweak to taste.
"""

from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from game_plugin import GamePlugin


class Game(GamePlugin):
    # ── Tunable constants ──────────────────────────────────────
    BALL_RADIUS    = 22          # pixels
    FRICTION       = 0.985       # velocity multiplier per frame (< 1 = slows down)
    MIN_SPEED      = 60          # px/s — ball never fully stops (keeps rolling gently)
    MAX_SPEED      = 1800        # px/s — cap on maximum speed after a slap
    HIT_RADIUS     = 55          # px — how close a keypoint must be to "slap" the ball
    IMPULSE_SCALE  = 3.5         # multiplier on keypoint velocity → ball velocity
    TRAIL_LENGTH   = 18          # ghost trail frames
    TRAIL_ALPHA    = 0.55        # opacity of oldest trail ghost

    # Visual
    BALL_COLOR     = (255, 220, 40)    # BGR — gold core
    GLOW_COLOR     = (120, 200, 255)   # BGR — glow ring
    TRAIL_COLOR    = (80,  160, 255)   # BGR — motion trail
    TEXT_COLOR     = (255, 255, 255)
    SCORE_COLOR    = (100, 255, 180)
    HIT_FLASH_DUR  = 0.12              # seconds the flash stays visible

    def __init__(self):
        self._w = 640
        self._h = 480
        self._bx  = 0.0;  self._by  = 0.0    # ball position (float)
        self._vx  = 0.0;  self._vy  = 0.0    # ball velocity (px/s)
        self._trail: list[tuple[float, float]] = []
        self._prev_kps: list[tuple[float, float]] = []   # previous frame keypoint positions
        self._score  = 0
        self._t_last = time.perf_counter()
        self._hit_flash_t = 0.0   # timestamp of last hit (for flash effect)
        self._started = False

    # ── GamePlugin API ─────────────────────────────────────────

    def on_start(self, frame_w: int, frame_h: int) -> None:
        self._w, self._h = frame_w, frame_h
        self._reset_ball()
        self._score  = 0
        self._trail  = []
        self._prev_kps = []
        self._t_last = time.perf_counter()
        self._started = True

    def on_frame(
        self,
        frame: np.ndarray,
        keypoints: list[list[tuple[float, float, float]]],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        if not self._started:
            self.on_start(frame_w, frame_h)

        now = time.perf_counter()
        dt  = min(now - self._t_last, 0.1)   # cap dt to avoid tunnelling after lag
        self._t_last = now

        # ── Collect flat keypoint list across all hands ────────
        flat_kps: list[tuple[float, float]] = []
        for instance in keypoints:
            for (x, y, conf) in instance:
                if conf > 0.25:
                    flat_kps.append((x, y))

        # ── Compute keypoint velocities & check for hits ───────
        hit_occurred = False
        if self._prev_kps and flat_kps:
            n = min(len(flat_kps), len(self._prev_kps))
            for i in range(n):
                px, py = self._prev_kps[i]
                cx, cy = flat_kps[i]
                dist = math.hypot(cx - self._bx, cy - self._by)
                if dist < self.HIT_RADIUS + self.BALL_RADIUS:
                    # Velocity of this keypoint (px/s)
                    kv_x = (cx - px) / max(dt, 1e-4)
                    kv_y = (cy - py) / max(dt, 1e-4)
                    # Apply impulse — weighted by how close the kp is to ball center
                    weight = 1.0 - (dist / (self.HIT_RADIUS + self.BALL_RADIUS))
                    self._vx += kv_x * self.IMPULSE_SCALE * weight
                    self._vy += kv_y * self.IMPULSE_SCALE * weight
                    # Clamp speed
                    spd = math.hypot(self._vx, self._vy)
                    if spd > self.MAX_SPEED:
                        scale = self.MAX_SPEED / spd
                        self._vx *= scale;  self._vy *= scale
                    hit_occurred = True
                    self._score += 1
                    self._hit_flash_t = now

        self._prev_kps = flat_kps

        # ── Physics update ─────────────────────────────────────
        self._vx *= self.FRICTION
        self._vy *= self.FRICTION

        # Ensure minimum rolling speed
        spd = math.hypot(self._vx, self._vy)
        if spd < self.MIN_SPEED and spd > 0:
            scale = self.MIN_SPEED / spd
            self._vx *= scale;  self._vy *= scale
        elif spd == 0:
            # Give it a random nudge so it's never truly dead
            angle = random.uniform(0, 2 * math.pi)
            self._vx = math.cos(angle) * self.MIN_SPEED
            self._vy = math.sin(angle) * self.MIN_SPEED

        # Move
        self._bx += self._vx * dt
        self._by += self._vy * dt

        # Wall bouncing
        r = self.BALL_RADIUS
        if self._bx - r < 0:
            self._bx = float(r);  self._vx = abs(self._vx)
        if self._bx + r > frame_w:
            self._bx = float(frame_w - r);  self._vx = -abs(self._vx)
        if self._by - r < 0:
            self._by = float(r);  self._vy = abs(self._vy)
        if self._by + r > frame_h:
            self._by = float(frame_h - r);  self._vy = -abs(self._vy)

        # ── Trail ─────────────────────────────────────────────
        self._trail.append((self._bx, self._by))
        if len(self._trail) > self.TRAIL_LENGTH:
            self._trail.pop(0)

        # ── Draw ──────────────────────────────────────────────
        self._draw(frame, frame_w, frame_h, hit_occurred, now)

        return frame

    def on_stop(self) -> None:
        self._started = False

    # ── Internals ─────────────────────────────────────────────

    def _reset_ball(self):
        cx, cy = self._w / 2, self._h / 2
        self._bx = cx + random.uniform(-cx * 0.3, cx * 0.3)
        self._by = cy + random.uniform(-cy * 0.3, cy * 0.3)
        angle    = random.uniform(0, 2 * math.pi)
        speed    = random.uniform(250, 450)
        self._vx = math.cos(angle) * speed
        self._vy = math.sin(angle) * speed

    def _draw(self, frame: np.ndarray, fw: int, fh: int,
              hit: bool, now: float) -> None:
        bx, by = int(self._bx), int(self._by)
        r = self.BALL_RADIUS

        # -- Motion trail (fading circles) --
        n_trail = len(self._trail)
        for i, (tx, ty) in enumerate(self._trail[:-1]):
            alpha  = self.TRAIL_ALPHA * (i / max(n_trail - 1, 1))
            t_r    = max(2, int(r * (i / max(n_trail - 1, 1)) * 0.7))
            overlay = frame.copy()
            cv2.circle(overlay, (int(tx), int(ty)), t_r, self.TRAIL_COLOR, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)

        # -- Glow ring --
        flash = (now - self._hit_flash_t) < self.HIT_FLASH_DUR
        glow_color = (255, 255, 255) if flash else self.GLOW_COLOR
        glow_r = r + 10 if flash else r + 6
        overlay = frame.copy()
        cv2.circle(overlay, (bx, by), glow_r, glow_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # -- Ball core --
        core_color = (200, 240, 255) if flash else self.BALL_COLOR
        cv2.circle(frame, (bx, by), r, core_color, -1, cv2.LINE_AA)

        # -- Specular highlight --
        hl_x, hl_y = bx - r // 3, by - r // 3
        cv2.circle(frame, (hl_x, hl_y), max(3, r // 4), (255, 255, 255), -1, cv2.LINE_AA)

        # -- Speed indicator bar (top-left) --
        spd   = math.hypot(self._vx, self._vy)
        bar_w = int(min(spd / self.MAX_SPEED, 1.0) * 150)
        cv2.rectangle(frame, (8, 8), (158, 20), (40, 40, 40), -1)
        bar_color = (60, 220, 60) if spd < 600 else (60, 120, 255) if spd < 1200 else (40, 40, 255)
        cv2.rectangle(frame, (8, 8), (8 + bar_w, 20), bar_color, -1)
        cv2.putText(frame, "SPD", (162, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)

        # -- Score --
        score_txt = f"Hits: {self._score}"
        (sw, sh), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(frame, score_txt, (fw - sw - 10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, self.SCORE_COLOR, 2, cv2.LINE_AA)

        # -- Hit proximity rings (show all active keypoints) --
        if self._prev_kps:
            for (kx, ky) in self._prev_kps:
                dist = math.hypot(kx - self._bx, ky - self._by)
                if dist < self.HIT_RADIUS * 2:
                    alpha_ring = max(0.0, 1.0 - dist / (self.HIT_RADIUS * 2))
                    ring_overlay = frame.copy()
                    cv2.circle(ring_overlay, (int(kx), int(ky)),
                               self.HIT_RADIUS, (80, 255, 180), 1, cv2.LINE_AA)
                    cv2.addWeighted(ring_overlay, alpha_ring * 0.5,
                                    frame, 1 - alpha_ring * 0.5, 0, frame)
