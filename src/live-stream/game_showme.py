#!/usr/bin/env python3
"""
game_showme.py — "Show Me!" Object Recognition Quiz Game
---------------------------------------------------------
Load this from the streamer's Game Plugin browser.

How it works
------------
The game picks a random object from the model's class list and displays
it at the top of the screen. You have TIME_LIMIT seconds to hold that
object up in front of the camera. The faster the model recognises it
with sufficient confidence, the more points you earn.

A countdown timer counts down visually. When the model sees the target
object, points are awarded based on time remaining, and a new target is
chosen after a short celebration pause. If time runs out, a miss is
recorded and a new target is chosen.

Works with any YOLO detection or pose model — uses the detections
argument, so no pose output needed.

Configuration constants are at the top of the Game class.
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
    TIME_LIMIT        = 15.0   # seconds per round
    MIN_CONF          = 0.55   # model confidence needed to count as "seen"
    CELEBRATE_DUR     = 2.2    # seconds to show the "nice!" screen before next round
    MAX_POINTS        = 1000   # points for an instant recognition
    FONT              = cv2.FONT_HERSHEY_DUPLEX
    FONT_MONO         = cv2.FONT_HERSHEY_PLAIN

    # Colours (BGR)
    COL_BG_TOP        = (30,  20,  60)    # banner background
    COL_TARGET        = (255, 220,  60)   # target word colour
    COL_TIMER_OK      = (80,  220,  80)   # timer bar — plenty of time
    COL_TIMER_WARN    = (40,  160, 255)   # timer bar — getting low
    COL_TIMER_DANGER  = (40,   40, 255)   # timer bar — almost out
    COL_HIT           = (60,  255, 140)   # "found it" flash colour
    COL_MISS          = (40,   40, 220)   # "time's up" colour
    COL_SCORE         = (200, 255, 200)
    COL_WHITE         = (255, 255, 255)
    COL_SHADOW        = (0,     0,   0)

    def __init__(self):
        self._class_names: list[str] = []
        self._target: str = ""
        self._round_start: float = 0.0
        self._score: int = 0
        self._streak: int = 0
        self._misses: int = 0
        self._rounds: int = 0
        self._state: str = "waiting"   # "waiting" | "playing" | "celebrate" | "miss"
        self._state_ts: float = 0.0
        self._last_pts: int = 0
        self._found_conf: float = 0.0
        self._fw: int = 640
        self._fh: int = 480

    # ── GamePlugin API ─────────────────────────────────────────

    def on_start(self, frame_w: int, frame_h: int, class_names=None) -> None:
        self._fw, self._fh = frame_w, frame_h
        self._class_names = list(class_names) if class_names else []
        self._score   = 0
        self._streak  = 0
        self._misses  = 0
        self._rounds  = 0
        self._last_pts = 0
        if self._class_names:
            self._new_round()
        else:
            self._state = "waiting"

    def on_frame(
        self,
        frame: np.ndarray,
        keypoints: list,
        detections: list,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        self._fw, self._fh = frame_w, frame_h
        now = time.perf_counter()

        # If class names weren't available at on_start, wait for detections
        # to infer them (shouldn't happen normally, but good fallback)
        if not self._class_names and detections:
            seen = list({label for label, *_ in detections})
            if seen:
                self._class_names = seen

        if self._state == "waiting":
            self._draw_waiting(frame)
            return frame

        elapsed  = now - self._round_start
        remaining = max(0.0, self.TIME_LIMIT - elapsed)

        if self._state == "playing":
            # Check if target is in detections above confidence threshold
            found_conf = 0.0
            for (label, conf, *_) in detections:
                if label == self._target and conf >= self.MIN_CONF:
                    found_conf = max(found_conf, conf)

            if found_conf > 0:
                # Success!
                frac = remaining / self.TIME_LIMIT
                pts = max(50, int(self.MAX_POINTS * frac))
                # Streak bonus: +10% per consecutive hit, capped at 2x
                bonus = min(2.0, 1.0 + self._streak * 0.10)
                pts = int(pts * bonus)
                self._score    += pts
                self._last_pts  = pts
                self._found_conf = found_conf
                self._streak   += 1
                self._rounds   += 1
                self._state     = "celebrate"
                self._state_ts  = now

            elif remaining <= 0:
                # Time's up
                self._streak  = 0
                self._misses += 1
                self._rounds += 1
                self._state   = "miss"
                self._state_ts = now

        elif self._state == "celebrate":
            if now - self._state_ts >= self.CELEBRATE_DUR:
                self._new_round()

        elif self._state == "miss":
            if now - self._state_ts >= self.CELEBRATE_DUR:
                self._new_round()

        # ── Draw overlay ──
        self._draw_banner(frame, remaining)
        self._draw_scoreboard(frame)

        if self._state == "celebrate":
            self._draw_celebrate(frame)
        elif self._state == "miss":
            self._draw_miss(frame)
        else:
            self._draw_timer_bar(frame, remaining)
            self._highlight_target(frame, detections)

        return frame

    def on_stop(self) -> None:
        self._state = "waiting"

    # ── Internals ─────────────────────────────────────────────

    def _new_round(self):
        if not self._class_names:
            self._state = "waiting"
            return
        # Pick a new target, avoid repeating
        choices = [c for c in self._class_names if c != self._target]
        if not choices:
            choices = self._class_names
        self._target      = random.choice(choices)
        self._round_start = time.perf_counter()
        self._state       = "playing"

    def _draw_banner(self, frame: np.ndarray, remaining: float):
        fw, fh = self._fw, self._fh
        banner_h = max(70, fh // 7)

        # Semi-transparent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, banner_h), self.COL_BG_TOP, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if self._state == "playing":
            # "SHOW ME:" label
            small = "SHOW ME:"
            (sw, sh), _ = cv2.getTextSize(small, self.FONT, 0.7, 1)
            cv2.putText(frame, small, (fw // 2 - sw // 2, banner_h // 2 - sh),
                        self.FONT, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

            # Target word — big, centred, bold shadow
            target_txt = self._target.upper()
            scale = self._fit_text_scale(target_txt, fw - 40, banner_h // 2 - 4, self.FONT, 2)
            (tw, th), _ = cv2.getTextSize(target_txt, self.FONT, scale, 2)
            tx = fw // 2 - tw // 2
            ty = banner_h - 10
            cv2.putText(frame, target_txt, (tx + 2, ty + 2), self.FONT, scale,
                        self.COL_SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, target_txt, (tx, ty), self.FONT, scale,
                        self.COL_TARGET, 2, cv2.LINE_AA)

        elif self._state == "celebrate":
            txt = f"FOUND IT!  +{self._last_pts} pts"
            scale = self._fit_text_scale(txt, fw - 40, banner_h - 10, self.FONT, 2)
            (tw, th), _ = cv2.getTextSize(txt, self.FONT, scale, 2)
            tx, ty = fw // 2 - tw // 2, banner_h - 12
            cv2.putText(frame, txt, (tx + 2, ty + 2), self.FONT, scale, self.COL_SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty), self.FONT, scale, self.COL_HIT, 2, cv2.LINE_AA)

        elif self._state == "miss":
            txt = f"TIME'S UP!  ({self._target.upper()})"
            scale = self._fit_text_scale(txt, fw - 40, banner_h - 10, self.FONT, 2)
            (tw, th), _ = cv2.getTextSize(txt, self.FONT, scale, 2)
            tx, ty = fw // 2 - tw // 2, banner_h - 12
            cv2.putText(frame, txt, (tx + 2, ty + 2), self.FONT, scale, self.COL_SHADOW, 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty), self.FONT, scale, self.COL_MISS, 2, cv2.LINE_AA)

        elif self._state == "waiting":
            txt = "Waiting for model class list..."
            (tw, _), _ = cv2.getTextSize(txt, self.FONT, 0.7, 1)
            cv2.putText(frame, txt, (fw // 2 - tw // 2, banner_h // 2 + 10),
                        self.FONT, 0.7, self.COL_WHITE, 1, cv2.LINE_AA)

    def _draw_timer_bar(self, frame: np.ndarray, remaining: float):
        fw, fh = self._fw, self._fh
        banner_h = max(70, fh // 7)
        bar_y    = banner_h + 4
        bar_h    = 10
        frac     = remaining / self.TIME_LIMIT

        # Background track
        cv2.rectangle(frame, (0, bar_y), (fw, bar_y + bar_h), (40, 40, 40), -1)

        # Filled portion
        bar_w = int(fw * frac)
        if frac > 0.5:
            col = self.COL_TIMER_OK
        elif frac > 0.25:
            col = self.COL_TIMER_WARN
        else:
            col = self.COL_TIMER_DANGER

        if bar_w > 0:
            cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + bar_h), col, -1)

        # Countdown number on the right
        secs = f"{remaining:.1f}s"
        (tw, th), _ = cv2.getTextSize(secs, self.FONT, 0.55, 1)
        cv2.putText(frame, secs, (fw - tw - 6, bar_y + bar_h + th + 2),
                    self.FONT, 0.55, col, 1, cv2.LINE_AA)

    def _draw_scoreboard(self, frame: np.ndarray):
        fw, fh = self._fw, self._fh
        lines = [
            f"Score:  {self._score}",
            f"Streak: {self._streak}x",
            f"Misses: {self._misses}",
        ]
        x, y = 8, fh - 10 - (len(lines) - 1) * 22
        for line in lines:
            cv2.putText(frame, line, (x + 1, y + 1), self.FONT, 0.6, self.COL_SHADOW, 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), self.FONT, 0.6, self.COL_SCORE, 1, cv2.LINE_AA)
            y += 22

    def _draw_celebrate(self, frame: np.ndarray):
        fw, fh = self._fw, self._fh
        # Pulsing glow overlay
        t = time.perf_counter()
        alpha = 0.18 + 0.10 * math.sin(t * 10)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (60, 200, 80), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Streak badge if > 1
        if self._streak > 1:
            badge = f"{self._streak}x STREAK!"
            (bw, bh), _ = cv2.getTextSize(badge, self.FONT, 1.1, 2)
            bx = fw // 2 - bw // 2
            by = fh // 2 + 30
            cv2.putText(frame, badge, (bx + 2, by + 2), self.FONT, 1.1, self.COL_SHADOW, 4, cv2.LINE_AA)
            cv2.putText(frame, badge, (bx, by), self.FONT, 1.1, (255, 240, 80), 2, cv2.LINE_AA)

        # Confidence readout
        conf_txt = f"Confidence: {self._found_conf:.0%}"
        (cw, _), _ = cv2.getTextSize(conf_txt, self.FONT, 0.65, 1)
        cv2.putText(frame, conf_txt, (fw // 2 - cw // 2, fh - 30),
                    self.FONT, 0.65, self.COL_WHITE, 1, cv2.LINE_AA)

    def _draw_miss(self, frame: np.ndarray):
        fw, fh = self._fw, self._fh
        t = time.perf_counter()
        alpha = 0.15 + 0.08 * math.sin(t * 8)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (30, 30, 180), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        hint = "Try again!"
        (hw, _), _ = cv2.getTextSize(hint, self.FONT, 0.9, 2)
        cv2.putText(frame, hint, (fw // 2 - hw // 2 + 2, fh // 2 + 32),
                    self.FONT, 0.9, self.COL_SHADOW, 3, cv2.LINE_AA)
        cv2.putText(frame, hint, (fw // 2 - hw // 2, fh // 2 + 30),
                    self.FONT, 0.9, (160, 160, 255), 2, cv2.LINE_AA)

    def _highlight_target(self, frame: np.ndarray, detections: list):
        """Draw a bright highlight ring around any box whose label matches the target."""
        for (label, conf, x1, y1, x2, y2) in detections:
            if label != self._target:
                continue
            # Confidence-based alpha so it pulses as conf approaches threshold
            alpha = min(1.0, conf / self.MIN_CONF)
            t = time.perf_counter()
            pulse = 0.6 + 0.4 * math.sin(t * 6) * alpha
            thick = max(2, int(5 * pulse))
            col = (
                int(60 * (1 - pulse)),
                int(200 * pulse),
                int(255 * pulse),
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
            pct_txt = f"{conf:.0%}"
            (pw, ph), _ = cv2.getTextSize(pct_txt, self.FONT, 0.7, 1)
            cv2.putText(frame, pct_txt, (x1 + 4, y2 - 6),
                        self.FONT, 0.7, col, 1, cv2.LINE_AA)

    def _draw_waiting(self, frame: np.ndarray):
        pass   # banner handles this

    @staticmethod
    def _fit_text_scale(text: str, max_w: int, max_h: int, font, thickness: int) -> float:
        """Binary-search the largest font scale that fits within max_w x max_h."""
        lo, hi = 0.3, 6.0
        for _ in range(12):
            mid = (lo + hi) / 2
            (w, h), _ = cv2.getTextSize(text, font, mid, thickness)
            if w <= max_w and h <= max_h:
                lo = mid
            else:
                hi = mid
        return lo
