#!/usr/bin/env python3
"""
game_pong.py — Hand-Tracked Pong
---------------------------------
Load this from the streamer's Game Plugin browser.

Two players track their paddles with their hands.
- Left half of screen  → Player 1 (left paddle)
- Right half of screen → Player 2 (right paddle)

Both players must show a hand on their side to start.
The wrist keypoint (kp 0) drives the paddle, or the centre
of the detection box if no pose output is available.

Y-only mode  : paddle tracks hand's vertical position only
XY mode      : paddle also follows hand horizontally (toggle in-game)

Ball speeds up over time and resets on each goal.
"""

from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from game_plugin import GamePlugin


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

PADDLE_W        = 18       # pixels wide
PADDLE_H        = 90       # pixels tall
PADDLE_MARGIN   = 30       # distance from edge of screen
PADDLE_SMOOTH   = 0.18     # lerp factor toward hand (0=frozen, 1=snap)

BALL_R          = 12
BALL_START_SPD  = 320      # px / second
BALL_ACCEL      = 18       # px/s added per paddle hit
BALL_MAX_SPD    = 900

SCORE_WIN       = 7        # first to this wins

READY_HOLD      = 1.2      # seconds both hands must be visible to start

# colours (BGR)
C_BG            = (15,  12,  30)
C_COURT         = (40,  36,  70)
C_NET           = (60,  55,  90)
C_P1            = (255, 130,  60)   # left  / orange
C_P2            = ( 60, 200, 255)   # right / cyan
C_BALL          = (255, 255, 255)
C_BALL_GLOW     = (180, 180, 255)
C_TEXT          = (220, 220, 240)
C_SHADOW        = (  0,   0,   0)
C_WIN           = (100, 255, 160)
C_READY_OK      = ( 80, 220,  80)
C_READY_NO      = ( 60,  60, 200)

FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO       = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def shadow_text(frame, text, pos, font, scale, colour, thickness=1):
    x, y = pos
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, C_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y    ), font, scale, colour,   thickness,     cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  Paddle
# ─────────────────────────────────────────────────────────────

class Paddle:
    def __init__(self, side: str, fw: int, fh: int, xy_mode: bool = False):
        self.side    = side          # "left" | "right"
        self.fw      = fw
        self.fh      = fh
        self.xy_mode = xy_mode

        # X is fixed in Y-only mode
        if side == "left":
            self.x = float(PADDLE_MARGIN + PADDLE_W // 2)
        else:
            self.x = float(fw - PADDLE_MARGIN - PADDLE_W // 2)

        self.y = fh / 2.0

        # Target (where hand is)
        self.target_x = self.x
        self.target_y = self.y

        self.colour = C_P1 if side == "left" else C_P2

    def set_target(self, hx: float, hy: float):
        self.target_y = clamp(hy, PADDLE_H // 2, self.fh - PADDLE_H // 2)
        if self.xy_mode:
            # Restrict X to own half + a small buffer
            half = self.fw / 2
            buf  = 20
            if self.side == "left":
                self.target_x = clamp(hx, PADDLE_MARGIN + PADDLE_W // 2, half - buf)
            else:
                self.target_x = clamp(hx, half + buf, self.fw - PADDLE_MARGIN - PADDLE_W // 2)

    def update(self, dt: float):
        t = clamp(PADDLE_SMOOTH + dt * 8, 0.0, 1.0)   # faster lerp at lower FPS
        self.y = lerp(self.y, self.target_y, t)
        if self.xy_mode:
            self.x = lerp(self.x, self.target_x, t)
        else:
            # Drift back to edge when not in XY mode
            if self.side == "left":
                edge = float(PADDLE_MARGIN + PADDLE_W // 2)
            else:
                edge = float(self.fw - PADDLE_MARGIN - PADDLE_W // 2)
            self.x = lerp(self.x, edge, t)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2)"""
        hw, hh = PADDLE_W // 2, PADDLE_H // 2
        return (int(self.x - hw), int(self.y - hh),
                int(self.x + hw), int(self.y + hh))

    def draw(self, frame: np.ndarray):
        x1, y1, x2, y2 = self.rect
        # Glow
        ov = frame.copy()
        cv2.rectangle(ov, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), self.colour, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
        # Body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colour, -1, cv2.LINE_AA)
        # Shine
        shine_x = x1 + 3 if self.side == "left" else x2 - 6
        cv2.rectangle(frame, (shine_x, y1 + 6), (shine_x + 3, y2 - 6),
                      (255, 255, 255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  Ball
# ─────────────────────────────────────────────────────────────

class Ball:
    def __init__(self, fw: int, fh: int):
        self.fw = fw
        self.fh = fh
        self.trail: list[tuple[float, float]] = []
        self.reset()

    def reset(self, towards: int = 0):
        """towards: -1 = left, +1 = right, 0 = random"""
        self.x = self.fw / 2.0
        self.y = self.fh / 2.0
        self.speed = float(BALL_START_SPD)
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = towards if towards != 0 else random.choice([-1, 1])
        self.vx = direction * self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        self.trail.clear()

    def update(self, dt: float, p1: Paddle, p2: Paddle) -> str:
        """Move ball, bounce off paddles and walls. Returns 'goal_left',
        'goal_right', 'hit', or ''."""
        self.trail.append((self.x, self.y))
        if len(self.trail) > 12:
            self.trail.pop(0)

        self.x += self.vx * dt
        self.y += self.vy * dt

        result = ""

        # Top / bottom wall bounce
        if self.y - BALL_R < 0:
            self.y = float(BALL_R)
            self.vy = abs(self.vy)
        if self.y + BALL_R > self.fh:
            self.y = float(self.fh - BALL_R)
            self.vy = -abs(self.vy)

        # Paddle collision helper
        def paddle_hit(paddle: Paddle) -> bool:
            x1, y1, x2, y2 = paddle.rect
            # Expand by ball radius for collision
            if (x1 - BALL_R <= self.x <= x2 + BALL_R and
                    y1 - BALL_R <= self.y <= y2 + BALL_R):
                return True
            return False

        # Left paddle
        if self.vx < 0 and paddle_hit(p1):
            self.vx = abs(self.vx)
            # Add spin based on relative hit position
            rel = (self.y - p1.y) / (PADDLE_H / 2)
            self.vy = rel * self.speed * 0.8
            self.speed = min(BALL_MAX_SPD, self.speed + BALL_ACCEL)
            self._normalise()
            result = "hit"

        # Right paddle
        if self.vx > 0 and paddle_hit(p2):
            self.vx = -abs(self.vx)
            rel = (self.y - p2.y) / (PADDLE_H / 2)
            self.vy = rel * self.speed * 0.8
            self.speed = min(BALL_MAX_SPD, self.speed + BALL_ACCEL)
            self._normalise()
            result = "hit"

        # Goals
        if self.x - BALL_R < 0:
            result = "goal_right"   # right player scores
        elif self.x + BALL_R > self.fw:
            result = "goal_left"    # left player scores

        return result

    def _normalise(self):
        spd = math.hypot(self.vx, self.vy)
        if spd > 0:
            scale = self.speed / spd
            self.vx *= scale
            self.vy *= scale

    def draw(self, frame: np.ndarray):
        # Trail
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i / max(n, 1)) * 0.45
            r = max(3, int(BALL_R * (i / max(n, 1)) * 0.75))
            ov = frame.copy()
            cv2.circle(ov, (int(tx), int(ty)), r, C_BALL_GLOW, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

        # Glow
        ov = frame.copy()
        cv2.circle(ov, (int(self.x), int(self.y)), BALL_R + 6, C_BALL_GLOW, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)

        # Core
        cv2.circle(frame, (int(self.x), int(self.y)), BALL_R, C_BALL, -1, cv2.LINE_AA)
        # Specular
        cv2.circle(frame, (int(self.x) - 4, int(self.y) - 4), max(2, BALL_R // 3),
                   (255, 255, 255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  Game
# ─────────────────────────────────────────────────────────────

class Game(GamePlugin):

    def __init__(self):
        super().__init__()
        self._fw   = 640
        self._fh   = 480
        self._state = "waiting"   # waiting | ready | playing | goal | win
        self._xy_mode = False

        self._p1: Paddle | None = None
        self._p2: Paddle | None = None
        self._ball: Ball | None = None

        self._score  = [0, 0]          # [p1, p2]
        self._t_last = time.perf_counter()

        # Ready-up tracking
        self._p1_seen_since: float | None = None
        self._p2_seen_since: float | None = None

        # Goal / win pause
        self._state_ts   = 0.0
        self._goal_scorer = 0   # 1 or 2

        # Flash
        self._hit_flash_t = 0.0

    # ── GamePlugin API ────────────────────────────────────────

    def on_start(self, frame_w: int, frame_h: int, class_names=None) -> None:
        self._fw, self._fh = frame_w, frame_h
        self._reset_game()

    def on_frame(
        self,
        frame: np.ndarray,
        keypoints: list,
        detections: list,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        now = time.perf_counter()
        dt  = clamp(now - self._t_last, 0.0, 0.08)
        self._t_last = now
        self._fw, self._fh = frame_w, frame_h

        # ── Draw court ──
        self._draw_court(frame)

        # ── Find hand positions ──
        p1_pos, p2_pos = self._find_hands(keypoints, detections)

        # ── State machine ──
        if self._state == "waiting":
            self._handle_waiting(frame, p1_pos, p2_pos, now)

        elif self._state == "ready":
            self._handle_ready(frame, p1_pos, p2_pos, now)

        elif self._state == "playing":
            self._handle_playing(frame, p1_pos, p2_pos, dt, now)

        elif self._state == "goal":
            self._handle_goal(frame, now)

        elif self._state == "win":
            self._handle_win(frame, now)

        # ── Draw paddles & ball ──
        if self._p1:
            self._p1.draw(frame)
        if self._p2:
            self._p2.draw(frame)
        if self._ball and self._state in ("playing", "goal"):
            self._ball.draw(frame)

        # ── HUD ──
        self._draw_hud(frame)
        self._draw_xy_toggle_hint(frame)

        return frame

    def on_stop(self) -> None:
        self._state = "waiting"

    # ── State handlers ────────────────────────────────────────

    def _handle_waiting(self, frame, p1_pos, p2_pos, now):
        fw, fh = self._fw, self._fh

        # Track how long each player has been visible
        if p1_pos:
            if self._p1_seen_since is None:
                self._p1_seen_since = now
        else:
            self._p1_seen_since = None

        if p2_pos:
            if self._p2_seen_since is None:
                self._p2_seen_since = now
        else:
            self._p2_seen_since = None

        p1_held = (now - self._p1_seen_since) if self._p1_seen_since else 0.0
        p2_held = (now - self._p2_seen_since) if self._p2_seen_since else 0.0

        if p1_held >= READY_HOLD and p2_held >= READY_HOLD:
            self._state = "ready"
            self._state_ts = now
            return

        # Draw waiting overlay — semi-transparent dark tint only
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (20, 15, 40), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        title = "HAND  PONG"
        (tw, th), _ = cv2.getTextSize(title, FONT, 1.8, 3)
        shadow_text(frame, title, (fw // 2 - tw // 2, fh // 2 - 60), FONT, 1.8, C_TEXT, 3)

        sub = "Show your hand on your half to start"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.6, 1)
        shadow_text(frame, sub, (fw // 2 - sw // 2, fh // 2 - 10), FONT, 0.6, (160, 160, 200))

        # Player ready indicators
        def ready_bar(cx, label, held, colour):
            frac = clamp(held / READY_HOLD, 0.0, 1.0)
            bar_w = 120;  bar_h = 16
            bx = cx - bar_w // 2;  by = fh // 2 + 30
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 60), -1)
            if frac > 0:
                col = colour if frac < 1.0 else C_READY_OK
                cv2.rectangle(frame, (bx, by), (bx + int(bar_w * frac), by + bar_h), col, -1)
            col_lbl = C_READY_OK if frac >= 1.0 else C_READY_NO
            shadow_text(frame, label, (cx - 30, by - 8), FONT, 0.55, col_lbl)

        ready_bar(fw // 4,     "P1  LEFT",  p1_held, C_P1)
        ready_bar(3 * fw // 4, "P2  RIGHT", p2_held, C_P2)



    def _handle_ready(self, frame, p1_pos, p2_pos, now):
        # 1-second countdown then play
        elapsed = now - self._state_ts
        countdown = max(0, 3 - int(elapsed))
        if elapsed >= 3.0:
            self._state = "playing"
            self._ball.reset(towards=random.choice([-1, 1]))
            return

        # Update paddle positions during countdown
        if p1_pos and self._p1:
            self._p1.set_target(*p1_pos)
            self._p1.update(0.016)
        if p2_pos and self._p2:
            self._p2.set_target(*p2_pos)
            self._p2.update(0.016)

        # Countdown number
        fw, fh = self._fw, self._fh
        num = str(countdown + 1) if countdown > 0 else "GO!"
        scale = 3.5 if num != "GO!" else 2.5
        (tw, th), _ = cv2.getTextSize(num, FONT, scale, 4)
        shadow_text(frame, num, (fw // 2 - tw // 2, fh // 2 + th // 2), FONT, scale, C_WIN, 4)

    def _handle_playing(self, frame, p1_pos, p2_pos, dt, now):
        fw, fh = self._fw, self._fh
        # Update paddles
        if p1_pos and self._p1:
            self._p1.set_target(*p1_pos)
        if p2_pos and self._p2:
            self._p2.set_target(*p2_pos)
        if self._p1:
            self._p1.update(dt)
        if self._p2:
            self._p2.update(dt)

        # Update ball
        result = self._ball.update(dt, self._p1, self._p2)

        if result == "hit":
            self._hit_flash_t = now

        elif result == "goal_right":
            # Ball exited LEFT side — right player (P2) scores
            self._score[1] += 1
            self._goal_scorer = 2
            self._ball.reset(towards=-1)   # serve toward P1 (left, the loser)
            self._state    = "goal"
            self._state_ts = now

        elif result == "goal_left":
            # Ball exited RIGHT side — left player (P1) scores
            self._score[0] += 1
            self._goal_scorer = 1
            self._ball.reset(towards=1)    # serve toward P2 (right, the loser)
            self._state    = "goal"
            self._state_ts = now

        # Hit flash
        if now - self._hit_flash_t < 0.08:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (self._fw, self._fh), (200, 200, 255), -1)
            cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)

        # Check win
        if max(self._score) >= SCORE_WIN:
            self._state    = "win"
            self._state_ts = now

        # Speed readout (subtle)
        spd_txt = f"{self._ball.speed:.0f} px/s"
        cv2.putText(frame, spd_txt, (self._fw // 2 - 30, self._fh - 8),
                    FONT_MONO, 0.45, (80, 80, 100), 1, cv2.LINE_AA)

    def _handle_goal(self, frame, now):
        fw, fh = self._fw, self._fh
        elapsed = now - self._state_ts

        col  = C_P1 if self._goal_scorer == 1 else C_P2
        name = f"P{self._goal_scorer} SCORES!"
        (tw, th), _ = cv2.getTextSize(name, FONT, 1.6, 3)

        t = math.sin(now * 8) * 0.5 + 0.5
        pulse_col = tuple(int(c * (0.6 + 0.4 * t)) for c in col)
        shadow_text(frame, name, (fw // 2 - tw // 2, fh // 2 + th // 2),
                    FONT, 1.6, pulse_col, 3)

        if elapsed > 2.0:
            self._state = "playing"

    def _handle_win(self, frame, now):
        fw, fh = self._fw, self._fh
        winner = 1 if self._score[0] >= SCORE_WIN else 2
        col    = C_P1 if winner == 1 else C_P2

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, fh), (10, 8, 25), -1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

        t = math.sin(now * 5) * 0.5 + 0.5
        pulse = tuple(int(c * (0.7 + 0.3 * t)) for c in col)

        win_txt = f"P{winner}  WINS!"
        (tw, th), _ = cv2.getTextSize(win_txt, FONT, 2.2, 4)
        shadow_text(frame, win_txt, (fw // 2 - tw // 2, fh // 2 - 20), FONT, 2.2, pulse, 4)

        sub = "Show both hands to play again"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.65, 1)
        shadow_text(frame, sub, (fw // 2 - sw // 2, fh // 2 + 45), FONT, 0.65, C_TEXT)

        # Both hands visible → restart
        p1_pos, p2_pos = self._find_hands([], [])   # just check state
        if now - self._state_ts > 3.0:
            # Check real detections outside — we rely on the calling frame having them
            # so just wait for next on_frame call to pass hands; reset after 3s pause.
            pass
        if now - self._state_ts > 4.0:
            self._reset_game()

    # ── Hand detection ────────────────────────────────────────

    def _find_hands(
        self,
        keypoints: list,
        detections: list,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """
        Returns (p1_pos, p2_pos) where each is (x, y) in frame coords or None.

        Priority: wrist keypoint (kp 0) > detection box centre.
        Split: x < fw/2 → P1 (left),  x >= fw/2 → P2 (right).
        """
        fw, fh = self._fw, self._fh
        mid = fw / 2

        candidates_left:  list[tuple[float, float]] = []
        candidates_right: list[tuple[float, float]] = []

        # Pose keypoints — wrist is kp 0
        for instance in keypoints:
            if not instance:
                continue
            x, y, conf = instance[0]
            if conf < 0.25:
                # Fall back to centroid of all confident kps
                good = [(kx, ky) for kx, ky, kc in instance if kc > 0.25]
                if not good:
                    continue
                x = sum(p[0] for p in good) / len(good)
                y = sum(p[1] for p in good) / len(good)
            if x < mid:
                candidates_left.append((x, y))
            else:
                candidates_right.append((x, y))

        # Detection boxes — centre point
        if not candidates_left or not candidates_right:
            for (label, conf, x1, y1, x2, y2) in detections:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if cx < mid and not candidates_left:
                    candidates_left.append((cx, cy))
                elif cx >= mid and not candidates_right:
                    candidates_right.append((cx, cy))

        p1 = candidates_left[0]  if candidates_left  else None
        p2 = candidates_right[0] if candidates_right else None
        return p1, p2

    # ── Drawing ───────────────────────────────────────────────

    def _draw_court(self, frame: np.ndarray):
        fw, fh = self._fw, self._fh

        # Goal zones — semi-transparent tinted strips over live video
        goal_w = PADDLE_MARGIN
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (goal_w, fh), C_P1, -1)
        cv2.rectangle(ov, (fw - goal_w, 0), (fw, fh), C_P2, -1)
        cv2.addWeighted(ov, 0.30, frame, 0.70, 0, frame)

        # Goal lines
        cv2.line(frame, (goal_w, 0), (goal_w, fh), C_P1, 2, cv2.LINE_AA)
        cv2.line(frame, (fw - goal_w, 0), (fw - goal_w, fh), C_P2, 2, cv2.LINE_AA)

        # Centre dashed line
        dash = 18;  gap = 10
        y = 0
        while y < fh:
            cv2.line(frame, (fw // 2, y), (fw // 2, min(y + dash, fh)), C_NET, 1, cv2.LINE_AA)
            y += dash + gap

        # Top / bottom border lines only (no fill)
        cv2.line(frame, (0, 0),      (fw, 0),      C_COURT, 2)
        cv2.line(frame, (0, fh - 1), (fw, fh - 1), C_COURT, 2)

    def _draw_hud(self, frame: np.ndarray):
        fw, fh = self._fw, self._fh

        # Scores
        s1 = str(self._score[0])
        s2 = str(self._score[1])
        scale = 2.2
        (w1, h1), _ = cv2.getTextSize(s1, FONT, scale, 3)
        (w2, h2), _ = cv2.getTextSize(s2, FONT, scale, 3)

        shadow_text(frame, s1, (fw // 2 - 60 - w1, 50), FONT, scale, C_P1, 3)
        shadow_text(frame, s2, (fw // 2 + 60,       50), FONT, scale, C_P2, 3)

        # Player labels (top corners)
        shadow_text(frame, "P1", (12, 28),        FONT, 0.65, C_P1)
        shadow_text(frame, "P2", (fw - 45, 28),   FONT, 0.65, C_P2)

    def _draw_xy_toggle_hint(self, frame: np.ndarray):
        """XY mode LED indicator — small dot + label in top-centre of screen."""
        fw = self._fw
        led_x, led_y = fw // 2, 18
        if self._xy_mode:
            # Bright glowing dot
            ov = frame.copy()
            cv2.circle(ov, (led_x - 38, led_y), 9, C_READY_OK, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
            cv2.circle(frame, (led_x - 38, led_y), 6, C_READY_OK, -1, cv2.LINE_AA)
            cv2.putText(frame, "XY ON", (led_x - 28, led_y + 5),
                        FONT_MONO, 0.52, C_READY_OK, 1, cv2.LINE_AA)
        else:
            # Dim dot
            cv2.circle(frame, (led_x - 38, led_y), 6, (50, 70, 50), -1, cv2.LINE_AA)
            cv2.putText(frame, "XY OFF", (led_x - 28, led_y + 5),
                        FONT_MONO, 0.52, (70, 80, 70), 1, cv2.LINE_AA)

    # ── Internal ──────────────────────────────────────────────

    def _reset_game(self):
        fw, fh = self._fw, self._fh
        self._p1   = Paddle("left",  fw, fh, self._xy_mode)
        self._p2   = Paddle("right", fw, fh, self._xy_mode)
        self._ball = Ball(fw, fh)
        self._score = [0, 0]
        self._state = "waiting"
        self._p1_seen_since = None
        self._p2_seen_since = None
        self._hit_flash_t   = 0.0
        self._t_last        = time.perf_counter()

    # ── XY mode toggle (called by streamer keyboard hook if wired up,
    #    or you can expose it via the game_status label) ──────
    def reset(self):
        self._reset_game()

    def toggle_xy(self):
        self._xy_mode = not self._xy_mode
        if self._p1:
            self._p1.xy_mode = self._xy_mode
        if self._p2:
            self._p2.xy_mode = self._xy_mode
