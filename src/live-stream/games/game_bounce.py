#!/usr/bin/env python3
"""
game_bounce.py — Multi-Ball Hand Bounce
-----------------------------------------
  A / D        : add / remove a ball
  Open hand    : slap balls (velocity from hand movement)
  Closed fist  : catch / hold any ball you touch — open hand to release it
  SPACE        : pause  (via GamePlugin)

Physics
-------
  Ball-hand: the ball reflects off the hand's surface using the hand's
  own velocity.  If the hand is stationary the ball bounces like hitting
  a wall (no energy added).  Only moving hands add speed.

  Ball-ball: elastic circle collision — momentum is conserved exactly,
  no energy injection.

  A per-hand cooldown prevents a single contact from firing dozens of
  times per second while the surfaces overlap.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

import cv2
import numpy as np

from game_plugin import GamePlugin


# ── Helpers ───────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _lerp(a, b, t):
    return a + (b - a) * t

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ── Ball ─────────────────────────────────────────────────────

class Ball:
    RADIUS    = 22
    FRICTION  = 0.992        # per-frame speed multiplier
    MIN_SPEED = 80           # px/s — never fully stops
    MAX_SPEED = 1600         # px/s hard cap

    # Colour pool — each ball gets one
    COLOURS = [
        (255, 220,  40),   # gold
        (100, 220, 255),   # cyan
        (255, 100, 160),   # pink
        (80,  255, 140),   # green
        (255, 160,  60),   # orange
        (180, 120, 255),   # purple
    ]

    _colour_idx = 0

    def __init__(self, fw: int, fh: int):
        cx, cy = fw / 2, fh / 2
        self.x = cx + random.uniform(-cx * 0.4, cx * 0.4)
        self.y = cy + random.uniform(-cy * 0.4, cy * 0.4)
        angle  = random.uniform(0, 2 * math.pi)
        spd    = random.uniform(250, 500)
        self.vx = math.cos(angle) * spd
        self.vy = math.sin(angle) * spd
        self.r  = self.RADIUS
        self.colour = self.COLOURS[Ball._colour_idx % len(self.COLOURS)]
        Ball._colour_idx += 1
        self.trail: list[tuple[float, float]] = []
        self.flash_t = -99.0
        # Catch state
        self.caught    = False
        self.catch_off = (0.0, 0.0)   # offset from catching hand centre

    def update(self, dt: float, fw: int, fh: int):
        if self.caught:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > 14:
            self.trail.pop(0)

        self.vx *= self.FRICTION
        self.vy *= self.FRICTION

        spd = math.hypot(self.vx, self.vy)
        if spd < self.MIN_SPEED:
            if spd < 1e-3:
                ang = random.uniform(0, 2 * math.pi)
                self.vx = math.cos(ang) * self.MIN_SPEED
                self.vy = math.sin(ang) * self.MIN_SPEED
            else:
                scale = self.MIN_SPEED / spd
                self.vx *= scale;  self.vy *= scale

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Walls
        if self.x - self.r < 0:
            self.x = float(self.r);   self.vx =  abs(self.vx)
        if self.x + self.r > fw:
            self.x = float(fw - self.r); self.vx = -abs(self.vx)
        if self.y - self.r < 0:
            self.y = float(self.r);   self.vy =  abs(self.vy)
        if self.y + self.r > fh:
            self.y = float(fh - self.r); self.vy = -abs(self.vy)

    def draw(self, frame: np.ndarray, now: float):
        r  = self.r
        bx = int(self.x);  by = int(self.y)

        # Trail
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            a   = (i / max(n - 1, 1)) * 0.45
            tr  = max(2, int(r * (i / max(n - 1, 1)) * 0.65))
            ov  = frame.copy()
            cv2.circle(ov, (int(tx), int(ty)), tr, self.colour, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a * 0.5, frame, 1 - a * 0.5, 0, frame)

        flash = (now - self.flash_t) < 0.10
        gcol  = (255, 255, 255) if flash else tuple(min(255, c + 80) for c in self.colour)
        gr    = r + 10 if flash else r + 5
        ov = frame.copy()
        cv2.circle(ov, (bx, by), gr, gcol, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.30, frame, 0.70, 0, frame)

        col = (220, 240, 255) if flash else self.colour
        cv2.circle(frame, (bx, by), r, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx - r // 3, by - r // 3), max(3, r // 4),
                   (255, 255, 255), -1, cv2.LINE_AA)

        if self.caught:
            cv2.circle(frame, (bx, by), r + 4, (255, 255, 255), 1, cv2.LINE_AA)


# ── Hand tracker ─────────────────────────────────────────────

class Hand:
    """Tracks one logical hand: position, velocity, open/closed state."""
    HIT_COOL   = 0.18    # seconds between hits on the same ball from this hand
    CATCH_RAD  = 50      # px — must be this close to catch

    def __init__(self):
        self.x    = 0.0;  self.y    = 0.0
        self.vx   = 0.0;  self.vy   = 0.0
        self.px   = 0.0;  self.py   = 0.0    # previous position
        self.active  = False
        self.closed  = False
        self.speed   = 0.0
        self._cool: dict[int, float] = {}    # ball_id → last hit time

    def update(self, x: float, y: float, closed: bool, dt: float, now: float):
        self.px, self.py = self.x, self.y
        alpha = _clamp(0.30 + dt * 5, 0, 1)
        self.x = _lerp(self.x, x, alpha)
        self.y = _lerp(self.y, y, alpha)
        raw_vx = (self.x - self.px) / max(dt, 1e-4)
        raw_vy = (self.y - self.py) / max(dt, 1e-4)
        self.vx = _lerp(self.vx, raw_vx, 0.35)
        self.vy = _lerp(self.vy, raw_vy, 0.35)
        self.speed  = math.hypot(self.vx, self.vy)
        self.closed = closed
        self.active = True
        # Expire old cooldowns
        self._cool = {bid: t for bid, t in self._cool.items() if now - t < self.HIT_COOL}

    def can_hit(self, ball_id: int, now: float) -> bool:
        return now - self._cool.get(ball_id, -999) >= self.HIT_COOL

    def mark_hit(self, ball_id: int, now: float):
        self._cool[ball_id] = now


# ── Game ─────────────────────────────────────────────────────

class Game(GamePlugin):

    MAX_BALLS   = 8
    IMPULSE_MIN = 120    # px/s — minimum hand speed to add any energy
    IMPULSE_MAX = 2200   # px/s — hand speed that gives full energy transfer

    def __init__(self):
        super().__init__()
        self._fw     = 640
        self._fh     = 480
        self._balls: list[Ball] = []
        self._hands  = [Hand(), Hand()]   # up to 2 hands
        self._score  = 0
        self._t_last = time.perf_counter()
        self._started = False

    # ── API ───────────────────────────────────────────────────

    def on_start(self, fw, fh, class_names=None):
        self._fw, self._fh = fw, fh
        self._balls  = [Ball(fw, fh)]
        self._score  = 0
        self._t_last = time.perf_counter()
        self._started = True

    def on_frame(self, frame, keypoints, detections, fw, fh):
        if not self._started:
            self.on_start(fw, fh)

        now = time.perf_counter()
        dt  = min(now - self._t_last, 0.07)
        self._t_last = now
        self._fw, self._fh = fw, fh

        # ── Parse hands ──────────────────────────────────────
        hand_data = self._parse_hands(keypoints, detections, fw, fh)
        for i, hd in enumerate(hand_data[:2]):
            if hd is not None:
                hx, hy, closed = hd
                self._hands[i].update(hx, hy, closed, dt, now)
            else:
                self._hands[i].active = False

        # ── Ball-hand interaction ─────────────────────────────
        active_hands = [h for h in self._hands if h.active]
        for ball in self._balls:
            if ball.caught:
                # Find which hand is holding it
                holding = None
                for h in active_hands:
                    dx = ball.x - h.x;  dy = ball.y - h.y
                    if math.hypot(dx, dy) < Ball.RADIUS + Hand.CATCH_RAD + 10:
                        holding = h
                        break
                if holding is None or not holding.closed:
                    # Release
                    ball.caught = False
                    ball.vx = holding.vx if holding else 0.0
                    ball.vy = holding.vy if holding else 0.0
                    spd = math.hypot(ball.vx, ball.vy)
                    if spd < ball.MIN_SPEED:
                        ball.vx = ball.vx / max(spd, 1) * ball.MIN_SPEED
                        ball.vy = ball.vy / max(spd, 1) * ball.MIN_SPEED
                else:
                    # Drag with hand
                    ball.x = holding.x + ball.catch_off[0]
                    ball.y = holding.y + ball.catch_off[1]
                continue

            bid = id(ball)
            for h in active_hands:
                dist = math.hypot(ball.x - h.x, ball.y - h.y)
                contact_r = Ball.RADIUS + Hand.CATCH_RAD

                if dist > contact_r:
                    continue

                # ── Catch (closed fist + touching) ──────────
                if h.closed and h.can_hit(bid, now):
                    ball.caught    = True
                    ball.catch_off = (ball.x - h.x, ball.y - h.y)
                    ball.vx = 0.0;  ball.vy = 0.0
                    h.mark_hit(bid, now)
                    ball.flash_t = now
                    self._score += 1
                    break

                # ── Slap (open hand) ─────────────────────────
                if not h.closed and h.can_hit(bid, now):
                    # Normal from hand centre to ball
                    if dist < 1:
                        nx, ny = 1.0, 0.0
                    else:
                        nx = (ball.x - h.x) / dist
                        ny = (ball.y - h.y) / dist

                    # Reflect ball velocity off the hand surface
                    dot = ball.vx * nx + ball.vy * ny
                    # Remove component going INTO the hand
                    ball.vx -= 2 * dot * nx
                    ball.vy -= 2 * dot * ny

                    # Add hand velocity — only the component along the normal,
                    # and only if hand is actually moving toward the ball
                    hand_dot = h.vx * nx + h.vy * ny   # positive = moving away
                    if hand_dot < 0:                    # hand moving INTO ball
                        spd_contrib = min(abs(hand_dot), self.IMPULSE_MAX)
                        # Scale 0..1 over IMPULSE_MIN..IMPULSE_MAX
                        t = _clamp(
                            (spd_contrib - self.IMPULSE_MIN) /
                            max(self.IMPULSE_MAX - self.IMPULSE_MIN, 1),
                            0.0, 1.0)
                        # Only add energy if hand moving faster than ball surface
                        energy_add = spd_contrib * t
                        ball.vx += nx * energy_add
                        ball.vy += ny * energy_add

                    # Cap speed
                    spd = math.hypot(ball.vx, ball.vy)
                    if spd > Ball.MAX_SPEED:
                        ball.vx = ball.vx / spd * Ball.MAX_SPEED
                        ball.vy = ball.vy / spd * Ball.MAX_SPEED

                    # Push ball out of contact zone to prevent re-triggering
                    overlap = contact_r - dist + 2
                    ball.x += nx * overlap
                    ball.y += ny * overlap

                    h.mark_hit(bid, now)
                    ball.flash_t = now
                    self._score += 1

        # ── Ball-ball elastic collisions ─────────────────────
        for i in range(len(self._balls)):
            for j in range(i + 1, len(self._balls)):
                a, b = self._balls[i], self._balls[j]
                if a.caught and b.caught:
                    continue
                dx   = b.x - a.x;  dy = b.y - a.y
                dist = math.hypot(dx, dy)
                min_d = a.r + b.r
                if dist < min_d and dist > 1e-3:
                    # Separate
                    nx  = dx / dist;  ny = dy / dist
                    overlap = min_d - dist
                    if not a.caught:
                        a.x -= nx * overlap * 0.5
                        a.y -= ny * overlap * 0.5
                    if not b.caught:
                        b.x += nx * overlap * 0.5
                        b.y += ny * overlap * 0.5

                    # Elastic 1D collision along normal (equal masses)
                    if not a.caught and not b.caught:
                        av = a.vx * nx + a.vy * ny
                        bv = b.vx * nx + b.vy * ny
                        # Swap normal components
                        a.vx += (bv - av) * nx;  a.vy += (bv - av) * ny
                        b.vx += (av - bv) * nx;  b.vy += (av - bv) * ny

        # ── Physics ───────────────────────────────────────────
        for ball in self._balls:
            ball.update(dt, fw, fh)

        # ── Draw ─────────────────────────────────────────────
        for ball in self._balls:
            ball.draw(frame, now)
        self._draw_hands(frame, active_hands, now)
        self._draw_hud(frame, fw)

        return frame

    def on_stop(self):
        self._started = False

    def tune(self, key: str):
        """A = add ball, D = remove ball."""
        if key.lower() == 'a' and len(self._balls) < self.MAX_BALLS:
            self._balls.append(Ball(self._fw, self._fh))
        elif key.lower() == 'd' and len(self._balls) > 1:
            self._balls.pop()

    # ── Hand parsing ─────────────────────────────────────────

    def _parse_hands(self, keypoints, detections, fw, fh):
        """Return up to 2 (x, y, closed) tuples or None."""
        results = []

        for inst in keypoints:
            if not inst:
                continue
            good = [(x, y, c) for x, y, c in inst if c > 0.20]
            if not good:
                continue
            cx = sum(p[0] for p in good) / len(good)
            cy = sum(p[1] for p in good) / len(good)
            # Open/closed: tip-to-wrist ratio (same logic as angry hands)
            closed = _is_fist(inst)
            results.append((cx, cy, closed))
            if len(results) == 2:
                break

        # Fallback: detection boxes = closed (no keypoints = fist)
        if not results:
            for (label, conf, x1, y1, x2, y2) in detections:
                if conf > 0.25:
                    results.append(((x1+x2)/2, (y1+y2)/2, True))
                if len(results) == 2:
                    break

        # Pad to length 2
        while len(results) < 2:
            results.append(None)
        return results

    # ── Drawing ───────────────────────────────────────────────

    def _draw_hands(self, frame, hands, now):
        for h in hands:
            col = (60, 60, 220) if h.closed else (80, 220, 80)
            # Outer ring
            ov = frame.copy()
            cv2.circle(ov, (int(h.x), int(h.y)), Hand.CATCH_RAD, col, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
            cv2.circle(frame, (int(h.x), int(h.y)), Hand.CATCH_RAD, col, 2, cv2.LINE_AA)
            # Centre dot
            cv2.circle(frame, (int(h.x), int(h.y)), 5, (255,255,255), -1, cv2.LINE_AA)
            # Label
            lbl = "FIST" if h.closed else "OPEN"
            cv2.putText(frame, lbl, (int(h.x) - 18, int(h.y) - Hand.CATCH_RAD - 6),
                        FONTS, 0.45, col, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, fw):
        # Top banner
        banner_h = 40
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(fw, banner_h),(10,8,22),-1)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

        cv2.putText(frame, "BOUNCE", (fw//2 - 42, 28), FONT, 0.85, (200,200,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hits: {self._score}", (fw - 110, 28), FONTS, 0.65, (100,255,180), 1, cv2.LINE_AA)

        # Ball count + keys
        n = len(self._balls)
        cv2.putText(frame, f"A/D  balls: {n}/{self.MAX_BALLS}", (8, 28),
                    FONTS, 0.52, (180,180,220), 1, cv2.LINE_AA)


# ── Fist detection (reused from angry hands logic) ────────────

_TIP_IDS     = [4, 8, 12, 16, 20]
_KNUCKLE_IDS = [1, 5,  9, 13, 17]

def _is_fist(inst, thresh=0.22, ratio_thresh=1.15, min_tips=3) -> bool:
    """True if hand appears closed (tips not further than knuckles from wrist)."""
    if not inst or len(inst) < 21:
        return True   # no data = assume fist
    wx, wy, wc = inst[0]
    if wc < thresh:
        return True
    extended = 0
    for tip_id, knuck_id in zip(_TIP_IDS, _KNUCKLE_IDS):
        tx, ty, tc = inst[tip_id]
        kx, ky, kc = inst[knuck_id]
        if tc < thresh or kc < thresh:
            continue
        tip_d   = math.hypot(tx - wx, ty - wy)
        knuck_d = math.hypot(kx - wx, ky - wy)
        if knuck_d < 1:
            continue
        if tip_d / knuck_d > ratio_thresh:
            extended += 1
    return extended < min_tips