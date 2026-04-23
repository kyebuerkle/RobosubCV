#!/usr/bin/env python3
"""
game_bounce.py — Multi-Ball Hand Bounce
-----------------------------------------
  A / D        : add / remove a ball
  Open hand    : slap balls with keypoints (fingertips hit individually)
  Closed fist  : catch / hold any ball — open hand to release
  SPACE        : pause

Physics
-------
  Keypoint-based hitting: each visible fingertip/knuckle is its own
  collision point. Moving points impart velocity; stationary ones just
  reflect.

  Fist detection uses a majority-vote window — needs > half the recent
  frames to agree before state flips. Same for hand movement — only
  counts as "moving" if displacement is consistently above a threshold.

  Ball-ball elastic collision preserves momentum exactly.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque

import cv2
import numpy as np

from game_plugin import GamePlugin


# ── Helpers ───────────────────────────────────────────────────

def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _lerp(a, b, t):    return a + (b - a) * t

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX

# 21-pt hand keypoint indices used for hitting
# Fingertips and mid-finger joints — the parts that actually stick out
HIT_KP_IDS   = [4, 8, 12, 16, 20,    # fingertips
                 3, 7, 11, 15, 19]    # DIP joints (second-to-tip)
WRIST_ID     = 0
TIP_IDS      = [4, 8, 12, 16, 20]
KNUCKLE_IDS  = [1, 5,  9, 13, 17]


# ── Fist detection ────────────────────────────────────────────

def _raw_is_fist(inst, thresh=0.18) -> bool:
    """
    Fist detection based on keypoint spread relative to hand size.

    Open hand: fingertips are spread far from the palm centre.
    Closed fist: all keypoints cluster tightly — the bounding box
    of confident keypoints is small relative to the wrist-to-middle-
    knuckle distance (which stays roughly constant regardless of pose).

    This is more robust than tip/knuckle ratios on low-confidence models
    because it only needs the *overall pattern* of points, not specific tips.
    """
    if not inst or len(inst) < 21:
        return True

    good = [(x, y) for x, y, c in inst if c > thresh]
    if len(good) < 5:
        return True   # too few visible points — assume closed

    # Hand size reference: wrist (0) → middle MCP (9)
    wx, wy = inst[0][0], inst[0][1]
    mx, my = inst[9][0], inst[9][1]
    ref = math.hypot(mx - wx, my - wy)
    if ref < 1:
        return True

    # Spread = max distance between any two confident keypoints
    max_dist = 0.0
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            d = math.hypot(good[i][0]-good[j][0], good[i][1]-good[j][1])
            if d > max_dist:
                max_dist = d

    # Normalised spread: open hand ≈ 1.8-2.5×, fist ≈ 0.8-1.2×
    ratio = max_dist / ref
    return ratio < 1.45   # below this = fist


# ── Keypoint tracker ──────────────────────────────────────────

class KPTracker:
    """
    Tracks a single keypoint across frames.
    Smooths position and velocity.
    Provides majority-vote 'moving' state.
    """
    MOVE_THRESH   = 5.0      # px/frame — below this = stationary
    VOTE_WINDOW   = 5        # frames for majority vote

    def __init__(self, x: float, y: float):
        self.x  = x;  self.y  = y
        self.px = x;  self.py = y
        self.vx = 0.0; self.vy = 0.0
        self._move_votes: deque[bool] = deque(maxlen=self.VOTE_WINDOW)
        self.moving = False

    def update(self, x: float, y: float, dt: float):
        self.px, self.py = self.x, self.y
        alpha = _clamp(0.35 + dt * 4, 0, 1)
        self.x = _lerp(self.x, x, alpha)
        self.y = _lerp(self.y, y, alpha)
        raw_vx = (self.x - self.px) / max(dt, 1e-4)
        raw_vy = (self.y - self.py) / max(dt, 1e-4)
        self.vx = _lerp(self.vx, raw_vx, 0.40)
        self.vy = _lerp(self.vy, raw_vy, 0.40)
        pixel_disp = math.hypot(self.x - self.px, self.y - self.py)
        self._move_votes.append(pixel_disp > self.MOVE_THRESH)
        # Majority vote
        self.moving = sum(self._move_votes) > len(self._move_votes) / 2

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


# ── Sweep / tunneling detection ──────────────────────────────

def _segment_ball_intersect(
    px: float, py: float,   # keypoint prev position
    cx: float, cy: float,   # keypoint current position
    bx: float, by: float,   # ball centre
    r:  float,              # ball radius + kp hit radius
) -> tuple[float, float, float] | None:
    """
    Return (hit_x, hit_y, t) if the segment p→c passes within r of the ball,
    where t is the parametric position along the segment (0=start, 1=end).
    Used to catch fast-moving keypoints that tunnel through the ball between frames.
    Returns None if no intersection.
    """
    dx = cx - px;  dy = cy - py
    fx = px - bx;  fy = py - by
    a = dx*dx + dy*dy
    if a < 1e-6:
        return None
    b = 2 * (fx*dx + fy*dy)
    c_coef = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c_coef
    if disc < 0:
        return None
    disc_r = math.sqrt(disc)
    t1 = (-b - disc_r) / (2*a)
    t2 = (-b + disc_r) / (2*a)
    # Pick the entry point that lies within the segment
    for t in (t1, t2):
        if 0.0 <= t <= 1.0:
            hx = px + t*dx;  hy = py + t*dy
            return hx, hy, t
    return None


# ── Hand tracker ─────────────────────────────────────────────

class Hand:
    """
    Tracks one hand. Maintains:
      - per-keypoint KPTracker objects
      - majority-vote fist state
      - per-ball hit cooldowns
    """
    FIST_WINDOW  = 6     # frames for fist majority vote
    HIT_COOL     = 0.05  # seconds between hits — ~every other frame at 30fps
    CATCH_RAD    = 48    # px — fist catch radius (from wrist/palm centre)
    KP_HIT_RAD   = 28    # px — per-keypoint hit radius

    def __init__(self):
        self.active  = False
        self.closed  = False          # majority-voted fist state
        self.cx = 0.0; self.cy = 0.0  # hand centre (wrist or centroid)
        self._prev_cx = 0.0; self._prev_cy = 0.0
        self.kps: dict[int, KPTracker] = {}   # kid → tracker
        self._fist_votes: deque[bool] = deque(maxlen=self.FIST_WINDOW)
        self._cool: dict[tuple, float] = {}   # (ball_id, kp_id) → timestamp

    def update(self, inst: list, dt: float, now: float):
        raw_fist = _raw_is_fist(inst)
        self._fist_votes.append(raw_fist)
        self.closed = sum(self._fist_votes) > len(self._fist_votes) / 2

        # Store prev palm position for speed check in catch
        self._prev_cx = self.cx
        self._prev_cy = self.cy

        # Hand centre from wrist or centroid
        good = [(x, y, c) for x, y, c in inst if c > 0.20]
        if good:
            self.cx = sum(p[0] for p in good) / len(good)
            self.cy = sum(p[1] for p in good) / len(good)

        # Update keypoint trackers for hit points
        for kid in HIT_KP_IDS:
            if kid >= len(inst):
                continue
            x, y, conf = inst[kid]
            if conf < 0.22:
                continue
            if kid not in self.kps:
                self.kps[kid] = KPTracker(x, y)
            else:
                self.kps[kid].update(x, y, dt)

        # Expire old cooldowns
        self._cool = {k: t for k, t in self._cool.items()
                      if now - t < self.HIT_COOL}
        self.active = True

    def can_hit(self, ball_id: int, kp_id: int, now: float) -> bool:
        return now - self._cool.get((ball_id, kp_id), -999) >= self.HIT_COOL

    def mark_hit(self, ball_id: int, kp_id: int, now: float):
        self._cool[(ball_id, kp_id)] = now


# ── Ball ─────────────────────────────────────────────────────

class Ball:
    RADIUS    = 22
    FRICTION  = 0.990
    MIN_SPEED = 80
    MAX_SPEED = 1600

    COLOURS = [
        (255, 220,  40), (100, 220, 255), (255, 100, 160),
        (80,  255, 140), (255, 160,  60), (180, 120, 255),
    ]
    _cidx = 0

    def __init__(self, fw: int, fh: int):
        cx, cy = fw / 2, fh / 2
        self.x = cx + random.uniform(-cx * 0.4, cx * 0.4)
        self.y = cy + random.uniform(-cy * 0.4, cy * 0.4)
        ang    = random.uniform(0, 2 * math.pi)
        spd    = random.uniform(220, 480)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd
        self.r  = self.RADIUS
        self.colour  = self.COLOURS[Ball._cidx % len(self.COLOURS)]
        Ball._cidx  += 1
        self.trail:  list[tuple[float, float]] = []
        self.flash_t = -99.0
        self.caught  = False
        self.catch_off = (0.0, 0.0)

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
                self.vx = self.vx / spd * self.MIN_SPEED
                self.vy = self.vy / spd * self.MIN_SPEED

        self.x += self.vx * dt;  self.y += self.vy * dt

        if self.x - self.r < 0:
            self.x = float(self.r);     self.vx =  abs(self.vx)
        if self.x + self.r > fw:
            self.x = float(fw - self.r); self.vx = -abs(self.vx)
        if self.y - self.r < 0:
            self.y = float(self.r);     self.vy =  abs(self.vy)
        if self.y + self.r > fh:
            self.y = float(fh - self.r); self.vy = -abs(self.vy)

    def draw(self, frame: np.ndarray, now: float):
        bx, by = int(self.x), int(self.y)
        r = self.r
        for i, (tx, ty) in enumerate(self.trail):
            a  = (i / max(len(self.trail)-1, 1)) * 0.45
            tr = max(2, int(r * (i / max(len(self.trail)-1, 1)) * 0.65))
            ov = frame.copy()
            cv2.circle(ov, (int(tx), int(ty)), tr, self.colour, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a*0.5, frame, 1-a*0.5, 0, frame)

        flash = (now - self.flash_t) < 0.09
        gcol  = (255,255,255) if flash else tuple(min(255,c+80) for c in self.colour)
        ov = frame.copy()
        cv2.circle(ov, (bx,by), r+8 if flash else r+5, gcol, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
        cv2.circle(frame, (bx,by), r, (220,240,255) if flash else self.colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx-r//3, by-r//3), max(3,r//4), (255,255,255), -1, cv2.LINE_AA)
        if self.caught:
            cv2.circle(frame, (bx,by), r+4, (255,255,255), 1, cv2.LINE_AA)


# ── Game ─────────────────────────────────────────────────────

class Game(GamePlugin):

    MAX_BALLS    = 8
    # Hand speed thresholds for impulse scaling
    MOVE_PX_S    = 80     # px/s — below this = treat as stationary (no energy add)
    MAX_IMPULSE  = 1800   # px/s — hand speed that gives maximum energy transfer

    def __init__(self):
        super().__init__()
        self._fw = 640;  self._fh = 480
        self._balls: list[Ball] = []
        self._hands = [Hand(), Hand()]
        self._score = 0
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

        # ── Update hand trackers ──────────────────────────────
        active_hands: list[Hand] = []
        for i, inst in enumerate(keypoints[:2]):
            if inst:
                self._hands[i].update(inst, dt, now)
                active_hands.append(self._hands[i])
            else:
                self._hands[i].active = False

        # Fallback: detection boxes
        if not active_hands:
            for j, (label, conf, x1, y1, x2, y2) in enumerate(detections[:2]):
                if conf > 0.25:
                    fake_inst = [(((x1+x2)/2), ((y1+y2)/2), 0.1)] * 21
                    self._hands[j].update(fake_inst, dt, now)
                    self._hands[j].closed = True   # box only = fist
                    active_hands.append(self._hands[j])

        # ── Ball-hand interaction ─────────────────────────────
        for ball in self._balls:
            bid = id(ball)

            if ball.caught:
                # Find holding hand (by proximity to centre)
                holder = None
                for h in active_hands:
                    if math.hypot(ball.x - h.cx, ball.y - h.cy) < Hand.CATCH_RAD + Ball.RADIUS + 15:
                        holder = h
                        break
                # Release if hand opened or moved away
                if holder is None or not holder.closed:
                    ball.caught = False
                    if holder:
                        # Release with hand velocity from wrist tracker if available
                        if WRIST_ID in holder.kps:
                            kp = holder.kps[WRIST_ID]
                            ball.vx = kp.vx;  ball.vy = kp.vy
                        spd = math.hypot(ball.vx, ball.vy)
                        if spd < Ball.MIN_SPEED:
                            ang = math.atan2(ball.vy, ball.vx)
                            ball.vx = math.cos(ang) * Ball.MIN_SPEED
                            ball.vy = math.sin(ang) * Ball.MIN_SPEED
                else:
                    ball.x = holder.cx + ball.catch_off[0]
                    ball.y = holder.cy + ball.catch_off[1]
                continue

            for h in active_hands:
                # ── Fist catch: closed AND hand moving slowly ────
                if h.closed:
                    dist = math.hypot(ball.x - h.cx, ball.y - h.cy)
                    # Only catch if palm is close AND hand is not sweeping fast
                    palm_speed = math.hypot(
                        h.cx - h._prev_cx, h.cy - h._prev_cy) / max(dt, 1e-4)
                    slow_enough = palm_speed < 200   # px/s threshold to catch
                    if dist < Hand.CATCH_RAD + Ball.RADIUS and slow_enough                             and h.can_hit(bid, -1, now):
                        ball.caught = True
                        ball.catch_off = (ball.x - h.cx, ball.y - h.cy)
                        ball.vx = 0.0;  ball.vy = 0.0
                        ball.flash_t = now
                        h.mark_hit(bid, -1, now)
                        self._score += 1
                    elif h.closed and not slow_enough:
                        # Moving fist = solid object — reflect ball off it
                        if dist < Hand.CATCH_RAD + Ball.RADIUS and h.can_hit(bid, -1, now):
                            if dist < 1: nx, ny = 1.0, 0.0
                            else:
                                nx = (ball.x - h.cx) / dist
                                ny = (ball.y - h.cy) / dist
                            dot = ball.vx*nx + ball.vy*ny
                            ball.vx -= 2*dot*nx;  ball.vy -= 2*dot*ny
                            # Add fist velocity
                            ball.vx += nx * palm_speed * 0.6
                            ball.vy += ny * palm_speed * 0.6
                            spd = math.hypot(ball.vx, ball.vy)
                            if spd > Ball.MAX_SPEED:
                                ball.vx = ball.vx/spd*Ball.MAX_SPEED
                                ball.vy = ball.vy/spd*Ball.MAX_SPEED
                            overlap = (Hand.CATCH_RAD+Ball.RADIUS) - dist + 2
                            ball.x += nx*overlap;  ball.y += ny*overlap
                            h.mark_hit(bid, -1, now)
                            ball.flash_t = now
                    continue   # fist never slaps via keypoints

                # ── Open hand: per-keypoint hits ─────────────
                for kid, kp in h.kps.items():
                    contact = Hand.KP_HIT_RAD + Ball.RADIUS
                    if not h.can_hit(bid, kid, now):
                        continue

                    dist = math.hypot(ball.x - kp.x, ball.y - kp.y)

                    # ── Sweep / tunnel detection ──────────────
                    # Check if the keypoint path from last frame to this frame
                    # swept through the ball — catches fast hands that jump over it.
                    sweep_hit = None
                    kp_speed = kp.speed
                    if kp_speed > 80 and dist < contact * 3:
                        sweep_hit = _segment_ball_intersect(
                            kp.px, kp.py, kp.x, kp.y,
                            ball.x, ball.y, float(contact))

                    hit_this_kp = (dist <= contact) or (sweep_hit is not None)
                    if not hit_this_kp:
                        continue

                    # Contact point and normal
                    if sweep_hit is not None:
                        # Use the sweep entry point for the normal
                        hx, hy, _ = sweep_hit
                        ddx = ball.x - hx;  ddy = ball.y - hy
                        dn = math.hypot(ddx, ddy)
                        if dn < 1: nx, ny = 1.0, 0.0
                        else:      nx, ny = ddx/dn, ddy/dn
                    elif dist < 1:
                        nx, ny = 1.0, 0.0
                    else:
                        nx = (ball.x - kp.x) / dist
                        ny = (ball.y - kp.y) / dist

                    # Reflect ball velocity off the keypoint surface
                    dot = ball.vx*nx + ball.vy*ny
                    ball.vx -= 2*dot*nx
                    ball.vy -= 2*dot*ny

                    # Add energy: proportional to how fast the keypoint is
                    # moving INTO the ball (along normal), above the dead zone.
                    hand_dot = kp.vx*nx + kp.vy*ny   # neg = kp moving toward ball
                    if hand_dot < -self.MOVE_PX_S:
                        contrib = abs(hand_dot)
                        t_scale = _clamp(
                            (contrib - self.MOVE_PX_S) /
                            max(self.MAX_IMPULSE - self.MOVE_PX_S, 1),
                            0.0, 1.0)
                        energy = contrib * t_scale
                        ball.vx += nx * energy
                        ball.vy += ny * energy

                    # For sweep hits boost by the actual travelled distance
                    # to account for the speed lost by slow inference
                    if sweep_hit is not None:
                        travel_spd = kp_speed
                        if travel_spd > self.MOVE_PX_S:
                            t_scale = _clamp(
                                (travel_spd - self.MOVE_PX_S) /
                                max(self.MAX_IMPULSE - self.MOVE_PX_S, 1),
                                0.0, 1.0)
                            ball.vx += nx * travel_spd * t_scale * 0.7
                            ball.vy += ny * travel_spd * t_scale * 0.7

                    # Cap speed
                    spd = math.hypot(ball.vx, ball.vy)
                    if spd > Ball.MAX_SPEED:
                        ball.vx = ball.vx/spd*Ball.MAX_SPEED
                        ball.vy = ball.vy/spd*Ball.MAX_SPEED

                    # Push ball out of contact zone
                    if dist < contact:
                        overlap = contact - dist + 2
                        ball.x += nx*overlap;  ball.y += ny*overlap

                    h.mark_hit(bid, kid, now)
                    ball.flash_t = now
                    self._score += 1

        # ── Ball-ball elastic collisions ─────────────────────
        for i in range(len(self._balls)):
            for j in range(i + 1, len(self._balls)):
                a, b = self._balls[i], self._balls[j]
                if a.caught and b.caught:
                    continue
                dx = b.x - a.x;  dy = b.y - a.y
                dist = math.hypot(dx, dy)
                min_d = a.r + b.r
                if dist >= min_d or dist < 1e-3:
                    continue
                nx = dx / dist;  ny = dy / dist
                overlap = min_d - dist
                if not a.caught:
                    a.x -= nx * overlap * 0.5;  a.y -= ny * overlap * 0.5
                if not b.caught:
                    b.x += nx * overlap * 0.5;  b.y += ny * overlap * 0.5
                if not a.caught and not b.caught:
                    av = a.vx*nx + a.vy*ny
                    bv = b.vx*nx + b.vy*ny
                    a.vx += (bv-av)*nx;  a.vy += (bv-av)*ny
                    b.vx += (av-bv)*nx;  b.vy += (av-bv)*ny

        # ── Physics ───────────────────────────────────────────
        for ball in self._balls:
            ball.update(dt, fw, fh)

        # ── Draw ─────────────────────────────────────────────
        for ball in self._balls:
            ball.draw(frame, now)
        self._draw_hands(frame, active_hands)
        self._draw_hud(frame, fw)
        return frame

    def on_stop(self):
        self._started = False

    def tune(self, key: str):
        if key.lower() == 'a' and len(self._balls) < self.MAX_BALLS:
            self._balls.append(Ball(self._fw, self._fh))
        elif key.lower() == 'd' and len(self._balls) > 1:
            self._balls.pop()

    # ── Drawing ───────────────────────────────────────────────

    def _draw_hands(self, frame, hands):
        for h in hands:
            fist_col = (60, 60, 220)    # red-ish BGR for fist
            hit_col  = (60, 220, 60)    # green for active hit point
            idle_col = (80, 140, 80)    # dim green when stationary

            for kid, kp in h.kps.items():
                if h.closed:
                    col = fist_col
                else:
                    col = hit_col if kp.moving else idle_col
                # Small filled dot — no rings
                cv2.circle(frame, (int(kp.x), int(kp.y)), 5, col, -1, cv2.LINE_AA)

    def _draw_hud(self, frame, fw):
        banner_h = 40
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(fw,banner_h),(10,8,22),-1)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
        cv2.putText(frame, "BOUNCE", (fw//2-42, 28), FONT, 0.85, (200,200,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hits: {self._score}", (fw-110, 28), FONTS, 0.65, (100,255,180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"A/D  balls: {len(self._balls)}/{self.MAX_BALLS}",
                    (8, 28), FONTS, 0.52, (180,180,220), 1, cv2.LINE_AA)