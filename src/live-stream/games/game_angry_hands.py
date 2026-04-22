#!/usr/bin/env python3
"""
game_angry_hands.py — Angry Hands
------------------------------------
Angry Birds style launcher using your hand as the slingshot.

CLOSED FIST (or just bounding box, no keypoints visible):
  - Grabs the bird and pulls the slingshot back.
  - The further you pull from the slingshot anchor, the more power.

OPEN HAND (keypoints visible AND fingers spread):
  - Releases the bird. It flies in the opposite direction of the pull.

The slingshot anchor is in the left third of the screen.
Targets (pigs) sit on the right side on platforms.
Hit them to score. Miss and the bird falls off screen — next bird loads.

Detection approach for nano models
------------------------------------
  CLOSED  = bounding box exists but keypoint confidence is LOW
            (fist hides keypoints, or model just can't see them)
  OPEN    = bounding box exists AND multiple keypoints are high-conf
            AND fingers appear spread (tip keypoints far from wrist)

Works best with a 21-point hand model but degrades gracefully.
"""

from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from game_plugin import GamePlugin


# ── Tuning ────────────────────────────────────────────────────
SLING_X_FRAC    = 0.22        # slingshot anchor X (fraction of frame width)
SLING_Y_FRAC    = 0.60        # slingshot anchor Y
MAX_PULL        = 0.28        # max pull distance (fraction of frame width)
LAUNCH_SCALE    = 9.0         # velocity multiplier on launch
GRAVITY         = 900         # px / s²
BIRD_R          = 18
PIG_R           = 22
PLATFORM_H      = 14
TRAIL_LEN       = 22
COAST_SEC       = 0.5         # seconds to coast slingshot when hand lost
# Hand-open detection — all tunable via in-game debug panel
OPEN_KP_THRESH      = 0.25    # min keypoint confidence to use a point
OPEN_RATIO_THRESH   = 1.18    # tip_dist / knuckle_dist must exceed this to count as open
                               # 1.0 = tips exactly as far as knuckles, raise to require more spread
OPEN_MIN_TIPS       = 3       # how many fingers must pass the ratio test (out of 5)
NEXT_BIRD_DELAY = 1.4         # seconds after bird lands/falls before next
NUM_BIRDS       = 5           # birds per round
SCORE_PER_PIG   = 100

# Colours (BGR)
C_SKY       = (210, 170, 100)    # not used as fill — overlay only
C_SLING     = (40,  90,  160)
C_BAND      = (30,  60,  200)
C_BIRD      = (60,  80,  230)
C_BIRD_EYE  = (255, 255, 255)
C_PIG       = (40,  180,  40)
C_PIG_HIT   = (40,   60, 200)
C_PLATFORM  = (60,  110,  180)
C_GROUND    = (50,   90,   50)
C_TEXT      = (230, 225, 255)
C_SHADOW    = (0,     0,   0)
C_TRAIL     = (120, 160, 255)
C_AIM       = (80,  200, 255)
C_OPEN      = (80,  220, 100)
C_CLOSED    = (60,   60, 220)
C_STAR      = (40,  220, 255)

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────

def shadow_text(frame, text, pos, font, scale, col, thick=1):
    x, y = pos
    cv2.putText(frame, text, (x+2,y+2), font, scale, C_SHADOW, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,  y  ), font, scale, col,      thick,   cv2.LINE_AA)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


# ── Hand state detection ──────────────────────────────────────

# 21-point hand keypoint indices
# Wrist: 0
# Knuckles (MCP): 1,5,9,13,17
# Tips:           4,8,12,16,20
_WRIST      = 0
_TIP_IDS    = [4, 8, 12, 16, 20]
_KNUCKLE_IDS= [1, 5,  9, 13, 17]   # MCP = base knuckle, closest to palm


def _hand_open_ratio(inst, thresh=OPEN_KP_THRESH, ratio_thresh=OPEN_RATIO_THRESH,
                     min_tips=OPEN_MIN_TIPS):
    """
    Returns (is_open, ratio_value, debug_dict).

    Logic:
      For each finger, measure:
        tip_dist    = distance from wrist to fingertip
        knuckle_dist= distance from wrist to base knuckle (MCP)

      ratio = tip_dist / knuckle_dist
        > ratio_thresh → finger is EXTENDED   (tip further than knuckle)
        < ratio_thresh → finger is CURLED     (tip folded back)

      is_open = at least min_tips fingers are extended.

    This works because in a fist the tips curl back toward the palm
    so tip_dist < knuckle_dist (ratio < 1), whereas in an open hand
    tips are always further than knuckles (ratio > 1.15+).
    """
    if not inst or len(inst) < 21:
        return False, 0.0, {}

    wx, wy, wc = inst[_WRIST]
    if wc < thresh:
        return False, 0.0, {}

    ratios = {}
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
        r = tip_d / knuck_d
        ratios[tip_id] = r
        if r > ratio_thresh:
            extended += 1

    avg_ratio = sum(ratios.values()) / len(ratios) if ratios else 0.0
    is_open   = extended >= min_tips
    return is_open, avg_ratio, {"ratios": ratios, "extended": extended}


def detect_hand_state(keypoints, detections, fw,
                      kp_thresh=None, ratio_thresh=None, min_tips=None):
    """
    Returns (state, hand_x, hand_y, debug_info) where state is:
      'open'   — hand open, ready to release
      'closed' — fist / no keypoints, ready to pull
      'none'   — no hand detected at all

    Uses left half of screen only (slingshot side).
    Falls back to detection box centre when keypoints are absent.
    """
    if kp_thresh   is None: kp_thresh   = OPEN_KP_THRESH
    if ratio_thresh is None: ratio_thresh = OPEN_RATIO_THRESH
    if min_tips    is None: min_tips    = OPEN_MIN_TIPS

    mid = fw / 2
    best_inst = None
    best_conf = 0.0

    for inst in keypoints:
        if not inst or len(inst) < 1:
            continue
        x0, y0, c0 = inst[0]
        good = [(x,y,c) for x,y,c in inst if c > kp_thresh]
        if not good:
            continue
        cx = sum(p[0] for p in good) / len(good)
        cy = sum(p[1] for p in good) / len(good)
        if cx < mid and c0 > best_conf:
            best_conf = c0
            best_inst = (inst, cx, cy)

    if best_inst is not None:
        inst, cx, cy = best_inst
        is_open, avg_ratio, dbg = _hand_open_ratio(
            inst, kp_thresh, ratio_thresh, min_tips)
        state = 'open' if is_open else 'closed'
        dbg['cx'] = cx;  dbg['cy'] = cy;  dbg['avg_ratio'] = avg_ratio
        return state, cx, cy, dbg

    # No keypoints — check detection boxes
    for (label, conf, x1, y1, x2, y2) in detections:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if cx < mid and conf > 0.25:
            return 'closed', cx, cy, {}

    return 'none', 0.0, 0.0, {}


# ── Bird ──────────────────────────────────────────────────────

class Bird:
    def __init__(self, x: float, y: float):
        self.x        = x
        self.y        = y
        self.vx       = 0.0
        self.vy       = 0.0
        self.launched = False
        self.dead     = False
        self.trail    = []
        self.angle    = 0.0

    def launch(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.launched = True

    def update(self, dt, fw, fh):
        if not self.launched:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)
        self.vy += GRAVITY * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.angle += math.degrees(math.atan2(self.vy, self.vx)) * dt * 0.5
        if self.x > fw + 60 or self.y > fh + 60 or self.x < -60:
            self.dead = True

    def draw(self, frame):
        # Trail
        for i, (tx, ty) in enumerate(self.trail):
            a = i / max(len(self.trail), 1)
            r = max(2, int(BIRD_R * a * 0.6))
            ov = frame.copy()
            cv2.circle(ov, (int(tx), int(ty)), r, C_TRAIL, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a * 0.5, frame, 1 - a * 0.5, 0, frame)
        # Body
        bx, by = int(self.x), int(self.y)
        cv2.circle(frame, (bx, by), BIRD_R, C_BIRD, -1, cv2.LINE_AA)
        # Eye
        ex = bx + int(math.cos(math.radians(self.angle - 30)) * BIRD_R * 0.45)
        ey = by + int(math.sin(math.radians(self.angle - 30)) * BIRD_R * 0.45)
        cv2.circle(frame, (ex, ey), 5, C_BIRD_EYE, -1, cv2.LINE_AA)
        cv2.circle(frame, (ex+1, ey-1), 2, C_SHADOW, -1, cv2.LINE_AA)
        # Angry brow
        brow_x1 = bx - BIRD_R//2;  brow_y = by - BIRD_R//2
        cv2.line(frame, (brow_x1, brow_y+4), (brow_x1 + BIRD_R, brow_y),
                 C_SHADOW, 2, cv2.LINE_AA)


# ── Pig ──────────────────────────────────────────────────────

class Pig:
    def __init__(self, x: float, y: float):
        self.x      = x
        self.y      = y
        self.hp     = 2
        self.hit_ts = -99.0

    @property
    def dead(self):
        return self.hp <= 0

    def hit(self, now):
        self.hp -= 1
        self.hit_ts = now

    def draw(self, frame, now):
        col = C_PIG_HIT if now - self.hit_ts < 0.2 else C_PIG
        bx, by = int(self.x), int(self.y)
        cv2.circle(frame, (bx, by), PIG_R, col, -1, cv2.LINE_AA)
        # Snout
        cv2.ellipse(frame, (bx, by + PIG_R//3), (PIG_R//2, PIG_R//3),
                    0, 0, 360, (30, 140, 30), -1)
        # Nostrils
        cv2.circle(frame, (bx-5, by+PIG_R//3), 3, (20, 100, 20), -1)
        cv2.circle(frame, (bx+5, by+PIG_R//3), 3, (20, 100, 20), -1)
        # Eyes
        cv2.circle(frame, (bx-8, by-5), 5, (255,255,255), -1)
        cv2.circle(frame, (bx+8, by-5), 5, (255,255,255), -1)
        cv2.circle(frame, (bx-7, by-5), 3, C_SHADOW, -1)
        cv2.circle(frame, (bx+9, by-5), 3, C_SHADOW, -1)
        # HP dots
        for i in range(self.hp):
            cv2.circle(frame, (bx - 6 + i*12, by - PIG_R - 8), 4, C_PIG, -1)


# ── Platform ──────────────────────────────────────────────────

class Platform:
    def __init__(self, x: float, y: float, w: float):
        self.x = x
        self.y = y
        self.w = w

    def draw(self, frame):
        x1 = int(self.x - self.w/2)
        x2 = int(self.x + self.w/2)
        y1 = int(self.y)
        y2 = int(self.y + PLATFORM_H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_PLATFORM, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 140, 220), 1)


# ── Slingshot ────────────────────────────────────────────────

class Slingshot:
    def __init__(self, ax, ay):
        self.ax = ax   # anchor (fork centre)
        self.ay = ay
        # Two fork tines
        self.fork_l = (ax - 18, ay - 28)
        self.fork_r = (ax + 18, ay - 28)
        self.base   = (ax,      ay + 55)

    def draw(self, frame, pull_x=None, pull_y=None, has_bird=True):
        # Stick
        cv2.line(frame, (int(self.ax), int(self.ay + 55)),
                 (int(self.ax), int(self.ay)), C_SLING, 10, cv2.LINE_AA)
        # Forks
        cv2.line(frame, (int(self.ax), int(self.ay)),
                 (int(self.fork_l[0]), int(self.fork_l[1])), C_SLING, 8, cv2.LINE_AA)
        cv2.line(frame, (int(self.ax), int(self.ay)),
                 (int(self.fork_r[0]), int(self.fork_r[1])), C_SLING, 8, cv2.LINE_AA)

        if pull_x is not None and pull_y is not None:
            px, py = int(pull_x), int(pull_y)
            # Elastic bands
            cv2.line(frame, (int(self.fork_l[0]), int(self.fork_l[1])),
                     (px, py), C_BAND, 3, cv2.LINE_AA)
            cv2.line(frame, (int(self.fork_r[0]), int(self.fork_r[1])),
                     (px, py), C_BAND, 3, cv2.LINE_AA)


# ── Score pop ────────────────────────────────────────────────

class ScorePop:
    def __init__(self, x: float, y: float, txt: str, ts: float, dur: float = 1.1):
        self.x   = x
        self.y   = y
        self.txt = txt
        self.ts  = ts
        self.dur = dur

    def alive(self, now):
        return now - self.ts < self.dur

    def draw(self, frame, now):
        a = clamp(1.0 - (now - self.ts) / self.dur, 0, 1)
        dy = int((1 - a) * 40)
        (tw, _), _ = cv2.getTextSize(self.txt, FONT, 0.9, 2)
        col = tuple(int(c * a) for c in C_STAR)
        shadow_text(frame, self.txt,
                    (int(self.x) - tw//2, int(self.y) - dy),
                    FONT, 0.9, col, 2)


# ── Game ─────────────────────────────────────────────────────

class Game(GamePlugin):

    def __init__(self):
        self._fw = 640
        self._fh = 480
        self._t_last = time.perf_counter()
        # Initialise everything to safe defaults before _reset touches them
        self._sling      = None
        self._bird       = None
        self._pigs       = []
        self._platforms  = []
        self._pops       = []
        self._score      = 0
        self._birds_left = NUM_BIRDS
        self._phase      = "waiting"
        self._phase_ts   = self._t_last
        self._pull_x     = float(self._fw * SLING_X_FRAC)
        self._pull_y     = float(self._fh * SLING_Y_FRAC)
        self._sx         = self._pull_x
        self._sy         = self._pull_y
        self._svx        = 0.0
        self._svy        = 0.0
        self._hand_lost  = 0.0
        self._cur_state  = 'none'

        # ── Live tuning (edited via debug panel) ──
        self._kp_thresh    = OPEN_KP_THRESH
        self._ratio_thresh = OPEN_RATIO_THRESH
        self._min_tips     = OPEN_MIN_TIPS
        self._show_debug   = True   # toggle with D key
        self._last_dbg     = {}

        self._reset()

    # ── API ───────────────────────────────────────────────────

    def on_start(self, fw, fh, class_names=None):
        self._fw, self._fh = fw, fh
        self._reset()

    def on_frame(self, frame, keypoints, detections, fw, fh):
        now = time.perf_counter()
        dt  = clamp(now - self._t_last, 0.0, 0.07)
        self._t_last = now
        self._fw, self._fh = fw, fh

        state, hx, hy, dbg = detect_hand_state(
            keypoints, detections, fw,
            self._kp_thresh, self._ratio_thresh, self._min_tips)
        self._last_dbg = dbg
        self._update_hand_coast(state, hx, hy, dt, now)

        if self._phase == "waiting":
            self._do_waiting(frame, now)
        elif self._phase == "aiming":
            self._do_aiming(frame, dt, now)
        elif self._phase == "flying":
            self._do_flying(frame, dt, now)
        elif self._phase == "next_bird":
            self._do_next_bird(frame, now)
        elif self._phase == "win":
            self._do_win(frame, now)
        elif self._phase == "lose":
            self._do_lose(frame, now)

        self._draw_hud(frame, now)
        if self._show_debug:
            self._draw_debug(frame)
        return frame

    # ── Phases ────────────────────────────────────────────────

    def _do_waiting(self, frame, now):
        fw, fh = self._fw, self._fh
        if self._sling:
            self._sling.draw(frame)
        self._draw_scene(frame, now)

        # Tint
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (fw, fh), (10,8,20), -1)
        cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)

        shadow_text(frame, "ANGRY  HANDS",
                    (fw//2 - 120, fh//2 - 55), FONT, 1.6, C_TEXT, 3)
        shadow_text(frame, "Show a CLOSED FIST on the LEFT to grab",
                    (fw//2 - 200, fh//2 + 5), FONT, 0.55, (170,160,210))
        shadow_text(frame, "OPEN hand to release",
                    (fw//2 - 105, fh//2 + 35), FONT, 0.55, (170,160,210))

        # Hand state indicator
        s = self._cur_state
        col  = C_OPEN if s == 'open' else C_CLOSED if s == 'closed' else (100,100,130)
        lbl  = f"Hand: {s.upper()}"
        shadow_text(frame, lbl, (fw//2 - 55, fh - 28), FONTS, 0.65, col)

        if self._cur_state == 'closed':
            self._phase = "aiming"

    def _do_aiming(self, frame, dt, now):
        fw, fh = self._fw, self._fh
        ax, ay = self._sling.ax, self._sling.ay

        # Current pull position from hand
        if self._cur_state != 'none':
            raw_pull_x = self._sx
            raw_pull_y = self._sy
        else:
            raw_pull_x = ax
            raw_pull_y = ay

        # Clamp pull distance to max pull radius
        max_px = fw * MAX_PULL
        dx = raw_pull_x - ax
        dy = raw_pull_y - ay
        dist = math.hypot(dx, dy)
        if dist > max_px:
            scale = max_px / dist
            dx *= scale;  dy *= scale
            dist = max_px
        self._pull_x = ax + dx
        self._pull_y = ay + dy

        # Bird sits at pull position
        self._bird.x = self._pull_x
        self._bird.y = self._pull_y

        # ── Exponential power curve ──────────────────────────────
        # frac = 0..1 (0 = no pull, 1 = fully pulled to max_px)
        # power = frac^0.4  — exponential: small pulls still give decent
        # velocity, full pull = 100% power. The exponent < 1 means the
        # curve bows upward: you get most of the power before max pull,
        # so even a modest pull feels responsive.
        #
        #   frac  power
        #   0.10  0.40      (10% pull → 40% power)
        #   0.25  0.63      (25% pull → 63% power)
        #   0.50  0.76      (50% pull → 76% power)
        #   0.75  0.87      (75% pull → 87% power)
        #   1.00  1.00      (full pull → 100% power)
        #
        frac  = dist / max_px if max_px > 0 else 0.0
        power = frac ** 0.4   # exponent < 1 = curves upward (more power early)

        # Direction unit vector, scaled by power and max launch speed
        max_launch = fw * LAUNCH_SCALE
        if dist > 0:
            nx = -dx / dist
            ny = -dy / dist
        else:
            nx, ny = 0.0, 0.0
        launch_vx = nx * power * max_launch
        launch_vy = ny * power * max_launch
        self._draw_aim_arc(frame, self._pull_x, self._pull_y, launch_vx, launch_vy)

        self._sling.draw(frame, self._pull_x, self._pull_y)
        self._draw_scene(frame, now)
        self._bird.draw(frame)

        # State indicator
        col = C_CLOSED if self._cur_state in ('closed','none') else C_OPEN
        lbl = "PULL BACK  (fist)" if self._cur_state in ('closed','none') else "OPEN to release!"
        shadow_text(frame, lbl, (fw//2-90, fh-28), FONTS, 0.62, col)

        # Release on open hand
        if self._cur_state == 'open':
            self._bird.launch(launch_vx, launch_vy)
            self._phase = "flying"
            self._phase_ts = now

    def _do_flying(self, frame, dt, now):
        self._bird.update(dt, self._fw, self._fh)
        self._sling.draw(frame)
        self._draw_scene(frame, now)
        self._bird.draw(frame)

        # Collision with pigs
        for pig in self._pigs:
            if pig.dead:
                continue
            d = math.hypot(self._bird.x - pig.x, self._bird.y - pig.y)
            if d < BIRD_R + PIG_R:
                pig.hit(now)
                self._score += SCORE_PER_PIG
                self._pops.append(ScorePop(pig.x, pig.y - PIG_R, f"+{SCORE_PER_PIG}", now))
                # Deflect bird
                self._bird.vx *= -0.4
                self._bird.vy *= -0.6

        # Remove dead pigs
        self._pigs = [p for p in self._pigs if not p.dead]

        if self._bird.dead:
            self._birds_left -= 1
            if not self._pigs:
                self._phase = "win";  self._phase_ts = now
            elif self._birds_left <= 0:
                self._phase = "lose"; self._phase_ts = now
            else:
                self._phase = "next_bird"; self._phase_ts = now

        if not self._pigs:
            self._phase = "win";  self._phase_ts = now

    def _do_next_bird(self, frame, now):
        self._sling.draw(frame)
        self._draw_scene(frame, now)
        shadow_text(frame, f"Next bird in {max(0, NEXT_BIRD_DELAY - (now-self._phase_ts)):.1f}s",
                    (self._fw//2-105, self._fh//2), FONT, 0.7, C_TEXT)
        if now - self._phase_ts > NEXT_BIRD_DELAY:
            self._spawn_bird()
            self._phase = "waiting"

    def _do_win(self, frame, now):
        fw, fh = self._fw, self._fh
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(fw,fh),(10,40,10),-1)
        cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)
        shadow_text(frame, "ALL PIGS DOWN!", (fw//2-145, fh//2-30), FONT, 1.5, C_STAR, 3)
        shadow_text(frame, f"Score: {self._score}", (fw//2-70, fh//2+25), FONT, 1.0, C_TEXT, 2)
        shadow_text(frame, "Fist to play again", (fw//2-100, fh//2+70), FONT, 0.65, (160,200,160))
        if self._cur_state == 'closed' and now - self._phase_ts > 2.0:
            self._reset()

    def _do_lose(self, frame, now):
        fw, fh = self._fw, self._fh
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(fw,fh),(30,10,10),-1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)
        shadow_text(frame, "OUT OF BIRDS!", (fw//2-130, fh//2-30), FONT, 1.4, C_CLOSED, 3)
        shadow_text(frame, f"Score: {self._score}", (fw//2-70, fh//2+25), FONT, 1.0, C_TEXT, 2)
        shadow_text(frame, "Fist to try again", (fw//2-98, fh//2+68), FONT, 0.65, (200,150,150))
        if self._cur_state == 'closed' and now - self._phase_ts > 2.0:
            self._reset()

    # ── Drawing ───────────────────────────────────────────────

    def _draw_debug(self, frame):
        """Always-visible power + hand tuning panel — top left."""
        fw, fh = self._fw, self._fh
        dbg    = self._last_dbg
        px, py = 8, 52
        lh     = 19

        ax     = self._sling.ax if self._sling else int(fw * SLING_X_FRAC)
        ay     = self._sling.ay if self._sling else int(fh * SLING_Y_FRAC)
        max_px = fw * MAX_PULL
        dist   = math.hypot(self._sx - ax, self._sy - ay)
        frac   = min(dist / max_px, 1.0) if max_px > 0 else 0.0
        power  = frac ** 0.4
        pct    = int(power * 100)

        lines = [
            (f"PWR  {pct:3d}%",               (60 + int(195*power), 60 + int(195*(1-power)), 80)),
            (f"+/-  open ratio : {self._ratio_thresh:.2f}", (180, 180, 220)),
            (f"Q/W  tips needed: {self._min_tips}",         (180, 180, 220)),
            (f"A/S  kp conf    : {self._kp_thresh:.2f}",    (180, 180, 220)),
            (f"D    hide panel",                             (100, 100, 130)),
        ]

        panel_w = 192
        panel_h = len(lines) * lh + 14
        ov = frame.copy()
        cv2.rectangle(ov, (px-4, py-20), (px+panel_w, py+panel_h), (8,6,18), -1)
        cv2.addWeighted(ov, 0.68, frame, 0.32, 0, frame)

        bar_y = py - 10
        cv2.rectangle(frame, (px, bar_y), (px+panel_w-4, bar_y+6), (30,28,45), -1)
        bar_filled = int((panel_w-4) * power)
        if bar_filled > 0:
            bar_col = (60+int(195*power), 60+int(195*(1-power)), 80)
            cv2.rectangle(frame, (px, bar_y), (px+bar_filled, bar_y+6), bar_col, -1)

        for i, (txt, col) in enumerate(lines):
            y = py + i * lh
            cv2.putText(frame, txt, (px+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (px,   y  ), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        col, 1, cv2.LINE_AA)

        state_col = (80,220,80) if self._cur_state=='open' else \
                    (60,60,220) if self._cur_state=='closed' else (60,60,80)
        cv2.circle(frame, (px + panel_w - 10, py - 13), 6, state_col, -1, cv2.LINE_AA)

    def _draw_scene(self, frame, now):
        fw, fh = self._fw, self._fh

        # Ground line
        ground_y = int(fh * 0.88)
        cv2.line(frame, (0, ground_y), (fw, ground_y), C_GROUND, 3)

        # Platforms + pigs
        for plat in self._platforms:
            plat.draw(frame)
        for pig in self._pigs:
            pig.draw(frame, now)

        # Score pops
        self._pops = [p for p in self._pops if p.alive(now)]
        for p in self._pops:
            p.draw(frame, now)

        # Birds remaining (bottom left icons)
        for i in range(self._birds_left - (1 if self._phase in ("aiming","flying") else 0)):
            bx = 30 + i * (BIRD_R*2 + 6)
            by = ground_y + 28
            cv2.circle(frame, (bx, by), BIRD_R - 4, C_BIRD, -1, cv2.LINE_AA)

    def _draw_aim_arc(self, frame, sx, sy, vx, vy):
        pts = []
        x, y = sx, sy
        tvx, tvy = vx, vy
        step = 0.045
        for _ in range(18):
            pts.append((int(x), int(y)))
            tvy += GRAVITY * step
            x   += tvx * step
            y   += tvy * step
            if x > self._fw or y > self._fh:
                break
        for i in range(len(pts) - 1):
            a = 1.0 - i / max(len(pts)-1, 1)
            r = max(2, int(6 * a))
            col = tuple(int(c * a) for c in C_AIM)
            cv2.circle(frame, pts[i], r, col, -1, cv2.LINE_AA)

    def _draw_hud(self, frame, now):
        fw, fh = self._fw, self._fh
        banner_h = 42
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(fw, banner_h),(10,8,22),-1)
        cv2.addWeighted(ov, 0.62, frame, 0.38, 0, frame)

        shadow_text(frame, "ANGRY  HANDS", (fw//2-90, 30), FONT, 0.85, C_TEXT)
        shadow_text(frame, f"Score: {self._score}", (fw-130, 30), FONTS, 0.65, C_STAR)

        # Hand state badge top-left
        s   = self._cur_state
        col = C_OPEN if s == 'open' else C_CLOSED if s == 'closed' else (80,80,110)
        shadow_text(frame, s.upper(), (10, 30), FONTS, 0.65, col)

        # Pigs left
        shadow_text(frame, f"Pigs: {len(self._pigs)}", (fw//2-30, banner_h+20),
                    FONTS, 0.5, C_PIG)

    # ── Hand coasting ─────────────────────────────────────────

    def _update_hand_coast(self, state, hx, hy, dt, now):
        if state != 'none':
            # Smooth position
            alpha = clamp(0.30 + dt*5, 0, 1)
            self._sx = lerp(self._sx, hx, alpha)
            self._sy = lerp(self._sy, hy, alpha)
            self._svx = (hx - self._sx) / max(dt, 1e-4) * 0.3
            self._svy = (hy - self._sy) / max(dt, 1e-4) * 0.3
            self._hand_lost = 0.0
            self._cur_state = state
        else:
            self._hand_lost += dt
            if self._hand_lost < COAST_SEC:
                # Coast on last velocity
                self._sx += self._svx * dt
                self._sy += self._svy * dt
                self._svx *= (1 - dt * 4)
                self._svy *= (1 - dt * 4)
                # Keep last known state during coast
            else:
                self._cur_state = 'none'

    # ── Level generation ──────────────────────────────────────

    def _gen_level(self):
        fw, fh = self._fw, self._fh
        ground_y = fh * 0.88
        self._platforms = []
        self._pigs = []

        configs = [
            # (x_frac, y_frac_above_ground, plat_w)
            (0.65, 0.22, 80),
            (0.80, 0.10, 70),
            (0.72, 0.38, 90),
            (0.88, 0.30, 70),
            (0.60, 0.45, 100),
        ]
        random.shuffle(configs)
        n_pigs = random.randint(3, 5)

        for i in range(n_pigs):
            xf, yf, pw = configs[i]
            px = fw * xf + random.uniform(-20, 20)
            py = ground_y - fh * yf - PLATFORM_H
            plat = Platform(px, py + PLATFORM_H//2 + PIG_R*2, pw)
            pig  = Pig(px, py)
            self._platforms.append(plat)
            self._pigs.append(pig)

    def _spawn_bird(self):
        ax, ay = self._sling.ax, self._sling.ay
        self._bird = Bird(ax, ay - BIRD_R - 5)

    # ── Reset ─────────────────────────────────────────────────

    def toggle_debug(self):
        self._show_debug = not self._show_debug

    def tune(self, key: str):
        """Called by streamer keyboard hook."""
        if   key == '+': self._ratio_thresh = round(min(2.5, self._ratio_thresh + 0.05), 2)
        elif key == '-': self._ratio_thresh = round(max(0.5, self._ratio_thresh - 0.05), 2)
        elif key == 'q': self._min_tips     = min(5, self._min_tips + 1)
        elif key == 'w': self._min_tips     = max(1, self._min_tips - 1)
        elif key == 'a': self._kp_thresh    = round(min(0.9, self._kp_thresh + 0.05), 2)
        elif key == 's': self._kp_thresh    = round(max(0.05, self._kp_thresh - 0.05), 2)

    def _reset(self):
        fw, fh = self._fw, self._fh
        ax = int(fw * SLING_X_FRAC)
        ay = int(fh * SLING_Y_FRAC)
        self._sling      = Slingshot(ax, ay)
        self._score      = 0
        self._birds_left = NUM_BIRDS
        self._phase      = "waiting"
        self._phase_ts   = time.perf_counter()
        self._pull_x     = float(ax)
        self._pull_y     = float(ay)
        self._pops: list[ScorePop] = []

        # Hand coast state
        self._sx = float(ax)
        self._sy = float(ay)
        self._svx = 0.0
        self._svy = 0.0
        self._hand_lost  = 0.0
        self._cur_state  = 'none'

        self._gen_level()
        self._spawn_bird()
        self._t_last = time.perf_counter()
