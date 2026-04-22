#!/usr/bin/env python3
"""
game_swordfish.py — SWORD FIGHT
---------------------------------
Two players. Each holds up a hand on their side of the screen.
Your hand position controls the tip of your sword.

Swords are drawn as glowing blades from a fixed hilt at the bottom
corner of your side, extending toward your hand.

CLASH: when the two blade line segments intersect, both players take
damage. The player whose blade is moving FASTER at the moment of clash
deals more damage and takes less.

LUNGE: if your hand crosses the centre line into enemy territory, your
sword extends further and deals bonus damage on clash — but you're
exposed (take 1.5x damage while lunging).

PARRY: if your blade angle is nearly perpendicular to the incoming
blade at the moment of clash, damage is reduced 70%.

First to 0 HP loses. Simple. Violent. Fun.

Works with pose keypoints (wrist) or detection boxes.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque

import cv2
import numpy as np

from game_plugin import GamePlugin


# ── Tuning ────────────────────────────────────────────────────
MAX_HP           = 100
CLASH_DAMAGE     = 22       # base damage per clash
LUNGE_BONUS      = 1.6      # damage multiplier when crossing centre
LUNGE_PENALTY    = 1.5      # damage taken multiplier when lunging
PARRY_THRESHOLD  = 65       # degrees between blades = parry
PARRY_REDUCTION  = 0.30     # multiplier on damage when parrying
SPEED_RATIO_PWR  = 0.6      # how much speed advantage matters
CLASH_COOLDOWN   = 0.30     # seconds between clashes
READY_HOLD       = 1.0      # seconds both hands visible to start
BLADE_LEN_BASE   = 0.38     # fraction of frame width
BLADE_LEN_LUNGE  = 0.55
SWORD_THICK      = 4
BLOOD_PARTICLES  = 18

# Colours (BGR)
C_P1_BLADE  = (180, 220, 255)   # silver-blue
C_P2_BLADE  = (100, 255, 180)   # silver-green
C_P1_HILT   = (60,  100, 255)   # orange-red in BGR
C_P2_HILT   = (60,  200,  60)
C_CLASH     = (80,  180, 255)
C_BLOOD     = (30,   30, 200)
C_HP_OK     = (60,  200,  80)
C_HP_LOW    = (40,   80, 220)
C_HP_CRIT  = (30,   30, 200)
C_WIN       = (100, 255, 160)
C_TEXT      = (230, 225, 255)
C_SHADOW    = (0,     0,   0)
C_LUNGE     = (40,   40, 255)

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────

def shadow_text(frame, text, pos, font, scale, col, thick=1):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), font, scale, C_SHADOW, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y  ), font, scale, col,      thick,   cv2.LINE_AA)


def seg_intersect(p1, p2, p3, p4):
    """Return intersection point of segments p1-p2 and p3-p4, or None."""
    x1,y1 = p1;  x2,y2 = p2;  x3,y3 = p3;  x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t*(x2-x1)
        iy = y1 + t*(y2-y1)
        return (ix, iy)
    return None


def blade_angle_deg(hilt, tip):
    dx, dy = tip[0]-hilt[0], tip[1]-hilt[1]
    return math.degrees(math.atan2(dy, dx))


def angle_between(a1, a2):
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Particle ──────────────────────────────────────────────────

class BloodParticle:
    def __init__(self, x, y):
        ang = random.uniform(-math.pi, math.pi)
        spd = random.uniform(80, 300)
        self.x, self.y = float(x), float(y)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd - random.uniform(50, 150)
        self.life = random.uniform(0.3, 0.9)
        self.max_life = self.life
        self.r = random.randint(3, 8)

    def update(self, dt):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += 400 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        a = clamp(self.life / self.max_life, 0, 1)
        r = max(1, int(self.r * a))
        ov = frame.copy()
        cv2.circle(ov, (int(self.x), int(self.y)), r, C_BLOOD, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, a * 0.9, frame, 1 - a * 0.9, 0, frame)


# ── Player ────────────────────────────────────────────────────

class Player:
    def __init__(self, side: str, fw: int, fh: int):
        self.side   = side
        self.fw, self.fh = fw, fh
        self.hp     = float(MAX_HP)
        self.colour = C_P1_BLADE if side == "left" else C_P2_BLADE
        self.hilt_colour = C_P1_HILT if side == "left" else C_P2_HILT

        # Hilt fixed at bottom corner of own side
        if side == "left":
            self.hilt = (int(fw * 0.08), int(fh * 0.88))
        else:
            self.hilt = (int(fw * 0.92), int(fh * 0.88))

        # Smoothed hand position (tip target)
        self.hand_x = float(fw * 0.25 if side == "left" else fw * 0.75)
        self.hand_y = float(fh * 0.35)

        # Velocity — used to coast when detection drops out
        self.vel_x = 0.0
        self.vel_y = 0.0

        # How long since we last had a real detection (seconds)
        self.lost_for = 0.0
        # Max seconds to coast on momentum before freezing
        self.coast_max = 0.55

        # Speed tracking for clash resolution
        self.speed = 0.0   # px/s

        # Damage flash
        self.hit_ts   = -99.0

        # Seen-since for ready tracking
        self.seen_since: float | None = None

    @property
    def lunging(self) -> bool:
        mid = self.fw / 2
        if self.side == "left":
            return self.hand_x > mid
        else:
            return self.hand_x < mid

    @property
    def tip(self) -> tuple[float, float]:
        hx, hy = float(self.hilt[0]), float(self.hilt[1])
        tx, ty = self.hand_x, self.hand_y
        dx, dy = tx - hx, ty - hy
        dist   = math.hypot(dx, dy)
        if dist < 1:
            return (tx, ty)
        base_len = self.fw * (BLADE_LEN_LUNGE if self.lunging else BLADE_LEN_BASE)
        scale = base_len / dist
        return (hx + dx * scale, hy + dy * scale)

    def update_hand(self, hx: float, hy: float, dt: float):
        """Called when we actually have a detection."""
        alpha = clamp(0.28 + dt * 4, 0.0, 1.0)
        prev_x, prev_y = self.hand_x, self.hand_y
        self.hand_x = lerp(self.hand_x, hx, alpha)
        self.hand_y = lerp(self.hand_y, hy, alpha)
        dx = self.hand_x - prev_x
        dy = self.hand_y - prev_y
        # Track instantaneous velocity for coasting
        self.vel_x = lerp(self.vel_x, dx / max(dt, 1e-4), 0.4)
        self.vel_y = lerp(self.vel_y, dy / max(dt, 1e-4), 0.4)
        dist = math.hypot(dx, dy)
        self.speed = lerp(self.speed, dist / max(dt, 1e-4), 0.35)
        self.lost_for = 0.0

    def coast(self, dt: float):
        """Called when detection is missing — keep sword moving on momentum."""
        self.lost_for += dt
        if self.lost_for > self.coast_max:
            # Freeze — damp velocity to zero
            self.vel_x *= 0.85
            self.vel_y *= 0.85
        else:
            # Decay velocity and apply
            decay = max(0.0, 1.0 - self.lost_for / self.coast_max)
            self.vel_x *= (1.0 - dt * 3.5)
            self.vel_y *= (1.0 - dt * 3.5)
            self.hand_x += self.vel_x * dt * decay
            self.hand_y += self.vel_y * dt * decay
        # Keep within frame
        self.hand_x = clamp(self.hand_x, 0, self.fw)
        self.hand_y = clamp(self.hand_y, 0, self.fh)
        self.speed  = lerp(self.speed, 0.0, dt * 4)

    def take_damage(self, dmg: float, now: float):
        self.hp = max(0.0, self.hp - dmg)
        self.hit_ts = now

    def draw(self, frame: np.ndarray, now: float):
        hilt = self.hilt
        tip  = (int(self.tip[0]), int(self.tip[1]))

        # Lunge warning
        if self.lunging:
            ov = frame.copy()
            col = C_P1_HILT if self.side == "left" else C_P2_HILT
            cv2.line(ov, (self.fw//2, 0), (self.fw//2, self.fh), C_LUNGE, 3)
            cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)

        # Hit flash
        flash = clamp((now - self.hit_ts), 0, 0.25) / 0.25
        if flash < 1.0:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (self.fw, self.fh), C_BLOOD, -1)
            cv2.addWeighted(ov, (1 - flash) * 0.3, frame, 1 - (1 - flash) * 0.3, 0, frame)

        # Blade glow (wide, faint)
        cv2.line(frame, hilt, tip, self.colour, SWORD_THICK + 6, cv2.LINE_AA)
        ov = frame.copy()
        cv2.line(ov, hilt, tip, self.colour, SWORD_THICK + 12, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)

        # Blade core (bright)
        cv2.line(frame, hilt, tip, (240, 240, 255), 2, cv2.LINE_AA)

        # Hilt cross-guard
        ang   = math.atan2(tip[1]-hilt[1], tip[0]-hilt[0])
        perp  = ang + math.pi / 2
        guard = 14
        gx1   = int(hilt[0] + math.cos(perp) * guard)
        gy1   = int(hilt[1] + math.sin(perp) * guard)
        gx2   = int(hilt[0] - math.cos(perp) * guard)
        gy2   = int(hilt[1] - math.sin(perp) * guard)
        cv2.line(frame, (gx1, gy1), (gx2, gy2), self.hilt_colour, 5, cv2.LINE_AA)
        cv2.circle(frame, hilt, 8, self.hilt_colour, -1, cv2.LINE_AA)


# ── Game ─────────────────────────────────────────────────────

class Game(GamePlugin):

    def __init__(self):
        self._fw, self._fh = 640, 480
        self._state  = "waiting"   # waiting | countdown | playing | win
        self._state_ts = 0.0
        self._t_last = time.perf_counter()

        self._p1: Player | None = None
        self._p2: Player | None = None

        self._particles: list[BloodParticle] = []
        self._clash_cooldown = 0.0
        self._clash_flash_ts = -99.0
        self._clash_pt: tuple | None = None

        self._winner = 0
        self._last_parry = ""   # "P1 PARRY!" etc for display

        # Clash log for drama text
        self._drama_txt  = ""
        self._drama_ts   = -99.0

    # ── API ───────────────────────────────────────────────────

    def on_start(self, fw, fh, class_names=None):
        self._fw, self._fh = fw, fh
        self._reset()

    def on_frame(self, frame, keypoints, detections, fw, fh):
        now = time.perf_counter()
        dt  = clamp(now - self._t_last, 0.0, 0.07)
        self._t_last = now
        self._fw, self._fh = fw, fh

        p1_pos, p2_pos = self._find_hands(keypoints, detections)

        if self._state == "waiting":
            self._do_waiting(frame, p1_pos, p2_pos, now)

        elif self._state == "countdown":
            self._do_countdown(frame, p1_pos, p2_pos, now)

        elif self._state == "playing":
            self._do_playing(frame, p1_pos, p2_pos, dt, now)

        elif self._state == "win":
            self._do_win(frame, now)

        # Particles
        self._particles = [p for p in self._particles if p.update(dt)]
        for p in self._particles:
            p.draw(frame)

        # Clash flash point
        if self._clash_pt and now - self._clash_flash_ts < 0.18:
            a = 1.0 - (now - self._clash_flash_ts) / 0.18
            ov = frame.copy()
            cv2.circle(ov, (int(self._clash_pt[0]), int(self._clash_pt[1])),
                       int(30 * a), C_CLASH, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a * 0.7, frame, 1 - a * 0.7, 0, frame)

        # Drama text
        if now - self._drama_ts < 1.2:
            a = clamp(1.0 - (now - self._drama_ts) / 1.2, 0, 1)
            scale = 1.0 + (1 - a) * 0.5
            (tw, th), _ = cv2.getTextSize(self._drama_txt, FONT, scale, 2)
            shadow_text(frame, self._drama_txt,
                        (fw//2 - tw//2, fh//2 + th//2),
                        FONT, scale,
                        tuple(int(c * a) for c in C_CLASH), 2)

        # HUD always
        if self._state in ("playing", "countdown"):
            self._draw_hud(frame)

        return frame

    # ── State handlers ────────────────────────────────────────

    def _do_waiting(self, frame, p1_pos, p2_pos, now):
        fw, fh = self._fw, self._fh

        if p1_pos:
            if self._p1.seen_since is None:
                self._p1.seen_since = now
        else:
            self._p1.seen_since = None

        if p2_pos:
            if self._p2.seen_since is None:
                self._p2.seen_since = now
        else:
            self._p2.seen_since = None

        p1h = (now - self._p1.seen_since) if self._p1.seen_since else 0.0
        p2h = (now - self._p2.seen_since) if self._p2.seen_since else 0.0

        if p1h >= READY_HOLD and p2h >= READY_HOLD:
            self._state    = "countdown"
            self._state_ts = now
            return

        # Draw swords in resting position
        self._p1.draw(frame, now)
        self._p2.draw(frame, now)

        # Overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, fh), (10, 5, 20), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        shadow_text(frame, "SWORD  FIGHT",
                    (fw//2 - 130, fh//2 - 55), FONT, 1.8, C_TEXT, 3)
        shadow_text(frame, "Show your hand on your side to ready up",
                    (fw//2 - 195, fh//2 + 5), FONT, 0.55, (160, 155, 200))

        def bar(cx, label, held, col):
            bw = 110;  bh = 14
            bx = cx - bw//2;  by = fh//2 + 30
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (35, 30, 55), -1)
            if held > 0:
                fw2 = int(bw * clamp(held/READY_HOLD, 0, 1))
                cv2.rectangle(frame, (bx, by), (bx+fw2, by+bh),
                              col if held < READY_HOLD else (60, 220, 80), -1)
            shadow_text(frame, label, (cx-28, by-8), FONT, 0.52,
                        (60, 220, 80) if held >= READY_HOLD else (140, 130, 180))

        bar(fw//4,     "P1  LEFT",  p1h, C_P1_BLADE)
        bar(3*fw//4,   "P2  RIGHT", p2h, C_P2_BLADE)

    def _do_countdown(self, frame, p1_pos, p2_pos, now):
        elapsed = now - self._state_ts
        n = 3 - int(elapsed)

        if p1_pos:
            self._p1.update_hand(*p1_pos, 0.016)
        else:
            self._p1.coast(0.016)
        if p2_pos:
            self._p2.update_hand(*p2_pos, 0.016)
        else:
            self._p2.coast(0.016)
        self._p1.draw(frame, now)
        self._p2.draw(frame, now)
        self._draw_hud(frame)

        if elapsed >= 3.0:
            self._state = "playing"
            return

        num = str(n + 1) if n >= 0 else "FIGHT!"
        scale = 3.0 if num != "FIGHT!" else 2.2
        (tw, th), _ = cv2.getTextSize(num, FONT, scale, 4)
        shadow_text(frame, num,
                    (self._fw//2 - tw//2, self._fh//2 + th//2),
                    FONT, scale, (200, 200, 255), 4)

    def _do_playing(self, frame, p1_pos, p2_pos, dt, now):
        # Move swords — coast on momentum when detection drops
        if p1_pos:
            self._p1.update_hand(*p1_pos, dt)
        else:
            self._p1.coast(dt)
        if p2_pos:
            self._p2.update_hand(*p2_pos, dt)
        else:
            self._p2.coast(dt)

        self._clash_cooldown = max(0.0, self._clash_cooldown - dt)

        # Clash detection — segment intersect + proximity fallback
        # (proximity catches fast-moving blades that skip through each other)
        if self._clash_cooldown <= 0:
            pt = seg_intersect(
                self._p1.hilt, self._p1.tip,
                self._p2.hilt, self._p2.tip,
            )
            if pt is None:
                # Proximity check: if tips are very close, count as clash
                tip1 = self._p1.tip
                tip2 = self._p2.tip
                if math.hypot(tip1[0]-tip2[0], tip1[1]-tip2[1]) < 38:
                    pt = ((tip1[0]+tip2[0])/2, (tip1[1]+tip2[1])/2)
            if pt:
                self._resolve_clash(pt, now)

        # Draw
        self._p1.draw(frame, now)
        self._p2.draw(frame, now)

        # Check win
        if self._p1.hp <= 0 or self._p2.hp <= 0:
            self._winner = 2 if self._p1.hp <= 0 else 1
            self._state  = "win"
            self._state_ts = now

    def _resolve_clash(self, pt, now):
        self._clash_cooldown = CLASH_COOLDOWN
        self._clash_flash_ts = now
        self._clash_pt = pt

        # Spawn blood
        for _ in range(BLOOD_PARTICLES):
            self._particles.append(BloodParticle(pt[0], pt[1]))

        # Speed advantage
        spd1 = max(1.0, self._p1.speed)
        spd2 = max(1.0, self._p2.speed)
        ratio1 = (spd1 / (spd1 + spd2)) ** SPEED_RATIO_PWR   # p1's share of damage dealt
        ratio2 = 1.0 - ratio1

        # Parry check
        ang1 = blade_angle_deg(self._p1.hilt, self._p1.tip)
        ang2 = blade_angle_deg(self._p2.hilt, self._p2.tip)
        between = angle_between(ang1, ang2)
        p1_parry = between >= PARRY_THRESHOLD
        p2_parry = between >= PARRY_THRESHOLD   # same clash, both could parry

        # Damage to P1 (from P2's blade)
        dmg_to_p1 = CLASH_DAMAGE * (ratio2 * 2)
        if p1_parry:
            dmg_to_p1 *= PARRY_REDUCTION
        if self._p1.lunging:
            dmg_to_p1 *= LUNGE_PENALTY

        # Damage to P2 (from P1's blade)
        dmg_to_p2 = CLASH_DAMAGE * (ratio1 * 2)
        if p2_parry:
            dmg_to_p2 *= PARRY_REDUCTION
        if self._p2.lunging:
            dmg_to_p2 *= LUNGE_PENALTY

        # Lunge bonus on attacker
        if self._p1.lunging:
            dmg_to_p2 *= LUNGE_BONUS
        if self._p2.lunging:
            dmg_to_p1 *= LUNGE_BONUS

        self._p1.take_damage(dmg_to_p1, now)
        self._p2.take_damage(dmg_to_p2, now)

        # Drama text
        if p1_parry or p2_parry:
            self._drama_txt = "PARRY!"
        elif self._p1.lunging or self._p2.lunging:
            self._drama_txt = "LUNGE!"
        elif max(ratio1, ratio2) > 0.65:
            self._drama_txt = "CLASH!"
        else:
            self._drama_txt = "HIT!"
        self._drama_ts = now

    def _do_win(self, frame, now):
        fw, fh = self._fw, self._fh
        elapsed = now - self._state_ts
        col = C_P1_BLADE if self._winner == 1 else C_P2_BLADE

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, fh), (8, 4, 18), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

        txt = f"P{self._winner}  WINS"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 2.4, 4)
        shadow_text(frame, txt, (fw//2-tw//2, fh//2-20), FONT, 2.4, col, 4)

        sub = "Show both hands to rematch"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.65, 1)
        shadow_text(frame, sub, (fw//2-sw//2, fh//2+45), FONT, 0.65, C_TEXT)

        if elapsed > 3.5:
            self._reset()

    def _draw_hud(self, frame):
        fw, fh = self._fw, self._fh

        # Banner
        banner_h = 44
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, banner_h), (10, 6, 22), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        shadow_text(frame, "SWORD FIGHT",
                    (fw//2 - 65, 32), FONT, 0.85, C_TEXT, 1)

        # HP bars
        def hp_bar(x, y, hp, col, flip=False):
            bw = int(fw * 0.28);  bh = 18
            filled = int(bw * clamp(hp / MAX_HP, 0, 1))
            if flip:
                x0, x1b = x + bw - filled, x + bw
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (35, 30, 50), -1)
                if filled > 0:
                    c = col if hp > MAX_HP*0.4 else C_HP_LOW if hp > MAX_HP*0.2 else C_HP_CRIT
                    cv2.rectangle(frame, (x0, y), (x1b, y+bh), c, -1)
            else:
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (35, 30, 50), -1)
                if filled > 0:
                    c = col if hp > MAX_HP*0.4 else C_HP_LOW if hp > MAX_HP*0.2 else C_HP_CRIT
                    cv2.rectangle(frame, (x, y), (x+filled, y+bh), c, -1)
            # Border
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (80, 70, 110), 1)
            txt = f"P{'1' if not flip else '2'}  {int(hp)} HP"
            shadow_text(frame, txt, (x+4, y+bh-4), FONTS, 0.45, C_TEXT)

        margin = 12
        hp_bar(margin, fh - 32, self._p1.hp, C_P1_BLADE, flip=False)
        hp_bar(fw - int(fw*0.28) - margin, fh - 32, self._p2.hp, C_P2_BLADE, flip=True)

        # Lunge indicators
        if self._p1 and self._p1.lunging:
            shadow_text(frame, "LUNGE", (margin, fh-38), FONTS, 0.45, C_LUNGE)
        if self._p2 and self._p2.lunging:
            shadow_text(frame, "LUNGE", (fw - 75, fh-38), FONTS, 0.45, C_LUNGE)

    # ── Hand detection ────────────────────────────────────────

    def _find_hands(self, keypoints, detections):
        fw = self._fw
        left, right = None, None

        for inst in keypoints:
            if not inst:
                continue
            x, y, conf = inst[0]
            if conf < 0.20:
                good = [(kx, ky) for kx, ky, kc in inst if kc > 0.20]
                if not good:
                    continue
                x = sum(p[0] for p in good) / len(good)
                y = sum(p[1] for p in good) / len(good)
            if x < fw / 2:
                left = (x, y)
            else:
                right = (x, y)

        for (label, conf, x1, y1, x2, y2) in detections:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if cx < fw/2 and not left:
                left = (cx, cy)
            elif cx >= fw/2 and not right:
                right = (cx, cy)

        return left, right

    # ── Reset ─────────────────────────────────────────────────

    def _reset(self):
        fw, fh = self._fw, self._fh
        self._p1 = Player("left",  fw, fh)
        self._p2 = Player("right", fw, fh)
        self._particles.clear()
        self._clash_cooldown = 0.0
        self._winner = 0
        self._drama_txt = ""
        self._state = "waiting"
        self._t_last = time.perf_counter()