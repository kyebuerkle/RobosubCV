#!/usr/bin/env python3
"""
game_bounce.py — Multi-Ball Hand Bounce
-----------------------------------------
  A / D    : add / remove a ball
  SPACE    : pause

Controls
--------
  Open hand   : slap balls — fingertips/DIP joints are individual hit points.
                Moving points add speed; stationary ones just reflect.
  Closed fist : catch a slow-moving ball (stationary fist),
                OR punch it (moving fist reflects + adds fist speed).

Charge explosion
----------------
  Hold a fist closed for CHARGE_TIER_SEC seconds → a circle timer appears.
  Each additional CHARGE_TIER_SEC grows the charge level (shown as rings).
  Opening the hand releases an explosion that blasts all nearby balls outward.
  Force and radius scale with charge level.
"""

from __future__ import annotations
import math, random, time
from collections import deque
import cv2, numpy as np
from game_plugin import GamePlugin


# ═══════════════════════════════════════════════════════════════
#  ── TUNABLE CONSTANTS ── edit these to adjust gameplay ──────
# ═══════════════════════════════════════════════════════════════

# ── Ball ──────────────────────────────────────────────────────
BALL_RADIUS        = 22      # px — radius of each ball
BALL_FRICTION      = 0.990   # velocity multiplier per frame  (< 1 = slows down)
BALL_MIN_SPEED     = 80      # px/s — balls never fully stop
BALL_MAX_SPEED     = 1600    # px/s — hard speed cap
BALL_SPAWN_SPEED   = (220, 480)   # (min, max) px/s on spawn

MAX_BALLS          = 8       # maximum balls at once

# ── Hand / keypoint detection ─────────────────────────────────
KP_CONF_THRESH     = 0.18    # min keypoint confidence to use a point
FIST_SPREAD_RATIO  = 1.45    # keypoint spread / hand-size below this = fist
                             # lower = harder to trigger fist
FIST_VOTE_WINDOW   = 6       # frames majority-voted for fist state
MOVE_VOTE_WINDOW   = 5       # frames majority-voted for "is moving"
MOVE_PX_THRESH     = 5.0     # px/frame displacement below = stationary

# ── Hit physics ───────────────────────────────────────────────
KP_HIT_RADIUS      = 28      # px — each keypoint's hit zone radius
CATCH_RADIUS       = 48      # px — fist catch zone (from palm centre)
CATCH_SPEED_LIMIT  = 200     # px/s — palm must be slower than this to catch
HIT_COOLDOWN       = 0.05    # s  — min time between hits on same ball / same kp
MOVE_IMPULSE_MIN   = 80      # px/s — hand speed below this adds no energy
MOVE_IMPULSE_MAX   = 1800    # px/s — hand speed that gives maximum energy add
SWEEP_MIN_SPEED    = 80      # px/s — keypoint speed to enable sweep detection

# ── Charge explosion ──────────────────────────────────────────
CHARGE_TIER_SEC    = 5.0     # seconds per charge tier
CHARGE_MAX_TIERS   = 4       # maximum charge levels
CHARGE_BASE_FORCE  = 1000     # px/s added to each ball at tier 1
CHARGE_FORCE_SCALE = 2     # multiplier per additional tier
CHARGE_BASE_RADIUS = 140     # px — explosion radius at tier 1
CHARGE_RADIUS_GROW = 80      # px added per tier
CHARGE_TIMER_THICK = 8       # px — arc thickness for circle timer

# ── Visual ────────────────────────────────────────────────────
TRAIL_LEN          = 14      # ball trail length in frames
FLASH_DUR          = 0.09    # s — ball flash on hit
BALL_COLOURS = [
    (255, 220,  40), (100, 220, 255), (255, 100, 160),
    (80,  255, 140), (255, 160,  60), (180, 120, 255),
]

# ═══════════════════════════════════════════════════════════════

def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _lerp(a, b, t):    return a + (b - a) * t

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX

HIT_KP_IDS  = [4, 8, 12, 16, 20, 3, 7, 11, 15, 19]  # fingertips + DIP
WRIST_ID    = 0


# ── Fist detection ────────────────────────────────────────────

def _raw_is_fist(inst) -> bool:
    """Spread-based fist check. True = closed / insufficient data."""
    if not inst or len(inst) < 21:
        return True
    good = [(x, y) for x, y, c in inst if c > KP_CONF_THRESH]
    if len(good) < 5:
        return True
    wx, wy = inst[0][0], inst[0][1]
    mx, my = inst[9][0], inst[9][1]
    ref = math.hypot(mx - wx, my - wy)
    if ref < 1:
        return True
    max_dist = max(
        (math.hypot(good[i][0]-good[j][0], good[i][1]-good[j][1])
         for i in range(len(good)) for j in range(i+1, len(good))),
        default=0.0)
    return (max_dist / ref) < FIST_SPREAD_RATIO


# ── Sweep detection ───────────────────────────────────────────

def _segment_ball_intersect(px, py, cx, cy, bx, by, r):
    dx = cx-px; dy = cy-py
    fx = px-bx; fy = py-by
    a = dx*dx + dy*dy
    if a < 1e-6: return None
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0: return None
    sq = math.sqrt(disc)
    for t in ((-b-sq)/(2*a), (-b+sq)/(2*a)):
        if 0.0 <= t <= 1.0:
            return px+t*dx, py+t*dy, t
    return None


# ── Keypoint tracker ──────────────────────────────────────────

class KPTracker:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.px = x; self.py = y
        self.vx = 0.0; self.vy = 0.0
        self._votes: deque[bool] = deque(maxlen=MOVE_VOTE_WINDOW)
        self.moving = False

    def update(self, x, y, dt):
        self.px, self.py = self.x, self.y
        alpha = _clamp(0.35 + dt*4, 0, 1)
        self.x = _lerp(self.x, x, alpha)
        self.y = _lerp(self.y, y, alpha)
        self.vx = _lerp(self.vx, (self.x-self.px)/max(dt,1e-4), 0.40)
        self.vy = _lerp(self.vy, (self.y-self.py)/max(dt,1e-4), 0.40)
        self._votes.append(math.hypot(self.x-self.px, self.y-self.py) > MOVE_PX_THRESH)
        self.moving = sum(self._votes) > len(self._votes)/2

    @property
    def speed(self): return math.hypot(self.vx, self.vy)


# ── Hand tracker ─────────────────────────────────────────────

class Hand:
    def __init__(self):
        self.active = False
        self.closed = False
        self.cx = 0.0; self.cy = 0.0
        self._prev_cx = 0.0; self._prev_cy = 0.0
        self.kps: dict[int, KPTracker] = {}
        self._fist_votes: deque[bool] = deque(maxlen=FIST_VOTE_WINDOW)
        self._cool: dict[tuple, float] = {}
        # Charge state
        self.fist_since:  float | None = None  # when fist started
        self.charge_tier: int = 0             # current tier
        self.prev_tier:   int = 0             # tier from last frame (for release)
        self.last_seen:   float = 0.0         # timestamp of last real detection

    def update(self, inst, dt, now):
        self._fist_votes.append(_raw_is_fist(inst))
        was_closed = self.closed
        self.closed = sum(self._fist_votes) > len(self._fist_votes)/2

        self._prev_cx, self._prev_cy = self.cx, self.cy
        good = [(x,y,c) for x,y,c in inst if c > KP_CONF_THRESH]
        if good:
            self.cx = sum(p[0] for p in good)/len(good)
            self.cy = sum(p[1] for p in good)/len(good)

        for kid in HIT_KP_IDS:
            if kid >= len(inst): continue
            x, y, conf = inst[kid]
            if conf < KP_CONF_THRESH + 0.04: continue
            if kid not in self.kps:
                self.kps[kid] = KPTracker(x, y)
            else:
                self.kps[kid].update(x, y, dt)

        # Track fist hold time for charge
        self.prev_tier = self.charge_tier   # snapshot before state update
        self.last_seen = now                 # hand was seen this frame

        if self.closed:
            if self.fist_since is None:
                self.fist_since = now
            held = now - self.fist_since
            self.charge_tier = min(int(held / CHARGE_TIER_SEC), CHARGE_MAX_TIERS)
        else:
            self.fist_since = None
            self.charge_tier = 0

        self._cool = {k:t for k,t in self._cool.items()
                      if now-t < HIT_COOLDOWN}
        self.active = True

    @property
    def palm_speed(self):
        return math.hypot(self.cx-self._prev_cx, self.cy-self._prev_cy)

    def reset_charge(self):
        """Call whenever the hand leaves the frame — clears all charge state."""
        self.fist_since  = None
        self.charge_tier = 0
        self.prev_tier   = 0
        self._fist_votes.clear()   # also clear vote window so stale fist votes don't linger

    def can_hit(self, bid, kid, now):
        return now - self._cool.get((bid,kid),-999) >= HIT_COOLDOWN

    def mark_hit(self, bid, kid, now):
        self._cool[(bid,kid)] = now


# ── Ball ─────────────────────────────────────────────────────

class Ball:
    _cidx = 0
    def __init__(self, fw, fh):
        cx, cy = fw/2, fh/2
        self.x = cx + random.uniform(-cx*0.4, cx*0.4)
        self.y = cy + random.uniform(-cy*0.4, cy*0.4)
        ang = random.uniform(0, 2*math.pi)
        spd = random.uniform(*BALL_SPAWN_SPEED)
        self.vx = math.cos(ang)*spd; self.vy = math.sin(ang)*spd
        self.r = BALL_RADIUS
        self.colour = BALL_COLOURS[Ball._cidx % len(BALL_COLOURS)]
        Ball._cidx += 1
        self.trail: list = []
        self.flash_t = -99.0
        self.caught = False
        self.catch_off = (0.0, 0.0)

    def update(self, dt, fw, fh):
        if self.caught: return
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LEN: self.trail.pop(0)
        self.vx *= BALL_FRICTION; self.vy *= BALL_FRICTION
        spd = math.hypot(self.vx, self.vy)
        if spd < BALL_MIN_SPEED:
            if spd < 1e-3:
                ang = random.uniform(0, 2*math.pi)
                self.vx = math.cos(ang)*BALL_MIN_SPEED
                self.vy = math.sin(ang)*BALL_MIN_SPEED
            else:
                self.vx = self.vx/spd*BALL_MIN_SPEED
                self.vy = self.vy/spd*BALL_MIN_SPEED
        self.x += self.vx*dt; self.y += self.vy*dt
        if self.x-self.r < 0:   self.x=float(self.r);     self.vx= abs(self.vx)
        if self.x+self.r > fw:  self.x=float(fw-self.r);  self.vx=-abs(self.vx)
        if self.y-self.r < 0:   self.y=float(self.r);     self.vy= abs(self.vy)
        if self.y+self.r > fh:  self.y=float(fh-self.r);  self.vy=-abs(self.vy)

    def draw(self, frame, now):
        bx, by, r = int(self.x), int(self.y), self.r
        for i,(tx,ty) in enumerate(self.trail):
            a  = (i/max(len(self.trail)-1,1))*0.45
            tr = max(2, int(r*(i/max(len(self.trail)-1,1))*0.65))
            ov = frame.copy()
            cv2.circle(ov,(int(tx),int(ty)),tr,self.colour,-1,cv2.LINE_AA)
            cv2.addWeighted(ov,a*0.5,frame,1-a*0.5,0,frame)
        flash = (now-self.flash_t) < FLASH_DUR
        gcol  = (255,255,255) if flash else tuple(min(255,c+80) for c in self.colour)
        ov = frame.copy()
        cv2.circle(ov,(bx,by),r+8 if flash else r+5,gcol,-1,cv2.LINE_AA)
        cv2.addWeighted(ov,0.28,frame,0.72,0,frame)
        cv2.circle(frame,(bx,by),r,(220,240,255) if flash else self.colour,-1,cv2.LINE_AA)
        cv2.circle(frame,(bx-r//3,by-r//3),max(3,r//4),(255,255,255),-1,cv2.LINE_AA)
        if self.caught:
            cv2.circle(frame,(bx,by),r+4,(255,255,255),1,cv2.LINE_AA)


# ── Game ─────────────────────────────────────────────────────

class Game(GamePlugin):

    def __init__(self):
        super().__init__()
        self._fw = 640; self._fh = 480
        self._balls: list[Ball] = []
        self._hands = [Hand(), Hand()]
        self._score = 0
        self._t_last = time.perf_counter()
        self._started = False
        self._explosions: list[dict] = []   # for visual effects

    def on_start(self, fw, fh, class_names=None):
        self._fw, self._fh = fw, fh
        self._balls  = [Ball(fw, fh)]
        self._score  = 0
        self._t_last = time.perf_counter()
        self._started = True

    def on_frame(self, frame, keypoints, detections, fw, fh):
        if not self._started: self.on_start(fw, fh)
        now = time.perf_counter()
        dt  = min(now-self._t_last, 0.07)
        self._t_last = now
        self._fw, self._fh = fw, fh

        # ── Update hands ──────────────────────────────────────
        HAND_TIMEOUT = 2.0   # seconds unseen before charge resets
        active_hands: list[Hand] = []
        seen_indices = set()
        for i, inst in enumerate(keypoints[:2]):
            if inst:
                self._hands[i].update(inst, dt, now)
                self._hands[i].active = True
                active_hands.append(self._hands[i])
                seen_indices.add(i)
        # For hands not seen this frame: keep charge alive for HAND_TIMEOUT seconds,
        # then reset so a long absence clears the charge but brief dropouts don't.
        for i, h in enumerate(self._hands):
            if i not in seen_indices:
                h.active = False
                if h.last_seen > 0 and (now - h.last_seen) > HAND_TIMEOUT:
                    h.reset_charge()
        if not active_hands:
            for j,(label,conf,x1,y1,x2,y2) in enumerate(detections[:2]):
                if conf > 0.25:
                    fake = [(((x1+x2)/2),((y1+y2)/2),0.1)]*21
                    self._hands[j].update(fake, dt, now)
                    self._hands[j].closed = True
                    active_hands.append(self._hands[j])

        # ── Check for charge release (fist→open) ──────────────
        for h in active_hands:
            # prev_tier holds what the charge was BEFORE this frame's update
            # (charge_tier is already 0 once hand opens)
            if not h.closed and h.prev_tier > 0:
                tier   = h.prev_tier
                force  = CHARGE_BASE_FORCE * (CHARGE_FORCE_SCALE ** (tier - 1))
                radius = CHARGE_BASE_RADIUS + CHARGE_RADIUS_GROW * (tier - 1)
                self._explode(h.cx, h.cy, force, radius, now)
                self._explosions.append({
                    'x': h.cx, 'y': h.cy, 'r': radius,
                    'tier': tier, 'ts': now, 'dur': 0.55
                })

        # ── Ball-hand interaction ─────────────────────────────
        for ball in self._balls:
            bid = id(ball)

            if ball.caught:
                holder = next(
                    (h for h in active_hands
                     if math.hypot(ball.x-h.cx,ball.y-h.cy) < CATCH_RADIUS+ball.r+15),
                    None)
                if holder is None or not holder.closed:
                    ball.caught = False
                    if holder and WRIST_ID in holder.kps:
                        kp = holder.kps[WRIST_ID]
                        ball.vx, ball.vy = kp.vx, kp.vy
                    spd = math.hypot(ball.vx, ball.vy)
                    if spd < BALL_MIN_SPEED:
                        ang = math.atan2(ball.vy, ball.vx)
                        ball.vx = math.cos(ang)*BALL_MIN_SPEED
                        ball.vy = math.sin(ang)*BALL_MIN_SPEED
                else:
                    ball.x = holder.cx + ball.catch_off[0]
                    ball.y = holder.cy + ball.catch_off[1]
                continue

            for h in active_hands:
                if h.closed:
                    dist = math.hypot(ball.x-h.cx, ball.y-h.cy)
                    ps   = h.palm_speed / max(dt, 1e-4)
                    if dist < CATCH_RADIUS + ball.r:
                        if ps < CATCH_SPEED_LIMIT and h.can_hit(bid,-1,now):
                            # Catch
                            ball.caught = True
                            ball.catch_off = (ball.x-h.cx, ball.y-h.cy)
                            ball.vx = 0.0; ball.vy = 0.0
                            ball.flash_t = now
                            h.mark_hit(bid,-1,now)
                            self._score += 1
                        elif ps >= CATCH_SPEED_LIMIT and h.can_hit(bid,-1,now):
                            # Punch
                            if dist < 1: nx,ny = 1.0,0.0
                            else:        nx,ny = (ball.x-h.cx)/dist,(ball.y-h.cy)/dist
                            dot = ball.vx*nx+ball.vy*ny
                            ball.vx -= 2*dot*nx; ball.vy -= 2*dot*ny
                            ball.vx += nx*ps*0.6; ball.vy += ny*ps*0.6
                            spd = math.hypot(ball.vx,ball.vy)
                            if spd > BALL_MAX_SPEED:
                                ball.vx=ball.vx/spd*BALL_MAX_SPEED
                                ball.vy=ball.vy/spd*BALL_MAX_SPEED
                            ol = CATCH_RADIUS+ball.r - dist + 2
                            ball.x += nx*ol; ball.y += ny*ol
                            h.mark_hit(bid,-1,now)
                            ball.flash_t = now
                    continue

                # Open hand — per-keypoint hits
                for kid, kp in h.kps.items():
                    contact = KP_HIT_RADIUS + ball.r
                    if not h.can_hit(bid,kid,now): continue
                    dist = math.hypot(ball.x-kp.x, ball.y-kp.y)
                    sweep = None
                    if kp.speed > SWEEP_MIN_SPEED and dist < contact*3:
                        sweep = _segment_ball_intersect(
                            kp.px,kp.py,kp.x,kp.y,ball.x,ball.y,float(contact))
                    if dist > contact and sweep is None: continue

                    if sweep:
                        hx,hy,_ = sweep
                        ddx,ddy = ball.x-hx, ball.y-hy
                        dn = math.hypot(ddx,ddy)
                        nx,ny = (ddx/dn,ddy/dn) if dn>1 else (1.0,0.0)
                    elif dist<1: nx,ny=1.0,0.0
                    else: nx,ny=(ball.x-kp.x)/dist,(ball.y-kp.y)/dist

                    dot = ball.vx*nx+ball.vy*ny
                    ball.vx -= 2*dot*nx; ball.vy -= 2*dot*ny

                    hand_dot = kp.vx*nx+kp.vy*ny
                    if hand_dot < -MOVE_IMPULSE_MIN:
                        contrib = abs(hand_dot)
                        ts = _clamp((contrib-MOVE_IMPULSE_MIN)/max(MOVE_IMPULSE_MAX-MOVE_IMPULSE_MIN,1),0,1)
                        ball.vx += nx*contrib*ts; ball.vy += ny*contrib*ts

                    if sweep:
                        ts = _clamp((kp.speed-MOVE_IMPULSE_MIN)/max(MOVE_IMPULSE_MAX-MOVE_IMPULSE_MIN,1),0,1)
                        ball.vx += nx*kp.speed*ts*0.7; ball.vy += ny*kp.speed*ts*0.7

                    spd = math.hypot(ball.vx,ball.vy)
                    if spd > BALL_MAX_SPEED:
                        ball.vx=ball.vx/spd*BALL_MAX_SPEED
                        ball.vy=ball.vy/spd*BALL_MAX_SPEED
                    if dist < contact:
                        ol = contact-dist+2; ball.x+=nx*ol; ball.y+=ny*ol
                    h.mark_hit(bid,kid,now); ball.flash_t=now; self._score+=1

        # ── Ball-ball collisions ──────────────────────────────
        for i in range(len(self._balls)):
            for j in range(i+1,len(self._balls)):
                a,b = self._balls[i],self._balls[j]
                if a.caught and b.caught: continue
                dx,dy = b.x-a.x, b.y-a.y
                dist = math.hypot(dx,dy)
                md = a.r+b.r
                if dist>=md or dist<1e-3: continue
                nx,ny = dx/dist,dy/dist
                ol = md-dist
                if not a.caught: a.x-=nx*ol*0.5; a.y-=ny*ol*0.5
                if not b.caught: b.x+=nx*ol*0.5; b.y+=ny*ol*0.5
                if not a.caught and not b.caught:
                    av=a.vx*nx+a.vy*ny; bv=b.vx*nx+b.vy*ny
                    a.vx+=(bv-av)*nx; a.vy+=(bv-av)*ny
                    b.vx+=(av-bv)*nx; b.vy+=(av-bv)*ny

        # ── Physics ───────────────────────────────────────────
        for ball in self._balls:
            ball.update(dt,fw,fh)

        # ── Draw ─────────────────────────────────────────────
        for ball in self._balls: ball.draw(frame,now)
        self._draw_explosions(frame,now)
        self._draw_hands(frame,active_hands,now)
        self._draw_hud(frame,fw)
        return frame

    def on_stop(self): self._started = False

    def tune(self, key):
        if key.lower()=='a' and len(self._balls)<MAX_BALLS:
            self._balls.append(Ball(self._fw,self._fh))
        elif key.lower()=='d' and len(self._balls)>1:
            self._balls.pop()

    # ── Explosion ─────────────────────────────────────────────

    def _explode(self, ex, ey, force, radius, now):
        for ball in self._balls:
            if ball.caught:
                ball.caught = False
            dx,dy = ball.x-ex, ball.y-ey
            dist = math.hypot(dx,dy)
            if dist > radius: continue
            if dist < 1: nx,ny=random.uniform(-1,1),random.uniform(-1,1)
            else:        nx,ny=dx/dist,dy/dist
            # Force falloff: full at centre, zero at edge
            falloff = 1.0 - (dist/radius)
            f = force * (falloff**0.5)
            ball.vx += nx*f; ball.vy += ny*f
            spd = math.hypot(ball.vx,ball.vy)
            if spd > BALL_MAX_SPEED:
                ball.vx=ball.vx/spd*BALL_MAX_SPEED
                ball.vy=ball.vy/spd*BALL_MAX_SPEED
            ball.flash_t = now

    # ── Drawing ───────────────────────────────────────────────

    def _draw_hands(self, frame, hands, now):
        for h in hands:
            fist_col  = (60,  60, 220)
            hit_col   = (60, 220,  60)
            idle_col  = (80, 140,  80)
            for kid,kp in h.kps.items():
                if h.closed: col = fist_col
                else:        col = hit_col if kp.moving else idle_col
                cv2.circle(frame,(int(kp.x),int(kp.y)),5,col,-1,cv2.LINE_AA)

            if h.closed and h.fist_since is not None:
                held  = now - h.fist_since
                if held < 2.5:          # don't show timer for first 2.5s
                    continue
                tier  = h.charge_tier
                frac  = (held % CHARGE_TIER_SEC) / CHARGE_TIER_SEC
                cx,cy = int(h.cx), int(h.cy)
                r_ring = CATCH_RADIUS + 16 + tier*10

                # Background ring
                cv2.circle(frame,(cx,cy),r_ring,(40,40,60),CHARGE_TIMER_THICK,cv2.LINE_AA)

                # Charge colour: yellow→orange→red per tier
                tier_cols = [
                    (60, 220, 255),   # tier 0 → cyan
                    (60, 180, 255),   # tier 1 → blue-orange
                    (40, 120, 255),   # tier 2 → orange
                    (30,  60, 255),   # tier 3 → red
                    (20,  20, 255),   # tier 4 → deep red
                ]
                arc_col = tier_cols[min(tier, len(tier_cols)-1)]

                # Arc for current tier progress
                angle_end = int(frac * 360)
                if angle_end > 0:
                    cv2.ellipse(frame,(cx,cy),(r_ring,r_ring),
                                -90, 0, angle_end, arc_col, CHARGE_TIMER_THICK, cv2.LINE_AA)

                # Completed tier dots around the ring
                for t in range(tier):
                    dot_ang = math.radians(-90 + t*(360//max(CHARGE_MAX_TIERS,1)))
                    dx2 = int(cx + (r_ring+14)*math.cos(dot_ang))
                    dy2 = int(cy + (r_ring+14)*math.sin(dot_ang))
                    cv2.circle(frame,(dx2,dy2),6,arc_col,-1,cv2.LINE_AA)

                # Tier label — only show if genuinely charged, capped at max
                show_tier = min(tier, CHARGE_MAX_TIERS)
                lbl = f"x{show_tier+1}" if show_tier >= 1 else ""
                if lbl:
                    (tw,_),_ = cv2.getTextSize(lbl,FONTS,0.65,2)
                    cv2.putText(frame,lbl,(cx-tw//2,cy+6),FONTS,0.65,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(frame,lbl,(cx-tw//2,cy+6),FONTS,0.65,arc_col,1,cv2.LINE_AA)

    def _draw_explosions(self, frame, now):
        still = []
        for ex in self._explosions:
            age = now - ex['ts']
            if age > ex['dur']:
                continue
            still.append(ex)
            a    = 1.0 - age/ex['dur']
            r    = int(ex['r'] * (0.5 + 0.5*age/ex['dur']))
            tier = ex['tier']
            cols = [(60,220,255),(60,180,255),(40,120,255),(30,60,255),(20,20,255)]
            col  = cols[min(tier-1, len(cols)-1)]
            ov   = frame.copy()
            cv2.circle(ov,(int(ex['x']),int(ex['y'])),r,col,3,cv2.LINE_AA)
            cv2.addWeighted(ov, a*0.7, frame, 1-a*0.7, 0, frame)
            # Inner flash
            inner = max(4, int(r*0.3*a))
            ov2 = frame.copy()
            cv2.circle(ov2,(int(ex['x']),int(ex['y'])),inner,(255,255,255),-1,cv2.LINE_AA)
            cv2.addWeighted(ov2, a*0.5, frame, 1-a*0.5, 0, frame)
        self._explosions = still

    def _draw_hud(self, frame, fw):
        banner_h = 40
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(fw,banner_h),(10,8,22),-1)
        cv2.addWeighted(ov,0.60,frame,0.40,0,frame)
        cv2.putText(frame,"BOUNCE",(fw//2-42,28),FONT,0.85,(200,200,255),1,cv2.LINE_AA)
        cv2.putText(frame,f"Hits: {self._score}",(fw-110,28),FONTS,0.65,(100,255,180),1,cv2.LINE_AA)
        cv2.putText(frame,f"A/D  balls: {len(self._balls)}/{MAX_BALLS}",
                    (8,28),FONTS,0.52,(180,180,220),1,cv2.LINE_AA)