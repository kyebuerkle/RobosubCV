#!/usr/bin/env python3
"""
YOLO Live Stream Viewer — CPU Edition  (Pose + Game Plugins)
---------------------------------------------------------------------
Squeezes maximum FPS out of CPU-only inference via:
  - OpenCV thread tuning
  - Configurable inference size (imgsz)
  - Frame skip (infer every N frames, display all)
  - ONNX / OpenVINO backend options
  - Threaded 3-stage pipeline: capture → infer → render

Pose features (via pose_renderer.py):
  - Keypoint dots and skeleton lines drawn on frame
  - Per-element colour and size controls in UI
  - Works with 21-pt hand models and 17-pt COCO body models

Tracking features (via object_tracker.py):
  - Centroid / IoU multi-object tracking
  - Occlusion memory and exponential box smoothing
  - Majority-vote label smoothing

Game plugin features:
  - Browse any Python script that defines a Game(GamePlugin) class
  - Game receives pose keypoints each frame and can draw on the frame
  - Hot-swap games without restarting the stream

Requirements:
    pip install ultralytics opencv-python pillow numpy
    pip install onnxruntime          ← optional, big speed boost on CPU
    pip install openvino             ← optional, Intel CPUs
    object_tracker.py   ← same directory
    pose_renderer.py    ← same directory
    game_plugin.py      ← same directory
    game_bounce.py      ← example game (optional)
"""

import importlib.util
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from object_tracker import ObjectTracker
from pose_renderer import PoseRenderer, PoseStyles, extract_keypoints


# ──────────────────────────────────────────────
#  CPU info
# ──────────────────────────────────────────────

def get_cpu_info() -> tuple[str, int]:
    core_count = 1
    try:
        import os
        core_count = os.cpu_count() or 1
    except Exception:
        pass

    name = "Unknown CPU"
    try:
        import platform
        name = platform.processor() or platform.machine()
    except Exception:
        pass

    try:
        import subprocess
        if sys.platform == "win32":
            out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True)
            lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"]
            if lines:
                name = lines[0]
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        name = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    return name, core_count


CPU_NAME, CPU_CORES = get_cpu_info()


def check_onnxruntime() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def check_openvino() -> bool:
    try:
        import openvino
        return True
    except ImportError:
        return False


HAS_ONNX = check_onnxruntime()
HAS_OV   = check_openvino()


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def find_cameras(max_index: int = 10) -> list[dict]:
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            backend = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
            cameras.append({"index": idx, "label": f"Camera {idx}  [{backend}]"})
            cap.release()
    return cameras


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def load_game_plugin(path: str):
    """Dynamically import a game script and return an instance of its Game class."""
    spec = importlib.util.spec_from_file_location("_game_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Game()


# ──────────────────────────────────────────────
#  Per-label style store
# ──────────────────────────────────────────────

DEFAULT_COLOR_BGR = (0, 255, 0)
DEFAULT_THICKNESS = 2


class LabelStyles:
    def __init__(self):
        self._styles: dict[str, dict] = {}

    def get(self, label: str) -> dict:
        return self._styles.get(label, {
            "color_bgr": DEFAULT_COLOR_BGR,
            "thickness": DEFAULT_THICKNESS,
        })

    def set(self, label: str, color_bgr: tuple, thickness: int):
        self._styles[label] = {"color_bgr": color_bgr, "thickness": thickness}


# ──────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────

class YoloStreamApp(tk.Tk):
    RENDER_DELAY_MS  = 16
    FRAME_QUEUE_SIZE = 2

    def __init__(self):
        super().__init__()
        self.title("YOLO Live Stream — CPU Edition  ✦ Pose + Games")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        # State
        self.model: YOLO | None = None
        self.model_path_str = tk.StringVar(value="No model loaded")
        self.cap: cv2.VideoCapture | None = None
        self.streaming = False
        self.cameras: list[dict] = []
        self.label_styles = LabelStyles()
        self._loaded_pt_path: str = ""

        # Pipeline
        self._raw_queue: queue.Queue     = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._display_queue: queue.Queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)

        # ── Tuning variables ──
        self.confidence   = tk.DoubleVar(value=0.50)
        self.imgsz_var    = tk.IntVar(value=320)
        self.skip_var     = tk.IntVar(value=2)
        self.cv_threads   = tk.IntVar(value=max(1, CPU_CORES // 2))
        self.cam_res_var  = tk.StringVar(value="640x480")
        self.backend_var  = tk.StringVar(value="pytorch")

        # ── Tracking settings ──
        self.tracking_enabled    = tk.BooleanVar(value=True)
        self.iou_threshold_var   = tk.DoubleVar(value=0.30)
        self.max_lost_frames_var = tk.IntVar(value=15)
        self.smooth_alpha_var    = tk.DoubleVar(value=0.40)
        self.label_smooth_var    = tk.IntVar(value=5)

        # ── Pose settings ──
        self.pose_styles   = PoseStyles()
        self.pose_renderer = PoseRenderer(self.pose_styles)
        self.show_boxes_var     = tk.BooleanVar(value=True)   # bounding boxes
        self.show_keypoints_var = tk.BooleanVar(value=True)
        self.show_skeleton_var  = tk.BooleanVar(value=True)

        # ── Mirror ──
        self.mirror_var = tk.BooleanVar(value=True)

        # ── Game plugin ──
        self._game        = None      # current game instance
        self._game_path   = tk.StringVar(value="No game loaded")
        self._game_lock   = threading.Lock()
        self._last_kps: list = []     # keypoints from last inference (for game)

        # ── Tracker ──
        self.tracker = ObjectTracker(
            iou_threshold       = self.iou_threshold_var.get(),
            max_lost_frames     = self.max_lost_frames_var.get(),
            smooth_alpha        = self.smooth_alpha_var.get(),
            label_smooth_frames = self.label_smooth_var.get(),
        )

        # FPS tracking
        self._fps_times: list[float]       = []
        self._infer_fps_times: list[float] = []

        self._build_ui()
        self._refresh_cameras()
        self._apply_cv_threads()

    # ── Sync tracker params from UI vars ─────

    def _sync_tracker(self, *_):
        self.tracker.iou_threshold       = self.iou_threshold_var.get()
        self.tracker.max_lost_frames     = self.max_lost_frames_var.get()
        self.tracker.smooth_alpha        = self.smooth_alpha_var.get()
        self.tracker.label_smooth_frames = self.label_smooth_var.get()

    def _sync_pose_styles(self, *_):
        self.pose_styles.show_keypoints = self.show_keypoints_var.get()
        self.pose_styles.show_skeleton  = self.show_skeleton_var.get()

    # ─────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────

    def _build_ui(self):
        PAD      = 10
        PANEL_BG = "#2a2a3e"
        FG       = "#cdd6f4"
        ACCENT   = "#89b4fa"
        BTN_BG   = "#313244"
        WARN     = "#fab387"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",       background=PANEL_BG)
        style.configure("TLabel",       background=PANEL_BG, foreground=FG,  font=("Segoe UI", 10))
        style.configure("TButton",      background=BTN_BG,   foreground=FG,  font=("Segoe UI", 10), borderwidth=0)
        style.configure("TScale",       background=PANEL_BG)
        style.configure("TCombobox",    fieldbackground=BTN_BG, background=BTN_BG, foreground=FG)
        style.configure("TRadiobutton", background=PANEL_BG, foreground=FG,  font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=PANEL_BG, foreground=FG)
        style.configure("Accent.TButton", background=ACCENT, foreground="#1e1e2e",
                        font=("Segoe UI", 10, "bold"))
        style.map("TButton",        background=[("active", ACCENT)])
        style.map("Accent.TButton", background=[("active", "#74c7ec")])

        def section(parent, text):
            ttk.Label(parent, text=text, foreground=ACCENT,
                      font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))

        def hint(parent, text):
            ttk.Label(parent, text=text, foreground="#6c7086",
                      font=("Segoe UI", 8)).pack(anchor="w")

        def slider_row(parent, var, lo, hi, fmt="{:.2f}"):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(row, text=fmt.format(var.get()), width=7)
            lbl.pack(side=tk.RIGHT)

            def _update(v):
                lbl.configure(text=fmt.format(float(v) if "." in fmt else int(float(v))))
                self._sync_tracker()

            ttk.Scale(row, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=_update
                      ).pack(side=tk.LEFT, expand=True, fill=tk.X)
            return row, lbl

        # ── Scrollable left panel ──────────────
        left_outer = tk.Frame(self, bg=PANEL_BG, width=270)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg=PANEL_BG, highlightthickness=0, width=265)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(canvas_scroll, padding=(PAD, PAD, PAD, PAD))
        ctrl_window = canvas_scroll.create_window((0, 0), window=ctrl, anchor="nw")

        def _on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        def _on_canvas_configure(event):
            canvas_scroll.itemconfig(ctrl_window, width=event.width)

        ctrl.bind("<Configure>", _on_frame_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_configure)
        canvas_scroll.bind_all("<MouseWheel>",
            lambda e: canvas_scroll.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ── CPU badge ──
        tk.Label(ctrl, text=f"🖥  {CPU_NAME}", bg=PANEL_BG, fg=WARN,
                 font=("Segoe UI", 8, "bold"), wraplength=240,
                 justify=tk.LEFT).pack(anchor="w", pady=(0, 2))
        tk.Label(ctrl, text=f"Logical cores: {CPU_CORES}", bg=PANEL_BG,
                 fg="#6c7086", font=("Segoe UI", 8)).pack(anchor="w")

        # ── Backend ──
        section(ctrl, "Inference backend")
        hint(ctrl, "ONNX / OpenVINO are faster on CPU than PyTorch")

        backends = [("PyTorch  (default)", "pytorch")]
        if HAS_ONNX:
            backends.append(("ONNX Runtime  ✓ installed", "onnx"))
        else:
            backends.append(("ONNX Runtime  (pip install onnxruntime)", "onnx"))
        if HAS_OV:
            backends.append(("OpenVINO  ✓ installed  [Intel best]", "openvino"))
        else:
            backends.append(("OpenVINO  (pip install openvino)", "openvino"))

        for label, val in backends:
            ttk.Radiobutton(ctrl, text=label, variable=self.backend_var,
                            value=val).pack(anchor="w")

        # ── Model ──
        section(ctrl, "Model")
        ttk.Label(ctrl, textvariable=self.model_path_str,
                  wraplength=230, foreground="#a6e3a1").pack(anchor="w")
        ttk.Button(ctrl, text="Browse .pt file…",
                   command=self._browse_model).pack(fill=tk.X, pady=4)

        # ── Camera ──
        section(ctrl, "Camera")
        cam_row = ttk.Frame(ctrl)
        cam_row.pack(fill=tk.X)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_var,
                                      state="readonly", width=20)
        self.cam_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cam_row, text="↻", width=3,
                   command=self._refresh_cameras).pack(side=tk.LEFT, padx=(4, 0))

        section(ctrl, "Resolution")
        hint(ctrl, "Lower res = faster capture + less resize work")
        res_row = ttk.Frame(ctrl)
        res_row.pack(fill=tk.X)
        for res in ("320x240", "640x480", "1280x720"):
            ttk.Radiobutton(res_row, text=res, variable=self.cam_res_var,
                            value=res).pack(side=tk.LEFT, padx=2)

        # ── Confidence ──
        section(ctrl, "Confidence threshold")
        conf_row = ttk.Frame(ctrl)
        conf_row.pack(fill=tk.X)
        self.conf_label = ttk.Label(conf_row, text=f"{self.confidence.get():.2f}", width=5)
        self.conf_label.pack(side=tk.RIGHT)
        ttk.Scale(conf_row, from_=0.05, to=0.95, variable=self.confidence,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.conf_label.configure(
                      text=f"{float(v):.2f}")).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ── Inference size ──
        section(ctrl, "Inference size (imgsz)")
        hint(ctrl, "320 ≈ 2-4× faster than 640, slight accuracy drop")
        imgsz_row = ttk.Frame(ctrl)
        imgsz_row.pack(fill=tk.X)
        for sz in (224, 320, 416, 480, 640):
            ttk.Radiobutton(imgsz_row, text=str(sz), variable=self.imgsz_var,
                            value=sz).pack(side=tk.LEFT, padx=2)

        # ── Frame skip ──
        section(ctrl, "Infer every N frames")
        hint(ctrl, "1 = every frame (slowest), 3-4 = good CPU balance")
        skip_row = ttk.Frame(ctrl)
        skip_row.pack(fill=tk.X, pady=2)
        self.skip_label = ttk.Label(skip_row, text=f"N = {self.skip_var.get()}", width=7)
        self.skip_label.pack(side=tk.RIGHT)
        ttk.Scale(skip_row, from_=1, to=8, variable=self.skip_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.skip_label.configure(
                      text=f"N = {int(float(v))}")).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ── OpenCV threads ──
        section(ctrl, "OpenCV CPU threads")
        hint(ctrl, f"Your CPU has {CPU_CORES} logical cores")
        hint(ctrl, "More ≠ always faster — try CPU_CORES/2 first")
        threads_row = ttk.Frame(ctrl)
        threads_row.pack(fill=tk.X, pady=2)
        self.threads_label = ttk.Label(threads_row,
                                       text=f"{self.cv_threads.get()} threads", width=10)
        self.threads_label.pack(side=tk.RIGHT)
        ttk.Scale(threads_row, from_=1, to=CPU_CORES, variable=self.cv_threads,
                  orient=tk.HORIZONTAL,
                  command=lambda v: (
                      self.threads_label.configure(text=f"{int(float(v))} threads"),
                      self._apply_cv_threads()
                  )).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ═══════════════════════════════════════
        #  TRACKING
        # ═══════════════════════════════════════
        section(ctrl, "━━  Object Tracking  ━━")
        ttk.Checkbutton(ctrl, text="Enable tracking & smoothing",
                        variable=self.tracking_enabled).pack(anchor="w", pady=(0, 4))

        hint(ctrl, "Match threshold (IoU) — higher = stricter matching")
        slider_row(ctrl, self.iou_threshold_var, 0.05, 0.90, "{:.2f}")

        section(ctrl, "Occlusion memory (frames)")
        hint(ctrl, "Frames a hidden track stays alive behind an object")
        lost_row = ttk.Frame(ctrl)
        lost_row.pack(fill=tk.X, pady=2)
        lost_lbl = ttk.Label(lost_row, text=str(self.max_lost_frames_var.get()), width=5)
        lost_lbl.pack(side=tk.RIGHT)

        def _lost_update(v):
            lost_lbl.configure(text=str(int(float(v))))
            self._sync_tracker()

        ttk.Scale(lost_row, from_=0, to=60, variable=self.max_lost_frames_var,
                  orient=tk.HORIZONTAL, command=_lost_update
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        section(ctrl, "Box smoothing  (alpha)")
        hint(ctrl, "0 = frozen / no update,  1 = raw / jumpy")
        slider_row(ctrl, self.smooth_alpha_var, 0.01, 1.0, "{:.2f}")

        section(ctrl, "Label smoothing window (frames)")
        hint(ctrl, "Majority-vote over last N frames to stabilise class")
        lbl_win_row = ttk.Frame(ctrl)
        lbl_win_row.pack(fill=tk.X, pady=2)
        lbl_win_lbl = ttk.Label(lbl_win_row, text=str(self.label_smooth_var.get()), width=5)
        lbl_win_lbl.pack(side=tk.RIGHT)

        def _lbl_win_update(v):
            lbl_win_lbl.configure(text=str(int(float(v))))
            self._sync_tracker()

        ttk.Scale(lbl_win_row, from_=1, to=30, variable=self.label_smooth_var,
                  orient=tk.HORIZONTAL, command=_lbl_win_update
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(ctrl, text="Reset Tracks", command=self.tracker.reset).pack(fill=tk.X, pady=4)

        # ═══════════════════════════════════════
        #  POSE OVERLAY
        # ═══════════════════════════════════════
        section(ctrl, "━━  Pose Overlay  ━━")

        ttk.Checkbutton(ctrl, text="Show bounding boxes",
                        variable=self.show_boxes_var).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Mirror (flip horizontal)",
                        variable=self.mirror_var).pack(anchor="w")

        kp_row = ttk.Frame(ctrl)
        kp_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(kp_row, text="Keypoints",
                        variable=self.show_keypoints_var,
                        command=self._sync_pose_styles).pack(side=tk.LEFT)

        # Keypoint colour swatch
        self._kp_color_swatch = tk.Label(kp_row,
            bg=bgr_to_hex(self.pose_styles.keypoint_color),
            width=3, relief="solid", cursor="hand2")
        self._kp_color_swatch.pack(side=tk.LEFT, padx=6)
        self._kp_color_swatch.bind("<Button-1>", self._pick_kp_color)

        sk_row = ttk.Frame(ctrl)
        sk_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(sk_row, text="Skeleton",
                        variable=self.show_skeleton_var,
                        command=self._sync_pose_styles).pack(side=tk.LEFT)

        # Skeleton colour swatch
        self._sk_color_swatch = tk.Label(sk_row,
            bg=bgr_to_hex(self.pose_styles.skeleton_color),
            width=3, relief="solid", cursor="hand2")
        self._sk_color_swatch.pack(side=tk.LEFT, padx=6)
        self._sk_color_swatch.bind("<Button-1>", self._pick_sk_color)

        # Keypoint radius
        kpr_row = ttk.Frame(ctrl)
        kpr_row.pack(fill=tk.X, pady=2)
        ttk.Label(kpr_row, text="Dot size:").pack(side=tk.LEFT)
        self._kp_radius_var = tk.IntVar(value=self.pose_styles.keypoint_radius)
        kpr_lbl = ttk.Label(kpr_row, text=str(self.pose_styles.keypoint_radius), width=3)
        kpr_lbl.pack(side=tk.RIGHT)
        ttk.Scale(kpr_row, from_=2, to=20, variable=self._kp_radius_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: (
                      setattr(self.pose_styles, "keypoint_radius", int(float(v))),
                      kpr_lbl.configure(text=str(int(float(v))))
                  )).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        # Skeleton thickness
        skt_row = ttk.Frame(ctrl)
        skt_row.pack(fill=tk.X, pady=2)
        ttk.Label(skt_row, text="Bone width:").pack(side=tk.LEFT)
        self._sk_thick_var = tk.IntVar(value=self.pose_styles.skeleton_thick)
        skt_lbl = ttk.Label(skt_row, text=str(self.pose_styles.skeleton_thick), width=3)
        skt_lbl.pack(side=tk.RIGHT)
        ttk.Scale(skt_row, from_=1, to=10, variable=self._sk_thick_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: (
                      setattr(self.pose_styles, "skeleton_thick", int(float(v))),
                      skt_lbl.configure(text=str(int(float(v))))
                  )).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        # Confidence gate
        conf_gate_row = ttk.Frame(ctrl)
        conf_gate_row.pack(fill=tk.X, pady=2)
        ttk.Label(conf_gate_row, text="KP conf:").pack(side=tk.LEFT)
        self._kp_conf_var = tk.DoubleVar(value=self.pose_styles.conf_threshold)
        kp_conf_lbl = ttk.Label(conf_gate_row, text=f"{self.pose_styles.conf_threshold:.2f}", width=5)
        kp_conf_lbl.pack(side=tk.RIGHT)
        ttk.Scale(conf_gate_row, from_=0.0, to=1.0, variable=self._kp_conf_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: (
                      setattr(self.pose_styles, "conf_threshold", float(v)),
                      kp_conf_lbl.configure(text=f"{float(v):.2f}")
                  )).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        # ═══════════════════════════════════════
        #  GAME PLUGIN
        # ═══════════════════════════════════════
        section(ctrl, "━━  Game Plugin  ━━")
        hint(ctrl, "Load a .py file with a Game class to play")
        ttk.Label(ctrl, textvariable=self._game_path,
                  wraplength=230, foreground="#cba6f7").pack(anchor="w", pady=(2, 0))

        game_btn_row = ttk.Frame(ctrl)
        game_btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(game_btn_row, text="Browse game…",
                   command=self._browse_game).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(game_btn_row, text="Unload",
                   command=self._unload_game).pack(side=tk.LEFT, padx=(4, 0))

        self._game_status_var = tk.StringVar(value="No game loaded.")
        tk.Label(ctrl, textvariable=self._game_status_var, bg=PANEL_BG,
                 fg="#cba6f7", font=("Segoe UI", 8), wraplength=230,
                 justify=tk.LEFT).pack(anchor="w")

        # ── Stream controls ──
        section(ctrl, "Stream")
        self.start_btn = ttk.Button(ctrl, text="▶  Start Stream",
                                    style="Accent.TButton", command=self._start_stream)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(ctrl, text="■  Stop Stream",
                                   command=self._stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        # ── Label styles ──
        section(ctrl, "Label Styles")
        hint(ctrl, "Select label → pick color & thickness")
        self.label_list_var = tk.StringVar()
        self.label_combo = ttk.Combobox(ctrl, textvariable=self.label_list_var,
                                        state="readonly", width=26)
        self.label_combo.pack(fill=tk.X, pady=2)
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_selected)

        color_row = ttk.Frame(ctrl)
        color_row.pack(fill=tk.X, pady=2)
        ttk.Label(color_row, text="Color:").pack(side=tk.LEFT)
        self.color_preview = tk.Label(color_row, bg=bgr_to_hex(DEFAULT_COLOR_BGR),
                                      width=4, relief="solid", cursor="hand2")
        self.color_preview.pack(side=tk.LEFT, padx=6)
        self.color_preview.bind("<Button-1>", self._pick_color)

        thick_row = ttk.Frame(ctrl)
        thick_row.pack(fill=tk.X, pady=2)
        ttk.Label(thick_row, text="Thickness:").pack(side=tk.LEFT)
        self.thick_var = tk.IntVar(value=DEFAULT_THICKNESS)
        ttk.Spinbox(thick_row, from_=1, to=10, textvariable=self.thick_var,
                    width=5, command=self._save_label_style).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Apply Style",
                   command=self._save_label_style).pack(fill=tk.X)

        # ── Stats ──
        section(ctrl, "Performance stats")
        self.fps_var = tk.StringVar(value="Display: — fps\nInfer:   — fps\nLatency: — ms\nTracks:  —")
        tk.Label(ctrl, textvariable=self.fps_var, bg=PANEL_BG, fg="#a6e3a1",
                 font=("Consolas", 10), justify=tk.LEFT).pack(anchor="w")

        # ── Status ──
        self.status_var = tk.StringVar(value="Ready — load a model and select a camera.")
        tk.Label(ctrl, textvariable=self.status_var, bg=PANEL_BG, fg="#f38ba8",
                 wraplength=240, justify=tk.LEFT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(12, 0))

        # ── CPU tips ──
        section(ctrl, "CPU tips")
        tips = (
            "• Use ONNX backend if available\n"
            "• imgsz 320 is the sweet spot\n"
            "• Frame skip 2–3 feels smooth\n"
            "• Tracking hides skip jitter\n"
            "• Try yolov8s or yolov8n models\n"
            "• Lower camera res reduces resize"
        )
        tk.Label(ctrl, text=tips, bg=PANEL_BG, fg="#6c7086",
                 font=("Segoe UI", 8), justify=tk.LEFT).pack(anchor="w")

        # ── Canvas ──
        canvas_frame = ttk.Frame(self, padding=PAD)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=PAD, padx=PAD)
        self.canvas = tk.Canvas(canvas_frame, bg="#11111b", highlightthickness=0,
                                width=854, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────
    #  Pose colour pickers
    # ─────────────────────────────────────────

    def _pick_kp_color(self, _event=None):
        result = colorchooser.askcolor(
            color=bgr_to_hex(self.pose_styles.keypoint_color),
            title="Keypoint dot colour")
        if result and result[1]:
            self.pose_styles.keypoint_color = hex_to_bgr(result[1])
            self._kp_color_swatch.configure(bg=result[1])

    def _pick_sk_color(self, _event=None):
        result = colorchooser.askcolor(
            color=bgr_to_hex(self.pose_styles.skeleton_color),
            title="Skeleton line colour")
        if result and result[1]:
            self.pose_styles.skeleton_color = hex_to_bgr(result[1])
            self._sk_color_swatch.configure(bg=result[1])

    # ─────────────────────────────────────────
    #  Game plugin
    # ─────────────────────────────────────────

    def _browse_game(self):
        path = filedialog.askopenfilename(
            title="Select game plugin (.py)",
            filetypes=[("Python script", "*.py"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            game = load_game_plugin(path)
            with self._game_lock:
                if self._game is not None:
                    try:
                        self._game.on_stop()
                    except Exception:
                        pass
                self._game = game
                if self.streaming:
                    cw = self.canvas.winfo_width()
                    ch = self.canvas.winfo_height()
                    self._game.on_start(cw if cw > 1 else 854, ch if ch > 1 else 480)
            short = path.split("/")[-1].split("\\")[-1]
            self._game_path.set(short)
            self._game_status_var.set(f"✓ Loaded: {short}")
        except Exception as exc:
            messagebox.showerror("Game load error", str(exc))
            self._game_status_var.set(f"Error: {exc}")

    def _unload_game(self):
        with self._game_lock:
            if self._game is not None:
                try:
                    self._game.on_stop()
                except Exception:
                    pass
                self._game = None
        self._game_path.set("No game loaded")
        self._game_status_var.set("No game loaded.")

    # ─────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────

    def _refresh_cameras(self):
        self.status_var.set("Scanning cameras…")
        self.update_idletasks()
        self.cameras = find_cameras()
        labels = [c["label"] for c in self.cameras]
        self.cam_combo["values"] = labels
        if labels:
            self.cam_combo.current(0)
            self.status_var.set(f"Found {len(labels)} camera(s).")
        else:
            self.status_var.set("No cameras found. Check connections.")

    # ─────────────────────────────────────────
    #  Model
    # ─────────────────────────────────────────

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[
                ("YOLO models", "*.pt *.onnx"),
                ("PyTorch model", "*.pt"),
                ("ONNX model", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._load_model(path)

    def _load_model(self, path: str):
        try:
            self.status_var.set("Loading model…")
            self.update_idletasks()

            backend = self.backend_var.get()
            # Keep the original .pt filename for display purposes
            display_name = path.split("/")[-1].split("\\")[-1]
            original_names = None   # will be populated from .pt before export

            if path.endswith(".pt") and backend in ("onnx", "openvino"):
                self.status_var.set(f"Exporting to {backend.upper()}… (one-time, please wait)")
                self.update_idletasks()
                tmp = YOLO(path)
                original_names = tmp.names   # save names before export
                export_fmt = "onnx" if backend == "onnx" else "openvino"
                exported_path = tmp.export(format=export_fmt, imgsz=self.imgsz_var.get())
                path = str(exported_path)
                self.status_var.set(f"Exported to {backend.upper()}: {path}")
                self.update_idletasks()

            self.model = YOLO(path)
            self._loaded_pt_path = path

            # Restore names from original .pt if ONNX export lost them
            if original_names is not None and not self.model.names:
                self.model.names = original_names

            self.status_var.set("Warming up model…")
            self.update_idletasks()
            dummy = np.zeros((self.imgsz_var.get(), self.imgsz_var.get(), 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device="cpu")

            self.model_path_str.set(display_name)
            self.status_var.set(
                f"Model ready — {len(self.model.names)} classes  [{backend.upper()}]"
            )
            self._populate_label_combo()
        except Exception as exc:
            messagebox.showerror("Model error", str(exc))
            self.status_var.set("Failed to load model.")

    def _populate_label_combo(self):
        if self.model is None:
            return
        labels = list(self.model.names.values())
        self.label_combo["values"] = labels
        if labels:
            self.label_combo.current(0)
            self._on_label_selected()

    # ─────────────────────────────────────────
    #  Label styles
    # ─────────────────────────────────────────

    def _on_label_selected(self, _event=None):
        label = self.label_list_var.get()
        s = self.label_styles.get(label)
        self.color_preview.configure(bg=bgr_to_hex(s["color_bgr"]))
        self.thick_var.set(s["thickness"])

    def _pick_color(self, _event=None):
        label = self.label_list_var.get()
        if not label:
            return
        initial = bgr_to_hex(self.label_styles.get(label)["color_bgr"])
        result = colorchooser.askcolor(color=initial, title=f"Color for '{label}'")
        if result and result[1]:
            self.color_preview.configure(bg=result[1])
            self._save_label_style()

    def _save_label_style(self):
        label = self.label_list_var.get()
        if not label:
            return
        self.label_styles.set(label,
                              hex_to_bgr(self.color_preview.cget("bg")),
                              self.thick_var.get())

    # ─────────────────────────────────────────
    #  OpenCV threads
    # ─────────────────────────────────────────

    def _apply_cv_threads(self):
        n = max(1, int(self.cv_threads.get()))
        cv2.setNumThreads(n)

    # ─────────────────────────────────────────
    #  Streaming
    # ─────────────────────────────────────────

    def _parse_resolution(self) -> tuple[int, int]:
        try:
            w, h = self.cam_res_var.get().split("x")
            return int(w), int(h)
        except Exception:
            return 640, 480

    def _start_stream(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a .pt model first.")
            return
        if not self.cameras:
            messagebox.showwarning("No camera", "No cameras detected.")
            return

        cam_index = self.cameras[self.cam_combo.current()]["index"]
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        cam_w, cam_h = self._parse_resolution()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {cam_index}.")
            return

        self._raw_queue     = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._display_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self.tracker.reset()
        self._sync_tracker()
        self._sync_pose_styles()

        # Notify game plugin
        with self._game_lock:
            if self._game is not None:
                try:
                    self._game.on_start(cam_w, cam_h)
                except Exception:
                    pass

        self.streaming = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Streaming…  (Press ■ Stop to end)")
        self._fps_times.clear()
        self._infer_fps_times.clear()
        self._apply_cv_threads()

        threading.Thread(target=self._capture_thread, daemon=True).start()
        threading.Thread(target=self._infer_thread,   daemon=True).start()
        self._render_loop()

    def _stop_stream(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.tracker.reset()

        with self._game_lock:
            if self._game is not None:
                try:
                    self._game.on_stop()
                except Exception:
                    pass

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.canvas.delete("all")
        self.status_var.set("Stream stopped.")

    # ─────────────────────────────────────────
    #  Thread 1 — capture
    # ─────────────────────────────────────────

    def _capture_thread(self):
        while self.streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if self._raw_queue.full():
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    pass
            self._raw_queue.put(frame)

    # ─────────────────────────────────────────
    #  Thread 2 — inference + tracking + pose + game
    # ─────────────────────────────────────────

    def _infer_thread(self):
        frame_count  = 0
        last_boxes: list = []
        last_results = []
        last_scale   = (1.0, 1.0)
        last_infer_ms = 0.0

        while self.streaming:
            try:
                frame = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.mirror_var.get():
                frame = cv2.flip(frame, 1)

            frame_count += 1
            skip      = max(1, int(self.skip_var.get()))
            run_infer = (frame_count % skip == 0)

            orig_h, orig_w = frame.shape[:2]

            if run_infer:
                imgsz = self.imgsz_var.get()

                t0 = time.perf_counter()
                # Pass original frame; YOLO letterboxes internally.
                # Pre-resizing to a square causes keypoints to be None.
                results = self.model(
                    frame,
                    verbose=False,
                    conf=self.confidence.get(),
                    device="cpu",
                    imgsz=imgsz,
                )
                t1 = time.perf_counter()
                last_infer_ms = (t1 - t0) * 1000

                self._infer_fps_times.append(t1)
                if len(self._infer_fps_times) > 30:
                    self._infer_fps_times.pop(0)

                # Coords are in original-frame space when YOLO handles resizing
                sx, sy = 1.0, 1.0

                raw_boxes = []
                for result in results:
                    for box in result.boxes:
                        cls_id     = int(box.cls[0])
                        label      = self.model.names.get(cls_id, str(cls_id))
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        raw_boxes.append((
                            label, conf_score,
                            int(x1), int(y1),
                            int(x2), int(y2),
                        ))

                if self.tracking_enabled.get():
                    last_boxes = self.tracker.update(raw_boxes)
                else:
                    self.tracker.reset()
                    last_boxes = raw_boxes

                last_results = results
                last_scale   = (sx, sy)   # cache alongside results
                # Extract keypoints for game (scaled to orig res)
                self._last_kps = extract_keypoints(
                    results,
                    scale_xy=(sx, sy),
                    conf_threshold=self.pose_styles.conf_threshold,
                )

            # ── Draw on full-res frame ──
            annotated = frame.copy()

            # Bounding boxes
            if self.show_boxes_var.get():
                for (label, conf_score, x1, y1, x2, y2) in last_boxes:
                    s     = self.label_styles.get(label)
                    color = s["color_bgr"]
                    thick = s["thickness"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                    text = f"{label} {conf_score:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Pose keypoints + skeleton
            if last_results and (self.pose_styles.show_keypoints or self.pose_styles.show_skeleton):
                self.pose_renderer.draw(annotated, last_results, scale_xy=last_scale)

            # Latency overlay
            cv2.putText(annotated, f"Infer: {last_infer_ms:.0f}ms",
                        (8, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (137, 180, 250), 1, cv2.LINE_AA)

            # ── Game plugin overlay ──
            with self._game_lock:
                if self._game is not None:
                    try:
                        annotated = self._game.on_frame(
                            annotated, self._last_kps,
                            orig_w, orig_h,
                        )
                    except Exception as exc:
                        cv2.putText(annotated, f"Game error: {exc}",
                                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 0, 255), 1, cv2.LINE_AA)

            n_tracks = len(last_boxes)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if self._display_queue.full():
                try:
                    self._display_queue.get_nowait()
                except queue.Empty:
                    pass
            self._display_queue.put((rgb, n_tracks, last_infer_ms))

    # ─────────────────────────────────────────
    #  Render loop (main thread)
    # ─────────────────────────────────────────

    def _render_loop(self):
        if not self.streaming:
            return

        try:
            frame, n_tracks, last_infer_ms = self._display_queue.get_nowait()
        except queue.Empty:
            frame, n_tracks, last_infer_ms = None, None, 0.0

        if frame is not None:
            now = time.perf_counter()
            self._fps_times.append(now)
            if len(self._fps_times) > 30:
                self._fps_times.pop(0)

            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                img   = Image.fromarray(frame).resize((cw, ch), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                self.canvas.delete("frame")
                self.canvas.create_image(0, 0, anchor="nw", image=photo, tags="frame")
                self.canvas.image = photo

            disp_fps  = 0.0
            infer_fps = 0.0
            if len(self._fps_times) >= 2:
                disp_fps = (len(self._fps_times) - 1) / (
                    self._fps_times[-1] - self._fps_times[0])
            if len(self._infer_fps_times) >= 2:
                infer_fps = (len(self._infer_fps_times) - 1) / (
                    self._infer_fps_times[-1] - self._infer_fps_times[0])

            infer_ms = (1000 / infer_fps) if infer_fps > 0 else last_infer_ms
            self.fps_var.set(
                f"Display: {disp_fps:5.1f} fps\n"
                f"Infer:   {infer_fps:5.1f} fps\n"
                f"Latency: {infer_ms:5.0f} ms\n"
                f"Tracks:  {n_tracks if n_tracks is not None else '—'}"
            )

        self.after(self.RENDER_DELAY_MS, self._render_loop)

    def on_close(self):
        self._stop_stream()
        self.destroy()


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = YoloStreamApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
