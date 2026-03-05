#!/usr/bin/env python3
"""
YOLO Live Stream Viewer — CPU Edition
--------------------------------------
Squeezes maximum FPS out of CPU-only inference via:
  - OpenCV thread tuning
  - Configurable inference size (imgsz)
  - Frame skip (infer every N frames, display all)
  - INT8 quantization option (if model supports it)
  - ONNX runtime option (much faster than PyTorch on CPU)
  - Threaded 3-stage pipeline: capture → infer → render
  - Camera buffer minimization to reduce latency

Requirements:
    pip install onnxruntime          ← optional but recommended for big speed boost
    pip install openvino             ← optional, Intel CPU further boost
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


# ──────────────────────────────────────────────
#  CPU info
# ──────────────────────────────────────────────

def get_cpu_info() -> tuple[str, int]:
    """Returns (cpu_name, logical_core_count)."""
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

    # Try to get a better name on Windows/Linux
    try:
        import subprocess, sys
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True)
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
        self.title("YOLO Live Stream — CPU Edition")
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
        self.skip_var     = tk.IntVar(value=2)          # infer every N frames
        self.cv_threads   = tk.IntVar(value=max(1, CPU_CORES // 2))
        self.cam_res_var  = tk.StringVar(value="640x480")
        self.backend_var  = tk.StringVar(value="pytorch")   # pytorch | onnx | openvino

        # FPS tracking
        self._fps_times: list[float]       = []
        self._infer_fps_times: list[float] = []

        self._build_ui()
        self._refresh_cameras()
        self._apply_cv_threads()   # apply thread count on startup

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
        style.configure("TFrame",      background=PANEL_BG)
        style.configure("TLabel",      background=PANEL_BG, foreground=FG,  font=("Segoe UI", 10))
        style.configure("TButton",     background=BTN_BG,   foreground=FG,  font=("Segoe UI", 10), borderwidth=0)
        style.configure("TScale",      background=PANEL_BG)
        style.configure("TCombobox",   fieldbackground=BTN_BG, background=BTN_BG, foreground=FG)
        style.configure("TRadiobutton",background=PANEL_BG, foreground=FG,  font=("Segoe UI", 9))
        style.configure("TCheckbutton",background=PANEL_BG, foreground=FG)
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

        # ── Scrollable left panel ──────────────
        left_outer = tk.Frame(self, bg=PANEL_BG, width=260)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg=PANEL_BG, highlightthickness=0, width=255)
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
                 font=("Segoe UI", 8, "bold"), wraplength=220,
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
                  wraplength=220, foreground="#a6e3a1").pack(anchor="w")
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
        self.fps_var = tk.StringVar(value="Display: — fps\nInfer:   — fps\nLatency: — ms")
        tk.Label(ctrl, textvariable=self.fps_var, bg=PANEL_BG, fg="#a6e3a1",
                 font=("Consolas", 10), justify=tk.LEFT).pack(anchor="w")

        # ── Status ──
        self.status_var = tk.StringVar(value="Ready — load a model and select a camera.")
        tk.Label(ctrl, textvariable=self.status_var, bg=PANEL_BG, fg="#f38ba8",
                 wraplength=220, justify=tk.LEFT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(12, 0))

        # ── Tips box ──
        section(ctrl, "CPU tips")
        tips = (
            "• Use ONNX backend if available\n"
            "• imgsz 320 is the sweet spot\n"
            "• Frame skip 2–3 feels smooth\n"
            "• Try yolov8s or yolov8n models\n"
            "• Lower camera res reduces\n"
            "  resize overhead"
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

            # Export to ONNX / OpenVINO if needed and .pt was selected
            if path.endswith(".pt") and backend in ("onnx", "openvino"):
                self.status_var.set(f"Exporting to {backend.upper()}… (one-time, please wait)")
                self.update_idletasks()
                tmp = YOLO(path)
                export_fmt = "onnx" if backend == "onnx" else "openvino"
                exported_path = tmp.export(format=export_fmt, imgsz=self.imgsz_var.get())
                path = str(exported_path)
                self.status_var.set(f"Exported to {backend.upper()}: {path}")
                self.update_idletasks()

            self.model = YOLO(path)
            self._loaded_pt_path = path

            # Warmup
            self.status_var.set("Warming up model…")
            self.update_idletasks()
            dummy = np.zeros((self.imgsz_var.get(), self.imgsz_var.get(), 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device="cpu")

            short = path.split("/")[-1].split("\\")[-1]
            self.model_path_str.set(short)
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # low buffer = low latency

        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {cam_index}.")
            return

        self._raw_queue     = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._display_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)

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
    #  Thread 2 — inference
    # ─────────────────────────────────────────

    def _infer_thread(self):
        frame_count = 0
        last_boxes: list = []
        last_infer_ms = 0.0

        while self.streaming:
            try:
                frame = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            skip = max(1, int(self.skip_var.get()))
            run_infer = (frame_count % skip == 0)

            if run_infer:
                imgsz = self.imgsz_var.get()
                infer_frame = cv2.resize(frame, (imgsz, imgsz))

                t0 = time.perf_counter()
                results = self.model(
                    infer_frame,
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

                # Scale boxes back to original frame resolution
                orig_h, orig_w = frame.shape[:2]
                sx = orig_w / imgsz
                sy = orig_h / imgsz

                last_boxes = []
                for result in results:
                    for box in result.boxes:
                        cls_id     = int(box.cls[0])
                        label      = self.model.names.get(cls_id, str(cls_id))
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        last_boxes.append((
                            label, conf_score,
                            int(x1 * sx), int(y1 * sy),
                            int(x2 * sx), int(y2 * sy),
                        ))

            # Draw on full-res frame
            annotated = frame.copy()
            for (label, conf_score, x1, y1, x2, y2) in last_boxes:
                s     = self.label_styles.get(label)
                color = s["color_bgr"]
                thick = s["thickness"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                text = f"{label} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
                cv2.putText(annotated, text, (x1 + 2, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Latency overlay (bottom-left)
            cv2.putText(annotated, f"Infer: {last_infer_ms:.0f}ms",
                        (8, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (137, 180, 250), 1, cv2.LINE_AA)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if self._display_queue.full():
                try:
                    self._display_queue.get_nowait()
                except queue.Empty:
                    pass
            self._display_queue.put(rgb)

    # ─────────────────────────────────────────
    #  Render loop (main thread)
    # ─────────────────────────────────────────

    def _render_loop(self):
        if not self.streaming:
            return

        try:
            frame = self._display_queue.get_nowait()
        except queue.Empty:
            frame = None

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

            # Compute FPS
            disp_fps  = 0.0
            infer_fps = 0.0
            if len(self._fps_times) >= 2:
                disp_fps = (len(self._fps_times) - 1) / (
                    self._fps_times[-1] - self._fps_times[0])
            if len(self._infer_fps_times) >= 2:
                infer_fps = (len(self._infer_fps_times) - 1) / (
                    self._infer_fps_times[-1] - self._infer_fps_times[0])

            infer_ms = (1000 / infer_fps) if infer_fps > 0 else 0
            self.fps_var.set(
                f"Display: {disp_fps:5.1f} fps\n"
                f"Infer:   {infer_fps:5.1f} fps\n"
                f"Latency: {infer_ms:5.0f} ms"
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