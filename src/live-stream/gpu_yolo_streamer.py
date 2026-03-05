#!/usr/bin/env python3
"""
YOLO Live Stream Viewer — GPU Edition
--------------------------------------
Targets ~30 FPS using:
  - CUDA (NVIDIA) or DirectML (AMD/Intel on Windows) acceleration
  - TensorRT export option for NVIDIA (uncomment to use)
  - FP16 half-precision inference
  - Threaded capture pipeline with a frame queue
  - Inference runs at reduced imgsz=640, display at full res
  - Frame-skip ratio configurable from UI

GPU Setup:
  NVIDIA  → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  AMD/Intel (Windows) → pip install torch-directml   (experimental, CPU fallback if unavailable)

Requirements:
    pip install ultralytics opencv-python pillow
    + one of the GPU packages above
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
#  GPU / device detection
# ──────────────────────────────────────────────

def detect_device() -> tuple[str, str]:
    """
    Returns (device_string_for_yolo, human_label).
    Priority: CUDA > DirectML > CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda:0", f"NVIDIA CUDA — {name}"
    except ImportError:
        pass

    try:
        import torch_directml          # AMD / Intel on Windows
        dml_device = torch_directml.device()
        return str(dml_device), "DirectML (AMD/Intel)"
    except (ImportError, Exception):
        pass

    return "cpu", "CPU (no GPU found)"


DEVICE, DEVICE_LABEL = detect_device()


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def find_cameras(max_index: int = 10) -> list[dict]:
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
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
    RENDER_DELAY_MS = 16          # UI refresh (~60 fps cap)
    FRAME_QUEUE_SIZE = 2          # keep pipeline lean to reduce latency

    def __init__(self):
        super().__init__()
        self.title("YOLO Live Stream — GPU Edition")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        # State
        self.model: YOLO | None = None
        self.model_path = tk.StringVar(value="No model loaded")
        self.cap: cv2.VideoCapture | None = None
        self.streaming = False
        self.cameras: list[dict] = []
        self.label_styles = LabelStyles()

        # Pipeline queues
        self._raw_queue: queue.Queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._display_queue: queue.Queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)

        # Settings vars
        self.confidence = tk.DoubleVar(value=0.40)
        self.imgsz_var = tk.IntVar(value=640)
        self.skip_var = tk.IntVar(value=1)    # infer every N frames (1 = every frame)
        self.use_half = tk.BooleanVar(value=DEVICE.startswith("cuda"))

        # FPS tracking
        self._fps_times: list[float] = []
        self._infer_fps_times: list[float] = []

        self._build_ui()
        self._refresh_cameras()

    # ── UI ──────────────────────────────────

    def _build_ui(self):
        PAD = 10
        PANEL_BG = "#2a2a3e"
        FG = "#cdd6f4"
        ACCENT = "#89b4fa"
        BTN_BG = "#313244"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TButton", background=BTN_BG, foreground=FG, font=("Segoe UI", 10), borderwidth=0)
        style.map("TButton", background=[("active", ACCENT)])
        style.configure("TScale", background=PANEL_BG)
        style.configure("TCombobox", fieldbackground=BTN_BG, background=BTN_BG, foreground=FG)
        style.configure("Accent.TButton", background=ACCENT, foreground="#1e1e2e",
                        font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#74c7ec")])
        style.configure("TCheckbutton", background=PANEL_BG, foreground=FG)

        def section(parent, text):
            ttk.Label(parent, text=text, foreground=ACCENT,
                      font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))

        # ── Control panel ──
        ctrl = ttk.Frame(self, padding=PAD)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(PAD, 0), pady=PAD)

        # Device badge
        dev_color = "#a6e3a1" if "CUDA" in DEVICE_LABEL else (
                    "#fab387" if "DirectML" in DEVICE_LABEL else "#f38ba8")
        tk.Label(ctrl, text=f"⚡ {DEVICE_LABEL}", bg=PANEL_BG, fg=dev_color,
                 font=("Segoe UI", 9, "bold"), wraplength=230).pack(anchor="w", pady=(0, 4))

        # Model
        section(ctrl, "Model")
        ttk.Label(ctrl, textvariable=self.model_path, wraplength=220,
                  foreground="#a6e3a1").pack(anchor="w")
        ttk.Button(ctrl, text="Browse .pt file…", command=self._browse_model).pack(fill=tk.X, pady=4)

        # Camera
        section(ctrl, "Camera")
        cam_row = ttk.Frame(ctrl)
        cam_row.pack(fill=tk.X)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_var,
                                      state="readonly", width=22)
        self.cam_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cam_row, text="↻", width=3, command=self._refresh_cameras).pack(side=tk.LEFT, padx=(4, 0))

        # Confidence
        section(ctrl, "Confidence")
        conf_row = ttk.Frame(ctrl)
        conf_row.pack(fill=tk.X)
        self.conf_label = ttk.Label(conf_row, text=f"{self.confidence.get():.2f}", width=5)
        self.conf_label.pack(side=tk.RIGHT)
        ttk.Scale(conf_row, from_=0.05, to=0.95, variable=self.confidence,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.conf_label.configure(
                      text=f"{float(v):.2f}")).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Inference size
        section(ctrl, "Inference size (imgsz)")
        ttk.Label(ctrl, text="Lower = faster, less accurate",
                  foreground="#bac2de", font=("Segoe UI", 8)).pack(anchor="w")
        imgsz_row = ttk.Frame(ctrl)
        imgsz_row.pack(fill=tk.X, pady=2)
        for sz in (320, 480, 640):
            ttk.Radiobutton(imgsz_row, text=str(sz), variable=self.imgsz_var,
                            value=sz).pack(side=tk.LEFT, padx=4)

        # Frame skip
        section(ctrl, "Infer every N frames")
        ttk.Label(ctrl, text="Higher = faster UI, boxes update less often",
                  foreground="#bac2de", font=("Segoe UI", 8)).pack(anchor="w")
        skip_row = ttk.Frame(ctrl)
        skip_row.pack(fill=tk.X, pady=2)
        ttk.Spinbox(skip_row, from_=1, to=8, textvariable=self.skip_var, width=4).pack(side=tk.LEFT)

        # Half precision
        section(ctrl, "Performance")
        ttk.Checkbutton(ctrl, text="FP16 half precision (NVIDIA only)",
                        variable=self.use_half).pack(anchor="w")

        # Stream controls
        section(ctrl, "Stream")
        self.start_btn = ttk.Button(ctrl, text="▶  Start Stream",
                                    style="Accent.TButton", command=self._start_stream)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(ctrl, text="■  Stop Stream",
                                   command=self._stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        # Label styles
        section(ctrl, "Label Styles")
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
        ttk.Button(ctrl, text="Apply Style", command=self._save_label_style).pack(fill=tk.X)

        # FPS display
        section(ctrl, "Performance stats")
        self.fps_var = tk.StringVar(value="Display: — fps\nInfer:   — fps")
        tk.Label(ctrl, textvariable=self.fps_var, bg=PANEL_BG, fg="#a6e3a1",
                 font=("Consolas", 10), justify=tk.LEFT).pack(anchor="w")

        # Status
        self.status_var = tk.StringVar(value="Ready — load a model and select a camera.")
        tk.Label(ctrl, textvariable=self.status_var, bg=PANEL_BG, fg="#f38ba8",
                 wraplength=230, justify=tk.LEFT, font=("Segoe UI", 9)).pack(
                     anchor="w", pady=(10, 0))

        # ── Canvas ──
        canvas_frame = ttk.Frame(self, padding=PAD)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=PAD, padx=PAD)
        self.canvas = tk.Canvas(canvas_frame, bg="#11111b", highlightthickness=0,
                                width=854, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ── Cameras ──────────────────────────────

    def _refresh_cameras(self):
        self.status_var.set("Scanning cameras…")
        self.update_idletasks()
        self.cameras = find_cameras()
        labels = [c["label"] for c in self.cameras]
        self.cam_combo["values"] = labels
        if labels:
            self.cam_combo.current(0)
            self.status_var.set(f"Found {len(labels)} camera(s). Device: {DEVICE_LABEL}")
        else:
            self.status_var.set("No cameras found.")

    # ── Model ────────────────────────────────

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO .pt model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.status_var.set("Loading model…")
            self.update_idletasks()
            self.model = YOLO(path)

            # Warm up on the target device
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, device=DEVICE, verbose=False,
                       half=self.use_half.get() and DEVICE.startswith("cuda"))

            self.model_path.set(path.split("/")[-1].split("\\")[-1])
            self.status_var.set(f"Model ready — {len(self.model.names)} classes on {DEVICE_LABEL}")
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

    # ── Label styles ─────────────────────────

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
        self.label_styles.set(label, hex_to_bgr(self.color_preview.cget("bg")), self.thick_var.get())

    # ── Streaming ────────────────────────────

    def _start_stream(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a .pt model first.")
            return
        if not self.cameras:
            messagebox.showwarning("No camera", "No cameras detected.")
            return

        cam_index = self.cameras[self.cam_combo.current()]["index"]
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        # Request higher FPS from the camera driver
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize buffer lag

        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {cam_index}.")
            return

        # Clear queues
        self._raw_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._display_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)

        self.streaming = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Streaming…")
        self._fps_times.clear()
        self._infer_fps_times.clear()

        # Two background threads: capture → infer → display queue
        threading.Thread(target=self._capture_thread, daemon=True).start()
        threading.Thread(target=self._infer_thread, daemon=True).start()
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

    # ── Thread 1: capture frames ─────────────

    def _capture_thread(self):
        """Reads camera frames as fast as possible and pushes to raw queue."""
        while self.streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Drop oldest if queue is full (keeps latency low)
            if self._raw_queue.full():
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    pass
            self._raw_queue.put(frame)

    # ── Thread 2: inference ──────────────────

    def _infer_thread(self):
        """Pulls raw frames, runs YOLO, pushes annotated frames to display queue."""
        frame_count = 0
        last_boxes: list = []
        use_half = self.use_half.get() and DEVICE.startswith("cuda")

        while self.streaming:
            try:
                frame = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            run_infer = (frame_count % max(1, self.skip_var.get()) == 0)

            # Resize for faster inference
            infer_frame = cv2.resize(frame, (self.imgsz_var.get(), self.imgsz_var.get()))

            if run_infer:
                t0 = time.perf_counter()
                results = self.model(
                    infer_frame,
                    verbose=False,
                    conf=self.confidence.get(),
                    device=DEVICE,
                    half=use_half,
                    imgsz=self.imgsz_var.get(),
                )
                t1 = time.perf_counter()
                self._infer_fps_times.append(t1)
                if len(self._infer_fps_times) > 30:
                    self._infer_fps_times.pop(0)

                # Scale boxes back to original frame size
                orig_h, orig_w = frame.shape[:2]
                infer_h, infer_w = infer_frame.shape[:2]
                sx = orig_w / infer_w
                sy = orig_h / infer_h

                last_boxes = []
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names.get(cls_id, str(cls_id))
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
                s = self.label_styles.get(label)
                color, thick = s["color_bgr"], s["thickness"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                text = f"{label} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
                cv2.putText(annotated, text, (x1 + 2, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Push to display queue (drop oldest if full)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if self._display_queue.full():
                try:
                    self._display_queue.get_nowait()
                except queue.Empty:
                    pass
            self._display_queue.put(rgb)

    # ── Render loop (main thread) ────────────

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
                img = Image.fromarray(frame).resize((cw, ch), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                self.canvas.delete("frame")
                self.canvas.create_image(0, 0, anchor="nw", image=photo, tags="frame")
                self.canvas.image = photo

            # Update FPS stats
            if len(self._fps_times) >= 2:
                disp_fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
            else:
                disp_fps = 0.0
            if len(self._infer_fps_times) >= 2:
                infer_fps = (len(self._infer_fps_times) - 1) / (
                    self._infer_fps_times[-1] - self._infer_fps_times[0])
            else:
                infer_fps = 0.0
            self.fps_var.set(f"Display: {disp_fps:5.1f} fps\nInfer:   {infer_fps:5.1f} fps")

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
