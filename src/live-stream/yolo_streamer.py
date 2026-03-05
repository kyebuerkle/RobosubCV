#!/usr/bin/env python3
"""
YOLO Live Stream Viewer
-----------------------
Features:
- Detect and select available cameras (webcam / USB)
- Browse and load any .pt model file
- Live stream with real-time YOLO bounding boxes
- Per-label color and thickness settings (ready to extend in the UI)

"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

#   edit this for thread count (useful without GPUs)
cv2.setNumThreads(4)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def find_cameras(max_index: int = 10) -> list[dict]:
    """Probe camera indices and return those that open successfully."""
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Try to get a friendly backend name
            backend = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
            cameras.append({"index": idx, "label": f"Camera {idx}  [{backend}]"})
            cap.release()
    return cameras


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' → (B, G, R) for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ──────────────────────────────────────────────
#  Per-label style store
# ──────────────────────────────────────────────

DEFAULT_COLOR_BGR = (0, 255, 0)   # green
DEFAULT_THICKNESS = 2


class LabelStyles:
    """Holds color (BGR) and thickness for each class label."""

    def __init__(self):
        self._styles: dict[str, dict] = {}

    def get(self, label: str) -> dict:
        return self._styles.get(label, {
            "color_bgr": DEFAULT_COLOR_BGR,
            "thickness": DEFAULT_THICKNESS,
        })

    def set(self, label: str, color_bgr: tuple, thickness: int):
        self._styles[label] = {"color_bgr": color_bgr, "thickness": thickness}

    def all_labels(self) -> list[str]:
        return list(self._styles.keys())


# ──────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────

class YoloStreamApp(tk.Tk):
    FRAME_DELAY_MS = 16   # ~60 fps cap on UI side

    def __init__(self):
        super().__init__()
        self.title("YOLO Live Stream Viewer")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        # State
        self.model: YOLO | None = None
        self.model_path = tk.StringVar(value="No model loaded")
        self.cap: cv2.VideoCapture | None = None
        self.streaming = False
        self.stream_thread: threading.Thread | None = None
        self.current_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.label_styles = LabelStyles()
        self.confidence = tk.DoubleVar(value=0.40)
        self.cameras: list[dict] = []

        #   halves the frames calculated
        self.last_boxes = []
        self.frame_count = 0

        self._build_ui()
        self._refresh_cameras()

    # ── UI construction ──────────────────────

    def _build_ui(self):
        PAD = 10
        BG = "#1e1e2e"
        PANEL_BG = "#2a2a3e"
        FG = "#cdd6f4"
        ACCENT = "#89b4fa"
        BTN_BG = "#313244"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TButton", background=BTN_BG, foreground=FG, font=("Segoe UI", 10), borderwidth=0)
        style.map("TButton", background=[("active", ACCENT), ("pressed", "#74c7ec")])
        style.configure("TScale", background=PANEL_BG)
        style.configure("TCombobox", fieldbackground=BTN_BG, background=BTN_BG, foreground=FG)
        style.configure("Accent.TButton", background=ACCENT, foreground="#1e1e2e", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#74c7ec")])

        # ── Left control panel ──
        ctrl = ttk.Frame(self, padding=PAD)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(PAD, 0), pady=PAD)

        # Section header helper
        def section(parent, text):
            ttk.Label(parent, text=text, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))

        # ── Model ──
        section(ctrl, "Model")
        ttk.Label(ctrl, textvariable=self.model_path, wraplength=220,
                foreground="#a6e3a1").pack(anchor="w")
        ttk.Button(ctrl, text="Browse .pt file…", command=self._browse_model).pack(fill=tk.X, pady=4)

        # ── Camera ──
        section(ctrl, "Camera")
        cam_row = ttk.Frame(ctrl)
        cam_row.pack(fill=tk.X)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_var,
                                    state="readonly", width=22)
        self.cam_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cam_row, text="↻", width=3, command=self._refresh_cameras).pack(side=tk.LEFT, padx=(4, 0))

        # ── Confidence ──
        section(ctrl, "Confidence threshold")
        conf_row = ttk.Frame(ctrl)
        conf_row.pack(fill=tk.X)
        self.conf_label = ttk.Label(conf_row, text=f"{self.confidence.get():.2f}", width=5)
        self.conf_label.pack(side=tk.RIGHT)
        ttk.Scale(conf_row, from_=0.05, to=0.95, variable=self.confidence,
                orient=tk.HORIZONTAL, command=self._update_conf_label).pack(
                    side=tk.LEFT, expand=True, fill=tk.X)

        # ── Stream controls ──
        section(ctrl, "Stream")
        self.start_btn = ttk.Button(ctrl, text="▶  Start Stream",
                                    style="Accent.TButton", command=self._start_stream)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(ctrl, text="■  Stop Stream", command=self._stop_stream,
                                state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        # ── Label style editor ──
        section(ctrl, "Label Styles")
        ttk.Label(ctrl, text="Select a label to edit its style:",
                foreground="#bac2de").pack(anchor="w")
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

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready — load a model and select a camera.")
        tk.Label(ctrl, textvariable=self.status_var, bg=PANEL_BG, fg="#f38ba8",
                wraplength=230, justify=tk.LEFT, font=("Segoe UI", 9)).pack(
                    anchor="w", pady=(16, 0))

        # ── Right: video canvas ──
        canvas_frame = ttk.Frame(self, padding=PAD)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=PAD, padx=PAD)

        self.canvas = tk.Canvas(canvas_frame, bg="#11111b", highlightthickness=0,
                                width=854, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── FPS label overlay ──
        self._fps_text = self.canvas.create_text(
            10, 10, anchor="nw", fill="#a6e3a1",
            font=("Consolas", 11, "bold"), text="")

    # ── Camera helpers ──────────────────────

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

    # ── Model helpers ───────────────────────

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
            self.model_path.set(path.split("/")[-1])
            self.status_var.set(f"Model loaded: {len(self.model.names)} classes.")
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

    # ── Label style helpers ─────────────────

    def _on_label_selected(self, _event=None):
        label = self.label_list_var.get()
        style = self.label_styles.get(label)
        self.color_preview.configure(bg=bgr_to_hex(style["color_bgr"]))
        self.thick_var.set(style["thickness"])

    def _pick_color(self, _event=None):
        label = self.label_list_var.get()
        if not label:
            return
        style = self.label_styles.get(label)
        initial = bgr_to_hex(style["color_bgr"])
        result = colorchooser.askcolor(color=initial, title=f"Color for '{label}'")
        if result and result[1]:
            hex_c = result[1]
            self.color_preview.configure(bg=hex_c)
            self._save_label_style()

    def _save_label_style(self):
        label = self.label_list_var.get()
        if not label:
            return
        hex_c = self.color_preview.cget("bg")
        bgr = hex_to_bgr(hex_c)
        self.label_styles.set(label, bgr, self.thick_var.get())

    # ── Streaming ───────────────────────────

    def _start_stream(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a .pt model first.")
            return
        if not self.cameras:
            messagebox.showwarning("No camera", "No cameras detected.")
            return

        # Get selected camera index
        sel_idx = self.cam_combo.current()
        cam_index = self.cameras[sel_idx]["index"]

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {cam_index}.")
            return

        self.streaming = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Streaming…  (Press ■ Stop to end)")

        self.stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stream_thread.start()
        self._render_loop()

    def _stop_stream(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.canvas.delete("all")
        self._fps_text = self.canvas.create_text(
            10, 10, anchor="nw", fill="#a6e3a1", font=("Consolas", 11, "bold"), text="")
        self.status_var.set("Stream stopped.")

    # ── Capture thread (reads + infers) ─────

    def _capture_loop(self):
        import time
        prev_time = time.time()
        while self.streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            #   Making frame smaller to reduce frames (this is set for YOLO anyways)
            frame = cv2.resize(frame, (640, 480))  # or even (416, 416)

            #   halves the frames calculated
            self.frame_count += 1
            run_inference = (self.frame_count % 2 == 0)

            if run_inference:
                results = self.model(frame, verbose=False, conf=self.confidence.get(), imgsz=320)
                self.last_boxes = []
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names.get(cls_id, str(cls_id))
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        self.last_boxes.append((label, conf_score, x1, y1, x2, y2))

            # Always draw last known boxes
            for (label, conf_score, x1, y1, x2, y2) in self.last_boxes:
                style = self.label_styles.get(label)
                color = style["color_bgr"]
                thick = style["thickness"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                text = f"{label} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Compute FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (166, 227, 161), 2, cv2.LINE_AA)

            with self.frame_lock:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Render loop (Tk main thread) ─────────

    def _render_loop(self):
        if not self.streaming:
            return

        with self.frame_lock:
            frame = self.current_frame

        if frame is not None:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                img = Image.fromarray(frame)
                img = img.resize((cw, ch), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                self.canvas.delete("frame")
                self.canvas.create_image(0, 0, anchor="nw", image=photo, tags="frame")
                self.canvas.image = photo   # keep reference

        self.after(self.FRAME_DELAY_MS, self._render_loop)

    # ── Misc ────────────────────────────────

    def _update_conf_label(self, _val=None):
        self.conf_label.configure(text=f"{self.confidence.get():.2f}")

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