"""
evaluate_yolo_labels.py
-----------------------
Companion to evaluate_yolo.py.

What it does
------------
1. Calls evaluate_yolo.py as a subprocess so all the normal evaluation
   outputs (yolo_evaluation_results.csv, per_class_ap.csv,
   all_object_results.csv) are produced exactly as before.

2. For every model × split combination, re-runs model.val() with
   save_txt=True so YOLO writes one prediction .txt per image under
      <output_dir>/runs/labels_<model>_<split>/labels/
   Each .txt line is YOLO format:
      class_id  cx  cy  w  h  [conf]   (all normalised 0-1)

3. Reads the ground-truth .txt files from the dataset split (located via
   the data YAML) using the same YOLO normalised format.

4. Matches every predicted box to its nearest GT box (same class, highest
   IoU) and produces a CSV:

   file_name, object_name, class_id,
   width_px, height_px,          ← predicted box in pixels
   gt_width_px, gt_height_px,    ← best-matching GT box in pixels
   iou50,                        ← IoU@0.50 (1.0 if TP, 0.0 if FP)
   % area,                       ← predicted box % area vs res=1.0 baseline
   gt % area                     ← GT box % area vs res=1.0 baseline

   % area is calculated by comparing the predicted/GT box against the
   corresponding res=1.0 (baseline) label for the same image.  If an
   object is cropped by an upscale augmentation the fraction that remains
   in frame is reported; unaffected labels report 100 %.

Requirements
------------
    pip install ultralytics==8.4.14 pandas numpy pyyaml

Usage
-----
    # mirrors evaluate_yolo.py flags — pass the same arguments
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml

    # % area columns are computed automatically when data.yaml contains
    # an 'augmentations' block (written by dataset_config.py).
    # For datasets augmented before that block existed, pass --resize directly:
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml \
        --resize 0.25,1.0,1.5,2.0

    # Override with --config if you have a standalone configuration.json
    # (only useful if resize list is non-empty in that file):
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml \
        --config configuration.json

    # manual confidence sweep
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml --conf 0.25,0.50

    # limit to specific splits
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml --splits val test
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import builtins as _builtins
builtins_print = _builtins.print


# ─────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING  (mirrors evaluate_yolo.py exactly)
# ─────────────────────────────────────────────────────────────────

def parse_float_list(s: str) -> list[float]:
    """Parse a comma-separated string of floats: '0.25,0.5' → [0.25, 0.5]"""
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got: {s!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wrapper around evaluate_yolo.py that also saves prediction "
                    "label .txt files and builds a per-label CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m", "--models", nargs="+", required=True, metavar="PATH")
    p.add_argument("-d", "--data",   required=True,            metavar="PATH")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])

    cg = p.add_mutually_exclusive_group()
    cg.add_argument("--find-best-conf", action="store_true", default=True)
    cg.add_argument("-c", "--conf", default=None, type=parse_float_list,
                    metavar="LIST",
                    help="Comma-separated confidence thresholds. Disables auto-F1.")

    p.add_argument("--discovery-conf", type=float, default=0.001)
    p.add_argument("--iou",   type=float, default=0.5)
    p.add_argument("-o", "--output", default="output", metavar="DIR")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")

    # Path to evaluate_yolo.py — defaults to same directory as this script
    p.add_argument("--eval-script", default=None, metavar="PATH",
                   help="Path to evaluate_yolo.py. Defaults to the same "
                        "directory as this script.")

    # Optional augmentation config override for % area calculation.
    # If omitted, augmentation parameters are read from the 'augmentations'
    # block in data.yaml (written automatically by dataset_config.py).
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Path to configuration.json. Overrides the "
                        "'augmentations' block in data.yaml if both exist.")
    p.add_argument("--resize", default=None, type=parse_float_list,
                   metavar="LIST",
                   help="Comma-separated resize values used during augmentation, "
                        "e.g. 0.25,1.0,1.5,2.0. Overrides both data.yaml and "
                        "--config for the resize list. Use this for datasets "
                        "augmented before augmentation metadata was saved to "
                        "data.yaml.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION CONFIG LOADING
# ─────────────────────────────────────────────────────────────────

def load_augmentation_config(config_path: str) -> dict:
    """
    Load a standalone configuration.json (manual override).

    Returns empty dict if path is None or file cannot be parsed.
    """
    if config_path is None:
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"[config] Loaded augmentation config from {config_path}")
        return cfg
    except Exception as e:
        print(f"[warn] Could not load augmentation config {config_path}: {e}")
        return {}


def load_augmentation_config_from_yaml(data_yaml_path: str) -> dict:
    """
    Read the 'augmentations' block written by dataset_config.py into data.yaml.

    Expected YAML block (written by dataset_config._save_augmentations_to_yaml):
        augmentations:
          order: [exp, con, res, mblur]
          naming_convention: "{file}_exp{ind}_con{ind}_res{ind}_{idx}{ext}"
          exposure:      [0.615, 1.0, 1.385]
          contrast:      [0.7, 1.0, 1.3]
          resize:        [0.25, 1.0, 1.5, 2.0]
          motion_blur:   [1.0, 1.5]

    NOTE: the new augment_strategy embeds float VALUES directly in filenames
    (e.g. _res1.5) rather than integer indices, so resize_baseline_index
    is no longer needed for % area calculation.

    Returns the augmentations sub-dict, or empty dict if not present.
    """
    try:
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        aug = data.get("augmentations", {})
        if aug:
            print(f"[config] Loaded augmentation metadata from data.yaml")
            print(f"  order:             {aug.get('order')}")
            print(f"  naming_convention: {aug.get('naming_convention')}")
            print(f"  resize:            {aug.get('resize')}  "
                  f"(baseline index: {aug.get('resize_baseline_index')})")
        else:
            print("[config] No 'augmentations' block found in data.yaml — "
                  "'% area' columns will be empty.")
        return aug
    except Exception as e:
        print(f"[warn] Could not read augmentation metadata from data.yaml: {e}")
        return {}


# build_resize_index_map / get_baseline_res_index are no longer needed.
# The new augment_strategy embeds resize VALUES directly in filenames
# (e.g. _res1.5) so the scale is read straight from the stem — no index
# map or baseline-index lookup required.


# ─────────────────────────────────────────────────────────────────
#  FILENAME AUGMENTATION PARSING
# ─────────────────────────────────────────────────────────────────
#
#  augment_strategy.py produces filenames in the format:
#
#    {original_stem}_{name0}{val0}_{name1}{val1}_..._{combo_idx}[_r{repeat}]{ext}
#
#  e.g.  img001_exp0.615_con0.7_res1.5_0.jpg
#        widths_png_exp0.7_con0.8_2_r1.jpg   (wrapped repeat)
#        widths_png.jpg                       (original copy, no tokens)
#
#  Key differences from the old index-based format (_sat0_exp1_res2):
#    - float values embedded directly, not integer indices
#    - a single trailing combo index, not one index per aug type
#    - the original copy has NO augmentation tokens at all
#
#  The baseline for % area is always the ORIGINAL image (the copy that
#  augment_strategy writes alongside the augmented images), whose label
#  file is simply {original_stem}.txt.
#
#  The resize scale is read directly from the _res token value — no
#  index → value mapping needed.

# Known augmentation token names (from augment_strategy._AUG_FUNCS)
_KNOWN_AUG_NAMES = {"exp", "con", "res", "mblur", "gblur", "hue", "sat"}

# Matches a single augmentation token: _name{float_or_int}
_AUG_VALUE_RE = re.compile(
    r'_(%s)([\.\d]+)' % '|'.join(sorted(_KNOWN_AUG_NAMES, key=len, reverse=True))
)
# Trailing combo index:  _N  or  _N_rM
_COMBO_INDEX_RE = re.compile(r'_(\d+)(?:_r(\d+))?$')


def parse_strategy_stem(stem: str) -> tuple[str, dict[str, float]]:
    """
    Parse a stem produced by augment_strategy._combo_filename and return
    (original_stem, {aug_name: float_value}).

    original_stem is the source image stem BEFORE any augmentation tokens —
    this matches the filename of the original copy in the output directory,
    and therefore the basename of the GT label file for that image.

    Examples
    --------
    "img001_exp0.615_con0.7_res1.5_0"
        → ("img001", {"exp": 0.615, "con": 0.7, "res": 1.5})

    "widths_png_exp0.7_con0.8_2_r1"
        → ("widths_png", {"exp": 0.7, "con": 0.8})

    "widths_png"   (original copy — no aug tokens)
        → ("widths_png", {})
    """
    s = stem

    # Strip trailing combo index (and optional _r{repeat})
    m = _COMBO_INDEX_RE.search(s)
    if m:
        s = s[:m.start()]

    # Collect all augmentation tokens
    augs: dict[str, float] = {}
    for m in _AUG_VALUE_RE.finditer(s):
        augs[m.group(1)] = float(m.group(2))

    # original_stem = everything before the first aug token
    first = _AUG_VALUE_RE.search(s)
    original_stem = s[:first.start()] if first else s

    return original_stem, augs


# ─────────────────────────────────────────────────────────────────
#  % AREA CALCULATION
# ─────────────────────────────────────────────────────────────────

def compute_percent_area(
    baseline_cx_n: float, baseline_cy_n: float,
    baseline_w_n: float, baseline_h_n: float,
    resize_scale: float,
) -> float:
    """
    Compute the fraction of an object that remains visible after an upscale
    augmentation, using the same affine+clamp logic as yolo_scale_label().

    The augmentation scales from the image centre (origin = 0.5, 0.5 in
    normalised coords), which is the default in yolo_scale_label.

    Steps
    -----
    1. Apply the affine shift to the baseline box centre:
           cx_new = scale * (cx - 0.5) + 0.5
           cy_new = scale * (cy - 0.5) + 0.5
           w_new  = scale * w
           h_new  = scale * h
    2. Compute unclamped area = w_new * h_new  (what the box WOULD be)
    3. Clamp edges to [0, 1] and compute clamped area
    4. fraction_visible = clamped_area / unclamped_area

    Returns
    -------
    Float in [0.0, 1.0].  Returns 1.0 if resize_scale <= 1.0.
    """
    if resize_scale <= 1.0:
        return 1.0

    ox_n, oy_n = 0.5, 0.5   # centre origin (normalised)

    cx_new = resize_scale * (baseline_cx_n - ox_n) + ox_n
    cy_new = resize_scale * (baseline_cy_n - oy_n) + oy_n
    w_new  = resize_scale * baseline_w_n
    h_new  = resize_scale * baseline_h_n

    unclamped_area = w_new * h_new
    if unclamped_area <= 0:
        return 1.0

    x1 = cx_new - w_new / 2;  x2 = cx_new + w_new / 2
    y1 = cy_new - h_new / 2;  y2 = cy_new + h_new / 2

    x1c = max(0.0, x1);  x2c = min(1.0, x2)
    y1c = max(0.0, y1);  y2c = min(1.0, y2)

    clamped_w = max(0.0, x2c - x1c)
    clamped_h = max(0.0, y2c - y1c)
    clamped_area = clamped_w * clamped_h

    return round(min(clamped_area / unclamped_area, 1.0), 6)


# ─────────────────────────────────────────────────────────────────
#  STEP 1 – CALL evaluate_yolo.py
# ─────────────────────────────────────────────────────────────────

def call_evaluate_yolo(args: argparse.Namespace) -> int:
    """Build the subprocess command and run evaluate_yolo.py."""
    eval_script = args.eval_script
    if eval_script is None:
        eval_script = Path(__file__).parent / "test_evaluate_yolo.py"

    eval_script = Path(eval_script)
    if not eval_script.exists():
        print(f"[ERROR] test_evaluate_yolo.py not found at: {eval_script}")
        print("        Use --eval-script to specify its location.")
        sys.exit(1)

    cmd = [
        sys.executable, str(eval_script),
        "-m", *args.models,
        "-d", args.data,
        "--splits", *args.splits,
        "--iou",    str(args.iou),
        "-o",       args.output,
        "--imgsz",  str(args.imgsz),
        "--device", args.device,
        "--discovery-conf", str(args.discovery_conf),
    ]

    if args.conf is not None:
        cmd += ["-c", ",".join(str(c) for c in args.conf)]

    print("=" * 60)
    print("STEP 1 – Running evaluate_yolo.py")
    print("=" * 60)
    print("CMD:", " ".join(cmd))
    print()

    sys.stdout.flush()
    result = subprocess.run(cmd, stderr=None, stdout=None)
    sys.stdout.flush()
    return result.returncode


# ─────────────────────────────────────────────────────────────────
#  STEP 2 – SAVE PREDICTION LABEL TXT FILES
# ─────────────────────────────────────────────────────────────────

def save_prediction_labels(model_path: str, data_yaml: str, split: str,
                            conf: float, iou: float, imgsz: int,
                            device: str, labels_dir: Path) -> Path:
    """
    Run model.val() with save_txt=True.
    Returns the path to the labels sub-directory.
    """
    model      = YOLO(model_path)
    model_name = Path(model_path).stem
    run_name   = f"{split}_{model_name}_conf{conf:.4f}".replace(".", "")

    model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save_txt=True,
        save_conf=True,
        project=str(labels_dir),
        name=run_name,
        exist_ok=True,
    )

    pred_labels_dir = labels_dir / run_name / "labels"
    return pred_labels_dir


# ─────────────────────────────────────────────────────────────────
#  HELPERS – READING LABEL TXT FILES
# ─────────────────────────────────────────────────────────────────

def read_yolo_labels(txt_path: Path) -> np.ndarray:
    """
    Read a YOLO-format label file.
    Returns float array shape (N, 5+): [class_id, cx, cy, w, h, (conf)].
    Returns empty (0,5) array if file is empty or missing.
    """
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=float)

    rows = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts])

    if not rows:
        return np.zeros((0, 5), dtype=float)

    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────
#  HELPERS – DATASET PATHS FROM YAML
# ─────────────────────────────────────────────────────────────────

def get_split_image_dir(data_yaml_path: str, split: str) -> Path | None:
    """
    Parse the dataset YAML and return the image directory for the given split.

    Resolution order:
        1. Absolute path in YAML → use directly
        2. (base_path / raw).resolve()  ← handles Roboflow '../split/images'
           relative to the dataset 'path' key
        3. (base_path.parent / raw).resolve()
        4. (yaml_dir / raw).resolve()   ← plain fallback
    """
    yaml_path = Path(data_yaml_path).resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    key = split if split in cfg else ("valid" if split == "val" else None)
    if key is None or cfg.get(key) is None:
        return None

    raw = cfg[key]
    if isinstance(raw, list):
        raw = raw[0]

    p = Path(raw)

    if p.is_absolute():
        return p.resolve() if p.exists() else None

    base = cfg.get("path")

    candidates = []
    # PRIMARY: relative to the YAML file itself (yaml_path is already resolved)
    candidates.append((yaml_path / p).resolve())
    if base:
        base_path = Path(base)
        # resolve '../split/images' relative to the dataset 'path' dir
        candidates.append((base_path / p).resolve())
        # resolve relative to path's parent
        candidates.append((base_path.parent / p).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def image_dir_to_label_dir(img_dir: Path) -> Path:
    """Swap the 'images' path component for 'labels'."""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except ImportError:
        pass

    suffix = img_path.suffix.lower()
    if suffix == ".png":
        with open(img_path, "rb") as f:
            import struct
            f.seek(16)
            w, h = struct.unpack(">II", f.read(8))
            return w, h

    if suffix in (".jpg", ".jpeg"):
        with open(img_path, "rb") as f:
            data = f.read()
        i = 2
        while i < len(data) - 8:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                import struct
                h, w = struct.unpack(">HH", data[i + 5: i + 9])
                return w, h
            seg_len = int.from_bytes(data[i + 2: i + 4], "big")
            i += 2 + seg_len

    raise RuntimeError(f"Cannot determine image size for: {img_path}")


# ─────────────────────────────────────────────────────────────────
#  HELPERS – BOX IoU
# ─────────────────────────────────────────────────────────────────

def box_iou_single(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> float:
    """IoU of two normalised YOLO boxes (cx, cy, w, h)."""
    x1a, y1a = cx1 - w1 / 2, cy1 - h1 / 2
    x2a, y2a = cx1 + w1 / 2, cy1 + h1 / 2
    x1b, y1b = cx2 - w2 / 2, cy2 - h2 / 2
    x2b, y2b = cx2 + w2 / 2, cy2 + h2 / 2

    inter_x1 = max(x1a, x1b); inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b); inter_y2 = min(y2a, y2b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a  = w1 * h1
    area_b  = w2 * h2
    union   = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
#  % AREA LOOKUP CACHE
# ─────────────────────────────────────────────────────────────────

class AreaCache:
    """
    Looks up the original (pre-augmentation) GT label for any augmented image
    stem and computes what fraction of each GT box remained in frame after the
    resize crop.

    How it works with the new augment_strategy filename format
    ----------------------------------------------------------
    augment_strategy always copies the ORIGINAL image into the output
    directory alongside the augmented images.  Its filename is the unchanged
    source stem (e.g. widths_png.jpg), and its GT label file is therefore
    {original_stem}.txt in the GT labels directory.

    For augmented images the stem encodes aug VALUES directly
    (e.g. widths_png_exp0.7_res1.5_0).  parse_strategy_stem() strips the
    aug tokens and combo index to recover the original_stem, and reads the
    resize scale directly from the _res token — no index→value map needed.

    Usage
    -----
        cache = AreaCache(gt_label_dir)
        pct   = cache.get_percent_area(stem, cls_id, cx, cy, w, h)
    """

    def __init__(self, gt_label_dir: Path):
        self.gt_label_dir = gt_label_dir
        # {original_stem: np.ndarray of GT labels from the original image}
        self._baseline_cache: dict[str, np.ndarray] = {}

    def _load_baseline(self, original_stem: str) -> np.ndarray:
        if original_stem not in self._baseline_cache:
            txt = self.gt_label_dir / f"{original_stem}.txt"
            self._baseline_cache[original_stem] = read_yolo_labels(txt)
        return self._baseline_cache[original_stem]

    def get_percent_area(
        self,
        stem: str,
        cls_id: int,
        cx_n: float, cy_n: float, w_n: float, h_n: float,
    ) -> float | None:
        """
        Compute the fraction of a GT box that remained visible after an
        upscale-and-crop augmentation.

        Returns
        -------
        float   : fraction in [0, 1] expressed as a percentage (×100) —
                  1.0 means the full box was in frame, 0.5 means half was
                  cropped away.
        100.0   : if the image had no resize augmentation, or resize ≤ 1.0
                  (downscale does not crop, it pads/letterboxes).
        None    : if the original GT label cannot be found, or the class
                  has no matching box in the original.

        Algorithm
        ---------
        1. parse_strategy_stem() → original_stem + {aug_name: value}
        2. If no 'res' token → 100.0  (not resize-augmented)
        3. If resize_scale ≤ 1.0 → 100.0  (downscale, no cropping)
        4. Load {original_stem}.txt — the GT labels for the source image
        5. Find the best-matching box of the same class by IoU
        6. compute_percent_area(baseline_box, resize_scale)
        """
        original_stem, augs = parse_strategy_stem(stem)

        if "res" not in augs:
            # Original copy or no resize aug applied — nothing was cropped
            return 100.0

        resize_scale = augs["res"]

        # Downscale pads/letterboxes, never crops
        if resize_scale <= 1.0:
            return 100.0

        baseline_labels = self._load_baseline(original_stem)
        if baseline_labels.shape[0] == 0:
            return None   # original label file not found

        same_cls = baseline_labels[baseline_labels[:, 0].astype(int) == cls_id]
        if same_cls.shape[0] == 0:
            return None   # class not in original labels

        # Match to the baseline box with highest IoU
        best_iou = -1.0
        best_row = None
        for row in same_cls:
            iou = box_iou_single(cx_n, cy_n, w_n, h_n,
                                 row[1], row[2], row[3], row[4])
            if iou > best_iou:
                best_iou = iou
                best_row = row

        if best_row is None:
            return None

        return compute_percent_area(
            best_row[1], best_row[2], best_row[3], best_row[4],
            resize_scale,
        )


# ─────────────────────────────────────────────────────────────────
#  STEP 3 – BUILD PER-LABEL CSV
# ─────────────────────────────────────────────────────────────────

def build_label_csv(args: argparse.Namespace,
                    conf_map: dict[tuple[str, str], float],
                    output_dir: Path,
                    labels_run_dir: Path,
                    aug_cfg: dict) -> Path:
    """
    Walk every model × split, read predicted and GT label files,
    match predictions to GT boxes, and write per_label_results.csv.
    """
    # resize_map / baseline_res_idx no longer needed — the new AreaCache reads
    # the resize scale directly from the filename token (_res{value}).

    class_names_cache: dict[str, dict] = {}
    rows = []

    for model_path in args.models:
        if not Path(model_path).exists():
            continue

        model_name = Path(model_path).name
        model_stem = Path(model_path).stem

        if model_path not in class_names_cache:
            model_obj = YOLO(model_path)
            class_names_cache[model_path] = model_obj.names

        class_names = class_names_cache[model_path]

        for split in args.splits:
            conf = conf_map.get((model_name, split))
            if conf is None:
                print(f"  [warn] No confidence found for {model_name}/{split}, skipping.")
                continue

            # ── Locate predicted labels directory ─────────────────────────
            run_name       = f"{split}_{model_stem}_conf{conf:.4f}".replace(".", "")
            pred_label_dir = labels_run_dir / run_name / "labels"

            if not pred_label_dir.exists():
                print(f"  [warn] Predicted labels dir not found: {pred_label_dir}")
                continue

            # ── Locate GT labels directory ─────────────────────────────────
            img_dir = get_split_image_dir(args.data, split)
            if img_dir is None:
                print(f"  [warn] Cannot find image dir for split '{split}' "
                      f"in {args.data}")
                continue

            gt_label_dir = image_dir_to_label_dir(img_dir)
            if not gt_label_dir.exists():
                print(f"  [warn] GT labels dir not found: {gt_label_dir}")

            # ── Build area cache (uses original GT labels from base stems) ───
            area_cache = AreaCache(gt_label_dir)

            # ── Collect image paths ────────────────────────────────────────
            img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            image_files    = {
                p.stem: p
                for p in img_dir.iterdir()
                if p.suffix.lower() in img_extensions
            }

            print(f"  Building label rows: {model_name} / {split} / conf={conf:.4f} "
                  f"({len(list(pred_label_dir.glob('*.txt')))} pred files, "
                  f"{len(image_files)} images)")

            # ── Iterate over every predicted label file ────────────────────
            for pred_txt in sorted(pred_label_dir.glob("*.txt")):
                stem = pred_txt.stem

                img_path = image_files.get(stem)
                if img_path is None:
                    matches  = [v for k, v in image_files.items()
                                if k.lower() == stem.lower()]
                    img_path = matches[0] if matches else None

                if img_path is not None:
                    try:
                        img_w, img_h = get_image_size(img_path)
                    except Exception as e:
                        print(f"    [warn] Cannot read image size for {img_path}: {e}")
                        img_w = img_h = args.imgsz
                else:
                    img_w = img_h = args.imgsz

                preds = read_yolo_labels(pred_txt)
                gt_txt = gt_label_dir / f"{stem}.txt"
                gt     = read_yolo_labels(gt_txt)

                for pred_row in preds:
                    p_cls        = int(pred_row[0])
                    p_cx, p_cy   = pred_row[1], pred_row[2]
                    p_w_n, p_h_n = pred_row[3], pred_row[4]

                    p_w_px = round(p_w_n * img_w, 2)
                    p_h_px = round(p_h_n * img_h, 2)

                    gt_match_iou = 0.0
                    gt_w_px      = float("nan")
                    gt_h_px      = float("nan")
                    best_gt_row  = None

                    if gt.shape[0] > 0:
                        same_cls_mask = gt[:, 0].astype(int) == p_cls
                        gt_same       = gt[same_cls_mask]

                        best_iou = 0.0
                        for gt_row in gt_same:
                            iou = box_iou_single(
                                p_cx, p_cy, p_w_n, p_h_n,
                                gt_row[1], gt_row[2], gt_row[3], gt_row[4],
                            )
                            if iou > best_iou:
                                best_iou    = iou
                                best_gt_row = gt_row

                        if best_gt_row is not None:
                            gt_match_iou = best_iou
                            gt_w_px      = round(best_gt_row[3] * img_w, 2)
                            gt_h_px      = round(best_gt_row[4] * img_h, 2)

                    iou50 = 1.0 if gt_match_iou >= 0.50 else 0.0

                    # ── % area (predicted box vs baseline) ────────────────
                    pct_area = area_cache.get_percent_area(
                        stem, p_cls, p_cx, p_cy, p_w_n, p_h_n,
                    )

                    # ── gt % area (GT box vs baseline) ────────────────────
                    gt_pct_area = None
                    if best_gt_row is not None:
                        gt_pct_area = area_cache.get_percent_area(
                            stem, p_cls,
                            best_gt_row[1], best_gt_row[2],
                            best_gt_row[3], best_gt_row[4],
                        )

                    rows.append({
                        "model":        model_name,
                        "split":        split,
                        "file_name":    stem,
                        "object_name":  class_names.get(p_cls, f"class_{p_cls}"),
                        "class_id":     p_cls,
                        "width_px":     p_w_px,
                        "height_px":    p_h_px,
                        "gt_width_px":  gt_w_px,
                        "gt_height_px": gt_h_px,
                        "iou50":        round(gt_match_iou, 4),
                        "% area":       pct_area,
                        "gt % area":    gt_pct_area,
                    })

    # ── Write CSV ──────────────────────────────────────────────────────────
    out_csv = output_dir / "per_label_results.csv"
    if rows:
        df   = pd.DataFrame(rows)
        cols = ["model", "split", "file_name", "object_name", "class_id",
                "width_px", "height_px", "gt_width_px", "gt_height_px",
                "iou50", "% area", "gt % area"]
        df[cols].to_csv(out_csv, index=False)
        print(f"\n✅ Per-label CSV → {out_csv}  ({len(df)} rows)")
    else:
        print("\n⚠️  No label rows produced.")

    return out_csv


# ─────────────────────────────────────────────────────────────────
#  LOAD CONFIDENCE MAP FROM SUMMARY CSV
# ─────────────────────────────────────────────────────────────────

def load_conf_map(summary_csv: Path) -> dict[tuple[str, str], float]:
    """Read yolo_evaluation_results.csv → {(model_name, split): confidence}."""
    if not summary_csv.exists():
        return {}
    df     = pd.read_csv(summary_csv)
    result = {}
    for _, row in df.iterrows():
        result[(str(row["model"]), str(row["split"]))] = float(row["confidence"])
    return result


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def _print(*a, **kw):
    """Print with immediate flush so sbatch/slurm captures output even on crash."""
    kw.setdefault("flush", True)
    builtins_print(*a, **kw)


def main():
    args       = parse_args()
    output_dir = Path(args.output).resolve()

    _print(f"[main] evaluate_yolo_labels.py starting")
    _print(f"[main] models:  {args.models}")
    _print(f"[main] data:    {args.data}")
    _print(f"[main] splits:  {args.splits}")
    _print(f"[main] output:  {output_dir}")
    _print(f"[main] resize:  {getattr(args, 'resize', None)}")
    _print(f"[main] device:  {args.device}")

    # ── Load augmentation config ──────────────────────────────────────────
    # Priority: --config (manual override) > augmentations block in data.yaml
    if args.config is not None:
        aug_cfg = load_augmentation_config(args.config)
    else:
        aug_cfg = load_augmentation_config_from_yaml(args.data)

    # --resize flag overrides the resize list from any config source
    if args.resize is not None:
        aug_cfg["resize"] = args.resize
        aug_cfg.pop("resize_baseline_index", None)   # force rescan since list changed
        print(f"[config] --resize override applied: {args.resize}")

    if aug_cfg:
        resize_list = aug_cfg.get("resize", [])
        if resize_list:
            print(f"[config] Resize values in dataset: {resize_list}")
            print("[config] % area will be computed from _res{{value}} filename tokens.")
        else:
            print("[config] No resize augmentation in dataset — "
                  "'% area' columns will default to 100.0.")
    else:
        print("[config] No augmentation config loaded — "
              "'% area' columns will default to 100.0 for non-resize images.")

    # ── Step 1: run evaluate_yolo.py ──────────────────────────────────────
    rc = call_evaluate_yolo(args)
    if rc != 0:
        print(f"\n[ERROR] evaluate_yolo.py exited with code {rc}. "
              "Aborting label extraction.")
        sys.exit(rc)

    # ── Step 2: read confidence values ────────────────────────────────────
    summary_csv = output_dir / "yolo_evaluation_results.csv"
    conf_map    = load_conf_map(summary_csv)

    if not conf_map:
        print("[ERROR] Could not read confidence values from "
              f"{summary_csv}. Cannot proceed with label saving.")
        sys.exit(1)

    # ── Step 3: save prediction label txt files ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 – Saving prediction label .txt files")
    print("=" * 60)

    labels_run_dir = output_dir / "pred_labels"
    labels_run_dir.mkdir(parents=True, exist_ok=True)

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"[WARNING] Model not found, skipping: {model_path}")
            continue
        model_name = Path(model_path).name
        for split in args.splits:
            conf = conf_map.get((model_name, split))
            if conf is None:
                print(f"  [warn] No conf found for {model_name}/{split}, skipping.")
                continue
            print(f"  Saving labels: {model_name} / {split} / conf={conf:.4f} ...",
                  end=" ", flush=True)
            pred_dir = save_prediction_labels(
                model_path=model_path,
                data_yaml=args.data,
                split=split,
                conf=conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                labels_dir=labels_run_dir,
            )
            n = len(list(pred_dir.glob("*.txt"))) if pred_dir.exists() else 0
            print(f"done  ({n} label files → {pred_dir})")

    # ── Step 4: build per-label CSV ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 – Building per-label CSV")
    print("=" * 60)

    build_label_csv(
        args=args,
        conf_map=conf_map,
        output_dir=output_dir,
        labels_run_dir=labels_run_dir,
        aug_cfg=aug_cfg,
    )

    print(f"""
Final output layout:
  {output_dir}/
  ├── runs/                          ← evaluate_yolo.py run artefacts
  ├── pred_labels/                   ← prediction label .txt files
  │   └── <split>_<model>_conf<X>/
  │       └── labels/
  │           └── <image_stem>.txt
  ├── yolo_evaluation_results.csv
  ├── per_class_ap.csv
  ├── all_object_results.csv
  └── per_label_results.csv          ← one row per predicted box
                                        (includes '% area' & 'gt % area'
                                         when --config is provided)
""")


if __name__ == "__main__":
    main()