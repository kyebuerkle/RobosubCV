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

    # with resize-augmentation config for % area columns
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

    # Optional augmentation config for % area calculation
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Path to configuration.json containing augmentation "
                        "parameters (e.g. resize list).  Required for "
                        "'%% area' and 'gt %% area' columns.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION CONFIG LOADING
# ─────────────────────────────────────────────────────────────────

def load_augmentation_config(config_path: str) -> dict:
    """
    Load configuration.json and return a parsed config dict.

    Expected structure (example):
        {
          "resize": [0.25, 1.0, 1.75],
          "saturation": [...],
          "exposure": [...]
        }

    Returns empty dict if path is None or file cannot be parsed.
    """
    if config_path is None:
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg
    except Exception as e:
        print(f"[warn] Could not load augmentation config {config_path}: {e}")
        return {}


def build_resize_index_map(aug_cfg: dict) -> dict[int, float]:
    """
    Return {index: resize_value} from the 'resize' list in the config.
    e.g. [0.25, 1.0, 1.75]  →  {0: 0.25, 1: 1.0, 2: 1.75}
    """
    resize_list = aug_cfg.get("resize", [])
    return {i: float(v) for i, v in enumerate(resize_list)}


def get_baseline_res_index(resize_map: dict[int, float]) -> int | None:
    """
    Return the first index whose resize value == 1.0, or None if absent.
    """
    for idx, val in resize_map.items():
        if abs(val - 1.0) < 1e-9:
            return idx
    return None


# ─────────────────────────────────────────────────────────────────
#  FILENAME AUGMENTATION PARSING
# ─────────────────────────────────────────────────────────────────

# Matches trailing augmentation tokens: _sat3, _exp1, _res2, etc.
_AUG_TOKEN_RE = re.compile(r'_(sat|exp|res)(\d+)$', re.IGNORECASE)


def split_aug_suffix(stem: str) -> tuple[str, dict[str, int]]:
    """
    Strip trailing augmentation tokens from a file stem and return
    (base_stem, {aug_type: index}).

    Example:
        "img.png.rf.a1234_sat0_exp1_res2"
        → ("img.png.rf.a1234", {"sat": 0, "exp": 1, "res": 2})

    Tokens are stripped from right to left; parsing stops when no
    augmentation token is found.
    """
    augs: dict[str, int] = {}
    s = stem
    while True:
        m = _AUG_TOKEN_RE.search(s)
        if not m:
            break
        aug_type  = m.group(1).lower()
        aug_index = int(m.group(2))
        augs[aug_type] = aug_index
        s = s[:m.start()]   # remove the matched token
    return s, augs


def make_baseline_stem(base: str, augs: dict[str, int],
                        baseline_res_idx: int) -> str:
    """
    Re-assemble a file stem with the res index replaced by baseline_res_idx,
    preserving the original order of augmentation tokens.

    Token ordering is inferred from the original augs dict.  Because Python
    3.7+ dicts are insertion-ordered and we strip tokens right-to-left, we
    reverse to get original left-to-right order.
    """
    ordered_types = list(reversed(list(augs.keys())))
    parts = [base]
    for aug_type in ordered_types:
        idx = baseline_res_idx if aug_type == "res" else augs[aug_type]
        parts.append(f"_{aug_type}{idx}")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────
#  % AREA CALCULATION
# ─────────────────────────────────────────────────────────────────

def compute_percent_area(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    baseline_cx_n: float, baseline_cy_n: float,
    baseline_w_n: float, baseline_h_n: float,
    resize_scale: float,
) -> float:
    """
    Compute the percentage of an object's true area that remains inside
    the frame after an upscale augmentation.

    Parameters
    ----------
    cx_n, cy_n, w_n, h_n          : normalised box in the AUGMENTED image
    baseline_cx_n, …, baseline_h_n: normalised box in the BASELINE (res=1.0) image
    resize_scale                   : the resize multiplier applied (e.g. 2.0)

    Returns
    -------
    Percentage 0–100 (float).  Returns 100.0 if resize_scale <= 1.0 since
    no upscaling occurred.

    Approach
    --------
    The true (unclipped) object dimensions in normalised coords would be
    baseline_w × resize_scale and baseline_h × resize_scale.  The clipped
    dimensions are what appears in the augmented label (w_n, h_n).  The
    ratio of clipped area to true area gives the fraction visible.
    """
    if resize_scale <= 1.0:
        return 100.0

    true_w = baseline_w_n * resize_scale
    true_h = baseline_h_n * resize_scale
    true_area = true_w * true_h

    if true_area <= 0:
        return 100.0

    clipped_area = w_n * h_n
    fraction_visible = min(clipped_area / true_area, 1.0)
    return round(fraction_visible * 100.0, 4)


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

    result = subprocess.run(cmd)
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
    candidates.append((yaml_path / p).resolve())
    if base:
        base_path = Path(base)
        # PRIMARY: resolve '../split/images' relative to the dataset 'path' dir
        candidates.append((base_path / p).resolve())
        # SECONDARY: resolve relative to path's parent
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
    Pre-indexes all GT label files in a label directory so we can quickly
    look up the baseline (res=1.0) labels for any augmented image stem.

    Usage
    -----
        cache = AreaCache(gt_label_dir, resize_map, baseline_res_idx)
        pct   = cache.get_percent_area(stem, cls_id, cx, cy, w, h,
                                       is_gt=False)
    """

    def __init__(self, gt_label_dir: Path,
                 resize_map: dict[int, float],
                 baseline_res_idx: int | None):
        self.gt_label_dir    = gt_label_dir
        self.resize_map      = resize_map
        self.baseline_res_idx = baseline_res_idx
        # {baseline_stem: np.ndarray of GT labels}
        self._baseline_cache: dict[str, np.ndarray] = {}
        self._available      = baseline_res_idx is not None and bool(resize_map)

    def _load_baseline(self, baseline_stem: str) -> np.ndarray:
        if baseline_stem not in self._baseline_cache:
            txt = self.gt_label_dir / f"{baseline_stem}.txt"
            self._baseline_cache[baseline_stem] = read_yolo_labels(txt)
        return self._baseline_cache[baseline_stem]

    def get_percent_area(
        self,
        stem: str,
        cls_id: int,
        cx_n: float, cy_n: float, w_n: float, h_n: float,
    ) -> float | None:
        """
        Compute % area for a single box.

        Returns None if:
          - no config was provided
          - the stem has no 'res' augmentation token
          - the resize value for this index is <= 1.0 (no upscaling)
          - no matching baseline label exists for the same class
        Returns 100.0 if the label was not cropped at all.
        """
        if not self._available:
            return None

        base, augs = split_aug_suffix(stem)

        if "res" not in augs:
            # Image was not resize-augmented; area is effectively 100 %
            return 100.0

        res_idx      = augs["res"]
        resize_scale = self.resize_map.get(res_idx)

        if resize_scale is None:
            print(f"  [warn] res index {res_idx} not in resize map for stem '{stem}'")
            return None

        # No upscaling → nothing could have been cropped
        if resize_scale <= 1.0:
            return 100.0

        # Build the baseline stem (same sat/exp, but res = baseline_res_idx)
        baseline_stem   = make_baseline_stem(base, augs, self.baseline_res_idx)
        baseline_labels = self._load_baseline(baseline_stem)

        if baseline_labels.shape[0] == 0:
            return None   # no baseline label file

        # Find best-matching baseline box of the same class
        same_cls = baseline_labels[baseline_labels[:, 0].astype(int) == cls_id]
        if same_cls.shape[0] == 0:
            return None   # class not present in baseline

        best_iou   = -1.0
        best_row   = None
        for row in same_cls:
            iou = box_iou_single(cx_n, cy_n, w_n, h_n,
                                 row[1], row[2], row[3], row[4])
            if iou > best_iou:
                best_iou = iou
                best_row = row

        if best_row is None:
            return None

        pct = compute_percent_area(
            cx_n, cy_n, w_n, h_n,
            best_row[1], best_row[2], best_row[3], best_row[4],
            resize_scale,
        )
        return pct


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
    resize_map        = build_resize_index_map(aug_cfg)
    baseline_res_idx  = get_baseline_res_index(resize_map)

    if aug_cfg and baseline_res_idx is None:
        print("[warn] No resize=1.0 entry found in config — "
              "'% area' columns will be empty.")

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

            # ── Build area cache (uses GT label dir for baselines) ─────────
            area_cache = AreaCache(gt_label_dir, resize_map, baseline_res_idx)

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

def main():
    args       = parse_args()
    output_dir = Path(args.output).resolve()

    # ── Load augmentation config (optional) ───────────────────────────────
    aug_cfg = load_augmentation_config(args.config)
    if aug_cfg:
        resize_map = build_resize_index_map(aug_cfg)
        bl_idx     = get_baseline_res_index(resize_map)
        print(f"[config] Resize map: {resize_map}")
        print(f"[config] Baseline res index (1.0): {bl_idx}")
    else:
        print("[config] No augmentation config loaded — "
              "'% area' columns will be empty.")

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