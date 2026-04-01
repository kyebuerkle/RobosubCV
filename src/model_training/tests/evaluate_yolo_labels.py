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
   iou50                         ← IoU@0.50 (1.0 if TP, 0.0 if FP)

   The confidence used for the prediction pass is taken from
   all_object_results.csv (the optimal confidence chosen by evaluate_yolo.py
   for each model/split), so the predicted boxes here correspond exactly to
   those evaluated there.

Requirements
------------
    pip install ultralytics==8.4.14 pandas numpy pyyaml

Usage
-----
    # mirrors evaluate_yolo.py flags — pass the same arguments
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml

    # manual confidence sweep
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml --conf 0.25,0.50

    # limit to specific splits
    python evaluate_yolo_labels.py -m model1.pt -d dataset/data.yaml --splits val test
"""

import argparse
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
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  STEP 1 – CALL evaluate_yolo.py
# ─────────────────────────────────────────────────────────────────

def call_evaluate_yolo(args: argparse.Namespace) -> int:
    """Build the subprocess command and run evaluate_yolo.py."""
    eval_script = args.eval_script
    if eval_script is None:
        # Default: same directory as this script
        eval_script = Path(__file__).parent / "evaluate_yolo.py"

    eval_script = Path(eval_script)
    if not eval_script.exists():
        print(f"[ERROR] evaluate_yolo.py not found at: {eval_script}")
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
    # if conf is None, evaluate_yolo.py defaults to --find-best-conf

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

    YOLO writes labels to:
        <project>/<name>/labels/<image_stem>.txt

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
        save_conf=True,       # appends confidence as 6th column in each txt line
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

    Each line: class_id  cx  cy  w  h  [conf]

    Returns float array shape (N, 5+) where columns are
    [class_id, cx, cy, w, h, (conf if present)].
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

    arr = np.array(rows, dtype=float)
    return arr


def normalised_to_pixels(cx_n, cy_n, w_n, h_n, img_w: int, img_h: int):
    """Convert normalised YOLO box to pixel (x1,y1,x2,y2, w_px, h_px)."""
    w_px = w_n * img_w
    h_px = h_n * img_h
    return w_px, h_px


# ─────────────────────────────────────────────────────────────────
#  HELPERS – DATASET PATHS FROM YAML
# ─────────────────────────────────────────────────────────────────

def get_split_image_dir(data_yaml_path: str, split: str) -> Path | None:
    """
    Parse the dataset YAML and return the image directory for the given split.

    Handles the standard Roboflow/YOLO layout:
        path: /dataset/foo                  ← base directory
        train: ../train/images              ← relative to path's PARENT
        val:   ../valid/images
        test:  ../test/images

    Resolution order:
        1. Absolute path → use directly
        2. path_parent / raw  (Roboflow standard: ../split/images)
        3. path_dir    / raw  (alternative: split/images with no ../)
        4. yaml_dir    / raw  (fallback for older layouts)
    """
    yaml_path = Path(data_yaml_path).resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # 'val' split key may be labelled 'val' or 'valid' in some YAMLs
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
    if base:
        base_path = Path(base)
        # Roboflow standard: path key is the dataset dir, splits use ../sibling/images
        # so the true root is path's parent
        candidates.append(base_path.parent / p)
        # Also try path / raw directly (no leading ../)
        candidates.append(base_path / p)

    # Fallback: relative to the YAML file's own directory
    candidates.append(yaml_path.parent / p)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return None


def image_dir_to_label_dir(img_dir: Path) -> Path:
    """
    YOLO dataset convention: swap the 'images' component for 'labels'.
    e.g.  .../dataset/images/val  →  .../dataset/labels/val
    """
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    # Fallback: sibling directory named 'labels'
    return img_dir.parent / "labels" / img_dir.name


def get_image_size(img_path: Path) -> tuple[int, int]:
    """
    Return (width, height) of an image without heavy dependencies.
    Uses PIL if available, otherwise reads PNG/JPEG headers directly.
    """
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size   # (width, height)
    except ImportError:
        pass

    # Minimal PNG header parse
    suffix = img_path.suffix.lower()
    if suffix == ".png":
        with open(img_path, "rb") as f:
            f.read(16)        # signature + IHDR length + type
            import struct
            f.seek(16)
            w, h = struct.unpack(">II", f.read(8))
            return w, h

    # JPEG: scan for SOF marker
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

    area_a = w1 * h1
    area_b = w2 * h2
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
#  STEP 3 – BUILD PER-LABEL CSV
# ─────────────────────────────────────────────────────────────────

def build_label_csv(args: argparse.Namespace,
                    conf_map: dict[tuple[str, str], float],
                    output_dir: Path,
                    labels_run_dir: Path) -> Path:
    """
    Walk every model × split, read predicted and GT label files,
    match predictions to GT boxes, and write per_label_results.csv.

    conf_map: {(model_name, split): optimal_conf}
    """
    class_names_cache: dict[str, dict] = {}   # model_path → {id: name}

    rows = []

    for model_path in args.models:
        if not Path(model_path).exists():
            continue

        model_name = Path(model_path).name
        model_stem = Path(model_path).stem

        # Cache class names
        if model_path not in class_names_cache:
            model_obj = YOLO(model_path)
            class_names_cache[model_path] = model_obj.names   # {int: str}

        class_names = class_names_cache[model_path]

        for split in args.splits:
            conf = conf_map.get((model_name, split))
            if conf is None:
                print(f"  [warn] No confidence found for {model_name}/{split}, skipping.")
                continue

            # ── Locate predicted labels directory ─────────────────────────
            run_name   = f"{split}_{model_stem}_conf{conf:.4f}".replace(".", "")
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

            # ── Collect image paths so we know actual pixel sizes ─────────
            img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            image_files    = {
                p.stem: p
                for p in img_dir.iterdir()
                if p.suffix.lower() in img_extensions
            }

            print(f"  Building label rows: {model_name} / {split} / conf={conf:.4f} "
                  f"({len(list(pred_label_dir.glob('*.txt')))} pred files, "
                  f"{len(image_files)} images)")

            # ── Iterate over every predicted label file ───────────────────
            for pred_txt in sorted(pred_label_dir.glob("*.txt")):
                stem = pred_txt.stem

                # Image size (pixels)
                img_path = image_files.get(stem)
                if img_path is None:
                    # Try to find image with any extension
                    matches = [v for k, v in image_files.items()
                               if k.lower() == stem.lower()]
                    img_path = matches[0] if matches else None

                if img_path is not None:
                    try:
                        img_w, img_h = get_image_size(img_path)
                    except Exception as e:
                        print(f"    [warn] Cannot read image size for {img_path}: {e}")
                        img_w = img_h = args.imgsz   # fallback to inference size
                else:
                    # No image found — use inference size as best guess
                    img_w = img_h = args.imgsz

                # Read predictions  — columns: [cls, cx, cy, w, h, (conf)]
                preds = read_yolo_labels(pred_txt)

                # Read GT
                gt_txt = gt_label_dir / f"{stem}.txt"
                gt     = read_yolo_labels(gt_txt)

                # ── Match each predicted box to best GT box ────────────────
                for pred_row in preds:
                    p_cls          = int(pred_row[0])
                    p_cx, p_cy     = pred_row[1], pred_row[2]
                    p_w_n, p_h_n   = pred_row[3], pred_row[4]

                    p_w_px = round(p_w_n * img_w, 2)
                    p_h_px = round(p_h_n * img_h, 2)

                    # Find GT boxes of the same class
                    gt_match_iou   = 0.0
                    gt_w_px        = float("nan")
                    gt_h_px        = float("nan")

                    if gt.shape[0] > 0:
                        same_cls_mask = gt[:, 0].astype(int) == p_cls
                        gt_same       = gt[same_cls_mask]

                        best_iou   = 0.0
                        best_gt_w  = float("nan")
                        best_gt_h  = float("nan")

                        for gt_row in gt_same:
                            iou = box_iou_single(
                                p_cx, p_cy, p_w_n, p_h_n,
                                gt_row[1], gt_row[2], gt_row[3], gt_row[4],
                            )
                            if iou > best_iou:
                                best_iou  = iou
                                best_gt_w = round(gt_row[3] * img_w, 2)
                                best_gt_h = round(gt_row[4] * img_h, 2)

                        gt_match_iou = best_iou
                        gt_w_px      = best_gt_w
                        gt_h_px      = best_gt_h

                    # iou50 column: 1.0 if TP (IoU >= 0.50), else 0.0
                    iou50 = 1.0 if gt_match_iou >= 0.50 else 0.0

                    rows.append({
                        "model":       model_name,
                        "split":       split,
                        "file_name":   stem,
                        "object_name": class_names.get(p_cls, f"class_{p_cls}"),
                        "class_id":    p_cls,
                        "width_px":    p_w_px,
                        "height_px":   p_h_px,
                        "gt_width_px": gt_w_px,
                        "gt_height_px":gt_h_px,
                        "iou50":       round(gt_match_iou, 4),
                    })

    # ── Write CSV ─────────────────────────────────────────────────────
    out_csv = output_dir / "per_label_results.csv"
    if rows:
        df = pd.DataFrame(rows)
        cols = ["model", "split", "file_name", "object_name", "class_id",
                "width_px", "height_px", "gt_width_px", "gt_height_px", "iou50"]
        df[cols].to_csv(out_csv, index=False)
        print(f"\n✅ Per-label CSV → {out_csv}  ({len(df)} rows)")
    else:
        print("\n⚠️  No label rows produced.")

    return out_csv


# ─────────────────────────────────────────────────────────────────
#  LOAD CONFIDENCE MAP FROM SUMMARY CSV
# ─────────────────────────────────────────────────────────────────

def load_conf_map(summary_csv: Path) -> dict[tuple[str, str], float]:
    """
    Read yolo_evaluation_results.csv and return
    {(model_name, split): confidence} so we know which conf each
    model/split was evaluated at (especially useful in auto-F1 mode).
    """
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv)
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

    # ── Step 1: run evaluate_yolo.py ──────────────────────────────────────
    rc = call_evaluate_yolo(args)
    if rc != 0:
        print(f"\n[ERROR] evaluate_yolo.py exited with code {rc}. "
              "Aborting label extraction.")
        sys.exit(rc)

    # ── Step 2: read the confidence values that were used ─────────────────
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
  └── per_label_results.csv          ← NEW: one row per predicted box
""")


if __name__ == "__main__":
    main()