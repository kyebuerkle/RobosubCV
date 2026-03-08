"""
YOLO Model Evaluation Script
------------------------------
Evaluates one or more YOLO models across train/valid/test splits
at multiple confidence thresholds, and writes results to two CSVs.

Tested against: ultralytics==8.4.14

Output folder structure:
    <output_dir>/
    ├── runs/
    │   ├── val_model1_conf025/
    │   └── ...
    ├── yolo_evaluation_results.csv
    └── all_object_results.csv

Requirements:
    pip install ultralytics==8.4.14 pandas numpy

Usage examples:
    # Single model, default confidence sweep, all splits
    python evaluate_yolo.py --models model1.pt --data dataset/data.yaml

    # Two models, custom confidence levels, specific splits
    python evaluate_yolo.py \\
        --models model1.pt model2.pt \\
        --data dataset/data.yaml \\
        --splits train val test \\
        --conf 0.25,0.50,0.75 \\
        --iou 0.5 \\
        --output output/ \\
        --imgsz 640 \\
        --device 0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO
from general_lib import parse_float_list


# ─────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model(s) across splits and confidence thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--models", nargs="+", required=True,
        metavar="PATH",
        help="Path(s) to one or more YOLO .pt model files.",
    )
    parser.add_argument(
        "-d", "--data", required=True,
        metavar="PATH",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        metavar="SPLIT",
        help="Dataset splits to evaluate. Choices: train val test.",
    )
    parser.add_argument(
        "-c", "--conf",
        default="0.25,0.5,0.75",
        type=parse_float_list,
        metavar="LIST",
        help="Comma-separated confidence thresholds to sweep (e.g. 0.25,0.5,0.75).",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        metavar="IOU",
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        metavar="DIR",
        help="Output folder. CSVs and runs/ subfolder are written here.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        metavar="SIZE",
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--device", default="cpu",
        metavar="DEVICE",
        help='Device to run on: "cpu", "0" (GPU 0), "0,1" (multi-GPU).',
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  PER-OBJECT ROW EXTRACTION
# ─────────────────────────────────────────────

def extract_object_rows(results, model_name: str, split: str,
                        conf_threshold: float, class_names: dict) -> list[dict]:
    """
    Extract per-detection IoU values from validator.metrics.stats.

    In ultralytics 8.4.x, DetectionValidator stores accumulated stats inside
    the DetMetrics object (validator.metrics), NOT directly on validator.stats.

    validator.metrics.stats is a dict with lists of numpy arrays, one entry
    per batch, with keys:
        "tp"         – list of (N, 10) bool arrays  [TP at 10 IoU thresholds]
        "conf"       – list of (N,)   float arrays  [detection confidence]
        "pred_cls"   – list of (N,)   int arrays    [predicted class index]
        "target_cls" – list of (M,)   int arrays    [ground-truth class index]
        "target_img" – list of unique GT classes per image

    After concatenating across batches:
        iou50    = tp[:, 0]           (matched at IoU ≥ 0.50)
        iou50-95 = tp.mean(axis=1)    (mean across 10 thresholds 0.50→0.95)
    """
    rows = []

    validator = getattr(results, "validator", None)
    if validator is None:
        print("    [warn] no validator attached to results")
        return rows

    # In 8.4.x stats live on validator.metrics.stats (a DetMetrics object)
    metrics_obj = getattr(validator, "metrics", None)
    if metrics_obj is None:
        print("    [warn] validator has no .metrics attribute")
        return rows

    stats = getattr(metrics_obj, "stats", None)
    if not stats:
        print("    [warn] validator.metrics.stats is empty")
        return rows

    # Each value is a list of per-batch arrays — concatenate them
    try:
        tp_raw   = stats.get("tp",       None)
        conf_raw = stats.get("conf",     None)
        cls_raw  = stats.get("pred_cls", None)

        if tp_raw is None or conf_raw is None or cls_raw is None:
            print(f"    [warn] missing keys in stats. Found keys: {list(stats.keys())}")
            return rows

        tp   = np.concatenate(tp_raw,   axis=0)   # (N_total, 10)
        conf = np.concatenate(conf_raw, axis=0)   # (N_total,)
        cls  = np.concatenate(cls_raw,  axis=0).astype(int)  # (N_total,)

    except Exception as e:
        print(f"    [warn] could not concatenate stats arrays: {e}")
        return rows

    if tp.ndim == 1:
        # Fallback: single IoU threshold stored as 1-D
        iou50    = tp.astype(float)
        iou50_95 = tp.astype(float)
    else:
        iou50    = tp[:, 0].astype(float)    # IoU threshold = 0.50
        iou50_95 = tp.mean(axis=1)           # mean over 0.50 : 0.05 : 0.95

    for cls_id, det_conf, i50, i5095 in zip(cls, conf, iou50, iou50_95):
        obj_name = class_names.get(int(cls_id), f"class_{cls_id}")
        rows.append({
            "model":                model_name,
            "split":                split,
            "confidence_threshold": conf_threshold,
            "detection_conf":       round(float(det_conf), 4),
            "object_name":          obj_name,
            "object_id":            int(cls_id),
            "iou50":                round(float(i50),   4),
            "iou50-95":             round(float(i5095), 4),
        })

    return rows


# ─────────────────────────────────────────────
#  SINGLE RUN
# ─────────────────────────────────────────────

def evaluate_model(model_path: str, data_yaml: str, split: str,
                   conf: float, iou: float, imgsz: int,
                   device: str, runs_dir: Path) -> tuple[dict, list[dict]]:
    """
    Run YOLO validation for one model / split / confidence combination.
    Returns (summary_dict, list_of_per_object_dicts).
    """
    model      = YOLO(model_path)
    model_name = Path(model_path).name
    model_stem = Path(model_path).stem

    # e.g.  val_model1_conf025
    run_name = f"{split}_{model_stem}_conf{conf:.2f}".replace(".", "")

    start = time.perf_counter()

    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        project=str(runs_dir),   # → <output>/runs/
        name=run_name,           # → <output>/runs/<run_name>/
        exist_ok=True,
    )

    elapsed = time.perf_counter() - start

    # ── Summary metrics ────────────────────────
    m         = results.results_dict
    precision = m.get("metrics/precision(B)", float("nan"))
    recall    = m.get("metrics/recall(B)",    float("nan"))
    map50     = m.get("metrics/mAP50(B)",     float("nan"))
    map50_95  = m.get("metrics/mAP50-95(B)",  float("nan"))

    summary = {
        "model":      model_name,
        "split":      split,
        "confidence": conf,
        "time_s":     round(elapsed, 3),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "mAP50":      round(map50,     4),
        "mAP50-95":   round(map50_95,  4),
    }

    # ── Per-object rows ────────────────────────
    obj_rows = extract_object_rows(
        results, model_name, split, conf, model.names
    )

    return summary, obj_rows


# ─────────────────────────────────────────────
#  MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def run_evaluation(args):
    output_dir = Path(args.output).resolve()
    runs_dir   = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    object_rows  = []

    total_runs = len(args.models) * len(args.splits) * len(args.conf)
    run_idx    = 0

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"[WARNING] Model not found, skipping: {model_path}")
            continue

        for split in args.splits:
            for conf in args.conf:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}]  "
                    f"Model={Path(model_path).name}  "
                    f"Split={split}  Conf={conf:.2f} ...",
                    end=" ", flush=True,
                )

                try:
                    summary, obj_rows = evaluate_model(
                        model_path=model_path,
                        data_yaml=args.data,
                        split=split,
                        conf=conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=args.device,
                        runs_dir=runs_dir,
                    )
                    summary_rows.append(summary)
                    object_rows.extend(obj_rows)

                    print(
                        f"P={summary['precision']:.3f}  "
                        f"R={summary['recall']:.3f}  "
                        f"mAP50={summary['mAP50']:.3f}  "
                        f"mAP50-95={summary['mAP50-95']:.3f}  "
                        f"({summary['time_s']}s)  "
                        f"[{len(obj_rows)} detections]"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    summary_rows.append({
                        "model":      Path(model_path).name,
                        "split":      split,
                        "confidence": conf,
                        "time_s":     float("nan"),
                        "precision":  float("nan"),
                        "recall":     float("nan"),
                        "mAP50":      float("nan"),
                        "mAP50-95":   float("nan"),
                        "error":      str(e),
                    })

    # ── Write yolo_evaluation_results.csv ─────
    summary_csv = output_dir / "yolo_evaluation_results.csv"
    if summary_rows:
        df_s = pd.DataFrame(summary_rows)
        col_order = ["model", "split", "confidence", "time_s",
                     "precision", "recall", "mAP50", "mAP50-95"]
        if "error" in df_s.columns:
            col_order.append("error")
        df_s[col_order].to_csv(summary_csv, index=False)
        print(f"\n✅ Summary CSV   → {summary_csv}")
        print(df_s[col_order].to_string(index=False))
    else:
        print("No summary results to write.")

    # ── Write all_object_results.csv ──────────
    object_csv = output_dir / "all_object_results.csv"
    if object_rows:
        df_o = pd.DataFrame(object_rows)
        obj_cols = ["model", "split", "confidence_threshold",
                    "detection_conf", "object_name", "object_id",
                    "iou50", "iou50-95"]
        df_o[obj_cols].to_csv(object_csv, index=False)
        print(f"✅ Object CSV    → {object_csv}  ({len(df_o)} detections)")
    else:
        print(
            "\n⚠️  No per-object data was extracted.\n"
            "   Run with a single conf threshold and add a quick debug print:\n"
            "     print(results.validator.metrics.stats.keys())\n"
            "   to confirm the exact key names on your build."
        )

    # ── Folder layout ──────────────────────────
    print(f"""
Output layout:
  {output_dir}/
  ├── runs/
  │   └── <split>_<model>_conf<X>/   ← one folder per run
  ├── yolo_evaluation_results.csv
  └── all_object_results.csv
""")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("YOLO Evaluation")
    print("=" * 60)
    print(f"  Models    : {args.models}")
    print(f"  Data YAML : {args.data}")
    print(f"  Splits    : {args.splits}")
    print(f"  Confidence: {args.conf}")
    print(f"  IoU       : {args.iou}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device    : {args.device}")
    print(f"  Output dir: {args.output}")
    print("=" * 60 + "\n")

    run_evaluation(args)
