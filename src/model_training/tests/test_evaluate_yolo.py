"""
YOLO Model Evaluation Script
------------------------------
Evaluates one or more YOLO models across train/valid/test splits
at multiple confidence thresholds, and writes results to two CSVs.

Tested against: ultralytics==8.4.14

Why the callback approach is needed
-------------------------------------
In 8.4.x, DetectionValidator.get_stats() calls self.metrics.clear_stats()
before returning, which wipes validator.stats. By the time on_val_end fires
the data is already gone. Instead we accumulate raw tensors ourselves on
every on_val_batch_end callback, which fires after each batch's
update_metrics() call while the data is still alive.

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

    # Two models, custom confidence levels, specific splits, GPU
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
        default="0.25,0.35,0.45,0.50,0.55,0.65,0.75",
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
#  BATCH-LEVEL STAT ACCUMULATOR
# ─────────────────────────────────────────────

class StatAccumulator:
    """
    Accumulates per-detection tensors on every on_val_batch_end callback.

    In ultralytics 8.4.x, DetectionValidator.update_metrics() appends to
    self.stats (a dict of lists of Tensors) each batch. The keys are:
        "tp"       – (N, 10) bool  TP at IoU 0.50 → 0.95
        "conf"     – (N,)    float detection confidence
        "pred_cls" – (N,)    int   predicted class index
        "target_cls"– (M,)   int   ground-truth class index (not used here)

    We grab them here before get_stats() clears them.
    """

    def __init__(self):
        self.tp_list:   list = []
        self.conf_list: list = []
        self.cls_list:  list = []
        self._keys_printed = False

    def reset(self):
        self.tp_list   = []
        self.conf_list = []
        self.cls_list  = []

    def on_val_batch_end(self, validator):
        """Called after each validation batch. Drain the latest batch stats."""
        stats = getattr(validator, "stats", None)
        if stats is None:
            print(f"\n    [warn] validator.stats doesn't exist")
            return

        # Print keys once so the user can verify / debug
        if not self._keys_printed:
            print(f"\n    [debug] validator.stats keys: {list(stats.keys())}")
            self._keys_printed = True

        # Each call to update_metrics appends one element to each list.
        # We pop the last item so we don't double-count on the next batch.
        tp_list   = stats.get("tp",       [])
        conf_list = stats.get("conf",     [])
        cls_list  = stats.get("pred_cls", [])

        if not tp_list:
            return

        # Take the latest batch entry (last in list)
        tp   = tp_list[-1]
        conf = conf_list[-1]
        cls  = cls_list[-1]

        try:
            import torch
            if isinstance(tp, torch.Tensor):
                tp   = tp.cpu().numpy()
                conf = conf.cpu().numpy()
                cls  = cls.cpu().numpy()

            if len(tp) > 0:
                self.tp_list.append(np.array(tp,   dtype=float))
                self.conf_list.append(np.array(conf, dtype=float))
                self.cls_list.append(np.array(cls,  dtype=int))
        except Exception as e:
            print(f"    [warn] could not accumulate batch stats: {e}")

    def build_rows(self, model_name: str, split: str,
                   conf_threshold: float, class_names: dict) -> list[dict]:
        """Convert accumulated tensors into per-detection row dicts."""
        if not self.tp_list:
            return []

        try:
            tp   = np.concatenate(self.tp_list,   axis=0)   # (N, 10)
            conf = np.concatenate(self.conf_list, axis=0)   # (N,)
            cls  = np.concatenate(self.cls_list,  axis=0)   # (N,)
        except Exception as e:
            print(f"    [warn] could not concatenate accumulated stats: {e}")
            return []

        if tp.ndim == 1:
            iou50    = tp
            iou50_95 = tp
        else:
            iou50    = tp[:, 0]        # matched at IoU ≥ 0.50
            iou50_95 = tp.mean(axis=1) # mean across 0.50 : 0.05 : 0.95

        rows = []
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
    run_name   = f"{split}_{model_stem}_conf{conf:.2f}".replace(".", "")

    # ── Set up per-batch accumulator ──────────
    accumulator = StatAccumulator()
    model.add_callback("on_val_batch_end", accumulator.on_val_batch_end)

    # ── Run validation ─────────────────────────
    start = time.perf_counter()

    metrics = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
    )

    elapsed = time.perf_counter() - start

    # ── Summary metrics from DetMetrics ───────
    # model.val() returns DetMetrics directly in 8.4.x
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)

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

    # ── Build per-object rows ──────────────────
    obj_rows = accumulator.build_rows(model_name, split, conf, model.names)

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
            "   Check the [debug] line above for the actual keys in validator.stats\n"
            "   and update the key names in StatAccumulator.on_val_batch_end()."
        )

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