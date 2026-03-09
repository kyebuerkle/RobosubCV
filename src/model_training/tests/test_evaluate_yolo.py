"""
YOLO Model Evaluation Script
------------------------------
Evaluates one or more YOLO models across train/valid/test splits.

Two modes
---------
1. (Default) Auto-find best confidence via F1 curve from a single discovery
   pass, then re-run at the optimal threshold.
2. --conf 0.25,0.50,0.75  Manual sweep of specific confidence values.

How per-object stats are captured (8.4.14)
------------------------------------------
Call order inside BaseValidator.__call__():
    for batch in dataloader:
        update_metrics(preds, batch)          # DetectionValidator appends to
                                              # validator.metrics.stats (dict of lists)
        run_callbacks('on_val_batch_end')     # validator.stats is still None here
    get_stats()                               # calls DetMetrics.process() which does
                                              # np.concatenate on metrics.stats lists
                                              # -- lists are consumed/replaced here
    run_callbacks('on_val_end')               # too late, lists already processed

Solution: monkey-patch validator.metrics.update_stats() during on_val_start
so we copy each batch's tensors into our own accumulator before they are
concatenated and replaced by process().

Output folder structure:
    <output_dir>/
    ├── runs/
    │   ├── val_model1_conf0001/
    │   └── ...
    ├── yolo_evaluation_results.csv
  ├── per_class_ap.csv
    └── all_object_results.csv

Requirements:
    pip install ultralytics==8.4.14 pandas numpy

Usage examples:
    # Auto-find best confidence (default)
    python evaluate_yolo.py -m model1.pt -d dataset/data.yaml

    # Manual confidence sweep
    python evaluate_yolo.py -m model1.pt -d dataset/data.yaml --conf 0.25,0.50,0.75
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
        help="Dataset splits to evaluate.",
    )

    conf_group = parser.add_mutually_exclusive_group()
    conf_group.add_argument(
        "--find-best-conf", action="store_true", default=True,
        help="(Default) Run a discovery pass and pick the confidence that "
             "maximises mean F1.",
    )
    conf_group.add_argument(
        "-c", "--conf",
        default=None,
        type=parse_float_list,
        metavar="LIST",
        help="Comma-separated confidence thresholds for a manual sweep. "
             "Disables --find-best-conf.",
    )

    parser.add_argument(
        "--discovery-conf", type=float, default=0.001,
        metavar="CONF",
        help="Low confidence used for the F1-discovery pass so all detections "
             "are visible and the full curve is populated.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        metavar="IOU",
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        metavar="DIR",
        help="Output folder.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        metavar="SIZE",
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--device", default="cpu",
        metavar="DEVICE",
        help='Device: "cpu", "0" (GPU 0), "0,1" (multi-GPU).',
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  STAT ACCUMULATOR
# ─────────────────────────────────────────────

class StatAccumulator:
    """
    Captures per-detection tp/conf/pred_cls by patching
    validator.metrics.update_stats() during on_val_start.

    DetectionValidator calls validator.metrics.update_stats(batch_stats_dict)
    once per batch inside update_metrics(). We wrap that method to copy the
    tensors into our own lists before they are later concatenated and replaced
    by DetMetrics.process() inside get_stats().

    validator.metrics.stats layout (dict of lists, one entry per batch):
        "tp"         – (N, 10) bool  TP at IoU 0.50 → 0.95 step 0.05
        "conf"       – (N,)    float detection confidence score
        "pred_cls"   – (N,)    int   predicted class index
        "target_cls" – (M,)    int   ground-truth class index (not used here)
        "target_img" – (M,)    int   per-image gt class (not used here)
    """

    def __init__(self):
        self._tp:   list = []
        self._conf: list = []
        self._cls:  list = []
        self._orig_update_stats = None

    def reset(self):
        self._tp   = []
        self._conf = []
        self._cls  = []

    def install(self, validator):
        """Monkey-patch validator.metrics.update_stats on on_val_start."""
        metrics = validator.metrics
        self._orig_update_stats = metrics.update_stats

        accumulator = self  # closure reference

        def patched_update_stats(stats_dict):
            # Call original first so normal bookkeeping still happens
            accumulator._orig_update_stats(stats_dict)

            # Now copy the tensors we care about
            import torch
            for key, dest in [("tp",       accumulator._tp),
                               ("conf",     accumulator._conf),
                               ("pred_cls", accumulator._cls)]:
                val = stats_dict.get(key)
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                arr = np.array(val)
                if arr.size > 0:
                    dest.append(arr)

        metrics.update_stats = patched_update_stats

    def uninstall(self, validator):
        """Restore the original update_stats method."""
        if self._orig_update_stats is not None:
            validator.metrics.update_stats = self._orig_update_stats
            self._orig_update_stats = None

    def build_rows(self, model_name: str, split: str,
                   conf_threshold: float, class_names: dict) -> list[dict]:
        if not self._tp:
            return []

        try:
            tp   = np.concatenate(self._tp,   axis=0).astype(float)
            conf = np.concatenate(self._conf, axis=0).astype(float)
            cls  = np.concatenate(self._cls,  axis=0).astype(int)
        except Exception as e:
            print(f"    [warn] stat concatenation error: {e}")
            return []

        iou50    = tp[:, 0]        if tp.ndim == 2 else tp
        iou50_95 = tp.mean(axis=1) if tp.ndim == 2 else tp

        rows = []
        for cls_id, det_conf, i50, i5095 in zip(cls, conf, iou50, iou50_95):
            rows.append({
                "model":                model_name,
                "split":                split,
                "confidence_threshold": conf_threshold,
                "detection_conf":       round(float(det_conf), 4),
                "object_name":          class_names.get(int(cls_id), f"class_{cls_id}"),
                "object_id":            int(cls_id),
                "iou50":                round(float(i50),   4),
                "iou50-95":             round(float(i5095), 4),
            })
        return rows


# ─────────────────────────────────────────────
#  OPTIMAL CONFIDENCE FROM F1 CURVE
# ─────────────────────────────────────────────

def find_optimal_conf(metrics) -> tuple[float, float]:
    """
    Read the F1-confidence curve built by ap_per_class() inside
    DetMetrics.process() and return (optimal_conf, peak_mean_f1).

    metrics.box.f1_curve – (nc, 1000) F1 per class over confidence axis
    metrics.box.px       – (1000,)    confidence axis 0 → 1
    """
    f1_curve = np.array(metrics.box.f1_curve)  # (nc, 1000)
    px       = np.array(metrics.box.px)         # (1000,)
    mean_f1  = f1_curve.mean(axis=0)            # (1000,)
    best_idx = int(mean_f1.argmax())
    return float(px[best_idx]), float(mean_f1[best_idx])


# ─────────────────────────────────────────────
#  SINGLE VAL RUN
# ─────────────────────────────────────────────

def run_val(model: YOLO, data_yaml: str, split: str,
            conf: float, iou: float, imgsz: int,
            device: str, runs_dir: Path,
            accumulator: StatAccumulator | None = None):
    """
    Run model.val() and return (DetMetrics, elapsed_seconds).
    If accumulator is provided, installs/uninstalls the patch around the run.
    """
    model_stem = Path(str(model.model_name)).stem
    run_name   = f"{split}_{model_stem}_conf{conf:.4f}".replace(".", "")

    if accumulator is not None:
        accumulator.reset()

        # Install patch via on_val_start (fires after init_metrics sets up
        # validator.metrics, so the object exists by the time we patch it)
        def on_val_start(validator):
            accumulator.install(validator)

        def on_val_end(validator):
            accumulator.uninstall(validator)

        model.add_callback("on_val_start", on_val_start)
        model.add_callback("on_val_end",   on_val_end)

    start   = time.perf_counter()
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

    # Remove our callbacks so they don't fire on the next run
    if accumulator is not None:
        for event, fn in [("on_val_start", on_val_start),
                          ("on_val_end",   on_val_end)]:
            model.callbacks[event] = [
                cb for cb in model.callbacks.get(event, []) if cb is not fn
            ]

    return metrics, elapsed



# ─────────────────────────────────────────────
#  PER-CLASS AP EXTRACTION
# ─────────────────────────────────────────────

def build_ap_rows(metrics, model_name: str, split: str,
                  conf_threshold: float, class_names: dict) -> list[dict]:
    """
    Extract per-class AP50 and AP50-95 from DetMetrics after a val run.

    metrics.box.ap50           – (nc,) AP at IoU 0.50 per class
    metrics.box.ap             – (nc,) AP averaged over IoU 0.50:0.95 per class
    metrics.box.ap_class_index – (nc,) integer class indices matching ap50/ap
    """
    try:
        ap50       = np.array(metrics.box.ap50)
        ap50_95    = np.array(metrics.box.ap)
        class_idxs = np.array(metrics.box.ap_class_index, dtype=int)
    except AttributeError as e:
        print(f"    [warn] could not extract per-class AP: {e}")
        return []

    rows = []
    for cls_id, a50, a5095 in zip(class_idxs, ap50, ap50_95):
        rows.append({
            "model":      model_name,
            "split":      split,
            "confidence": round(conf_threshold, 4),
            "object_name": class_names.get(int(cls_id), f"class_{cls_id}"),
            "object_id":  int(cls_id),
            "AP50":       round(float(a50),   4),
            "AP50-95":    round(float(a5095), 4),
        })
    return rows


# ─────────────────────────────────────────────
#  EVALUATE ONE MODEL / SPLIT / CONF
# ─────────────────────────────────────────────

def evaluate_combination(model_path: str, data_yaml: str, split: str,
                         conf: float | None, iou: float, imgsz: int,
                         device: str, runs_dir: Path,
                         find_best: bool,
                         discovery_conf: float) -> tuple[dict, list[dict]]:
    model      = YOLO(model_path)
    model_name = Path(model_path).name
    accum      = StatAccumulator()

    if find_best:
        # ── Discovery pass (no accumulator needed, just need F1 curve) ────
        print(f"    Discovery pass (conf={discovery_conf}) ...", end=" ", flush=True)
        disc_metrics, _ = run_val(
            model, data_yaml, split,
            conf=discovery_conf, iou=iou, imgsz=imgsz,
            device=device, runs_dir=runs_dir,
        )
        optimal_conf, peak_f1 = find_optimal_conf(disc_metrics)
        print(f"optimal conf = {optimal_conf:.4f}  (peak F1 = {peak_f1:.4f})")

        # ── Final pass at optimal conf ─────────────────────────────────────
        print(f"    Final pass   (conf={optimal_conf:.4f}) ...", end=" ", flush=True)
        metrics, elapsed = run_val(
            model, data_yaml, split,
            conf=optimal_conf, iou=iou, imgsz=imgsz,
            device=device, runs_dir=runs_dir,
            accumulator=accum,
        )
        used_conf = optimal_conf

    else:
        metrics, elapsed = run_val(
            model, data_yaml, split,
            conf=conf, iou=iou, imgsz=imgsz,
            device=device, runs_dir=runs_dir,
            accumulator=accum,
        )
        used_conf = conf

    summary = {
        "model":      model_name,
        "split":      split,
        "confidence": round(used_conf, 4),
        "time_s":     round(elapsed, 3),
        "precision":  round(float(metrics.box.mp),    4),
        "recall":     round(float(metrics.box.mr),    4),
        "mAP50":      round(float(metrics.box.map50), 4),
        "mAP50-95":   round(float(metrics.box.map),   4),
    }

    obj_rows = accum.build_rows(model_name, split, used_conf, model.names)
    ap_rows  = build_ap_rows(metrics, model_name, split, used_conf, model.names)
    return summary, obj_rows, ap_rows


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run_evaluation(args):
    output_dir = Path(args.output).resolve()
    runs_dir   = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    find_best   = args.conf is None
    conf_values = args.conf or [None]

    summary_rows = []
    object_rows  = []
    ap_rows_all  = []

    combos     = [(m, s, c) for m in args.models
                             for s in args.splits
                             for c in conf_values]
    total_runs = len(combos)
    run_idx    = 0

    for model_path, split, conf in combos:
        run_idx += 1
        if not Path(model_path).exists():
            print(f"[WARNING] Model not found, skipping: {model_path}")
            continue

        mode_label = "auto-F1" if find_best else f"conf={conf:.4f}"
        print(f"\n[{run_idx}/{total_runs}]  "
              f"Model={Path(model_path).name}  "
              f"Split={split}  Mode={mode_label}")

        try:
            summary, obj_rows, ap_rows = evaluate_combination(
                model_path=model_path,
                data_yaml=args.data,
                split=split,
                conf=conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                runs_dir=runs_dir,
                find_best=find_best,
                discovery_conf=args.discovery_conf,
            )
            summary_rows.append(summary)
            object_rows.extend(obj_rows)
            ap_rows_all.extend(ap_rows)

            print(f"    → conf={summary['confidence']}  "
                  f"P={summary['precision']:.3f}  "
                  f"R={summary['recall']:.3f}  "
                  f"mAP50={summary['mAP50']:.3f}  "
                  f"mAP50-95={summary['mAP50-95']:.3f}  "
                  f"({summary['time_s']}s)  "
                  f"[{len(obj_rows)} detections]")

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
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
        cols = ["model", "split", "confidence", "time_s",
                "precision", "recall", "mAP50", "mAP50-95"]
        if "error" in df_s.columns:
            cols.append("error")
        df_s[cols].to_csv(summary_csv, index=False)
        print(f"\n✅ Summary CSV   → {summary_csv}")
        print(df_s[cols].to_string(index=False))
    else:
        print("No summary results to write.")

    # ── Write per_class_ap.csv ───────────────
    ap_csv = output_dir / "per_class_ap.csv"
    if ap_rows_all:
        df_ap = pd.DataFrame(ap_rows_all)
        ap_cols = ["model", "split", "confidence", "object_name", "object_id",
                   "AP50", "AP50-95"]
        df_ap[ap_cols].to_csv(ap_csv, index=False)
        print(f"✅ Per-class AP  → {ap_csv}  ({len(df_ap)} class entries)")
    else:
        print("\n⚠️  No per-class AP data extracted.")

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
        print("\n⚠️  No per-object data extracted.")

    print(f"""
Output layout:
  {output_dir}/
  ├── runs/
  │   └── <split>_<model>_conf<X>/
  ├── yolo_evaluation_results.csv
  ├── per_class_ap.csv
  └── all_object_results.csv
""")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    find_best = args.conf is None

    print("=" * 60)
    print("YOLO Evaluation")
    print("=" * 60)
    print(f"  Models        : {args.models}")
    print(f"  Data YAML     : {args.data}")
    print(f"  Splits        : {args.splits}")
    print(f"  Conf mode     : {'auto (F1-peak)' if find_best else args.conf}")
    if find_best:
        print(f"  Discovery conf: {args.discovery_conf}")
    print(f"  IoU           : {args.iou}")
    print(f"  Image size    : {args.imgsz}")
    print(f"  Device        : {args.device}")
    print(f"  Output dir    : {args.output}")
    print("=" * 60 + "\n")

    run_evaluation(args)