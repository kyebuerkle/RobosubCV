"""
YOLO Model Evaluation Script
------------------------------
Evaluates one or more YOLO models across train/valid/test splits
at multiple confidence thresholds, and writes results to a CSV.

Requirements:
    pip install ultralytics pandas

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
        --output results.csv \\
        --imgsz 640 \\
        --device 0
"""

import argparse
import time
from pathlib import Path

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
        '-m',"--models", nargs="+", required=True,
        metavar="PATH",
        help="Path(s) to one or more YOLO .pt model files.",
    )
    parser.add_argument(
        '-d',"--data", required=True,
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
        '-c', "--conf",
        default="0.25,0.35,0.45,0.50,0.55,0.65,0.75",
        type=parse_float_list,
        metavar="LIST",
        help="One or more confidence thresholds to sweep (e.g. 0.25,0.5,0.75).",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        metavar="IOU",
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        '-o',"--output", default="output/yolo_evaluation_results.csv",
        metavar="FILE",
        help="Output CSV file path.",
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
#  EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model_path: str, data_yaml: str, split: str,
                   conf: float, iou: float, imgsz: int, device: str) -> dict:
    """
    Run YOLO validation for one model / split / confidence combination.
    Returns a dict of metrics.
    """
    model = YOLO(model_path)

    start = time.perf_counter()

    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    metrics = results.results_dict

    precision = metrics.get("metrics/precision(B)", float("nan"))
    recall    = metrics.get("metrics/recall(B)",    float("nan"))
    map50     = metrics.get("metrics/mAP50(B)",     float("nan"))
    map50_95  = metrics.get("metrics/mAP50-95(B)",  float("nan"))

    return {
        "model":      Path(model_path).name,
        "split":      split,
        "confidence": conf,
        "time_s":     round(elapsed, 3),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "mAP50":      round(map50,     4),
        "mAP50-95":   round(map50_95,  4),
    }


def run_evaluation(args):
    output = Path(args.output).resolve().absolute()
    output.parent.mkdir(exist_ok=True)
    rows = []

    total_runs = len(args.models) * len(args.splits) * len(args.conf)
    run_idx = 0

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
                    row = evaluate_model(
                        model_path=model_path,
                        data_yaml=args.data,
                        split=split,
                        conf=conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=args.device,
                    )
                    rows.append(row)
                    print(
                        f"P={row['precision']:.3f}  "
                        f"R={row['recall']:.3f}  "
                        f"mAP50={row['mAP50']:.3f}  "
                        f"mAP50-95={row['mAP50-95']:.3f}  "
                        f"({row['time_s']}s)"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    rows.append({
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

    # ── Write CSV ──────────────────────────────
    if not rows:
        print("No results to write.")
        return

    df = pd.DataFrame(rows)

    col_order = ["model", "split", "confidence", "time_s",
                 "precision", "recall", "mAP50", "mAP50-95"]
    if "error" in df.columns:
        col_order.append("error")
    df = df[col_order]

    df.to_csv(output, index=False)
    print(f"\n✅ Results saved to: {output}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # ── Print run summary ──────────────────────
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
    print(f"  Output    : {args.output}")
    print("=" * 60 + "\n")

    run_evaluation(args)