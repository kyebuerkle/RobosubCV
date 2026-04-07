"""
summarize_yolo_results.py
--------------------------
Reads the output CSV files produced by evaluate_yolo.py /
evaluate_yolo_labels.py and writes a single consolidated summary CSV.

Which columns appear in the summary depends on which input files exist:

  all_object_results.csv only
      model, split, confidence_threshold, class_name, class_id,
      object_count, avg_iou50, avg_iou50_95

  + per_class_ap.csv
      … + AP50, AP50_95

  + per_label_results.csv  (no AP)
      … + min_area_px, max_area_px, avg_area_px,
            min_pct_area, avg_pct_area

  all three files
      … + AP50, AP50_95 + area columns

Any column that cannot be computed (e.g. '% area' missing from
per_label_results.csv) triggers a debug warning but the rest of the
row is still written.

Usage
-----
    python summarize_yolo_results.py -d output/
    python summarize_yolo_results.py -d output/ -o summary.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise YOLO evaluation output CSVs into one file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-d", "--directory", required=True, metavar="DIR",
        help="Directory containing the evaluation output CSV files.",
    )
    p.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Output CSV path. Defaults to <directory>/summary.csv.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  FILE DETECTION
# ─────────────────────────────────────────────────────────────────

REQUIRED_FILE  = "all_object_results.csv"
AP_FILE        = "per_class_ap.csv"
LABEL_FILE     = "per_label_results.csv"


def detect_files(directory: Path) -> dict[str, Path | None]:
    """
    Return a dict of {logical_name: Path | None} for each expected file.
    Prints which files were found/missing.
    """
    files = {
        "raw":   directory / REQUIRED_FILE,
        "ap":    directory / AP_FILE,
        "label": directory / LABEL_FILE,
    }

    print("\n── Detected output files ──────────────────────────────────")
    for name, path in files.items():
        status = "✓ found" if path.exists() else "✗ missing"
        print(f"  [{status}]  {path.name}")
    print()

    return {k: (v if v.exists() else None) for k, v in files.items()}


# ─────────────────────────────────────────────────────────────────
#  SAFE COLUMN HELPERS
# ─────────────────────────────────────────────────────────────────

def safe_mean(series: pd.Series) -> float | None:
    """Mean of a numeric series; None if all NaN or empty."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else None


def safe_min(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.min()) if len(vals) > 0 else None


def safe_max(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.max()) if len(vals) > 0 else None


def warn_missing_col(col: str, source: str):
    print(f"  [debug] Column '{col}' not found in {source} — "
          "that field will be blank in the summary.")


# ─────────────────────────────────────────────────────────────────
#  LOAD & NORMALISE INPUT FILES
# ─────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    """
    Load all_object_results.csv.
    Expected columns:
        model, split, confidence_threshold, detection_conf,
        object_name, object_id, iou50, iou50-95
    """
    df = pd.read_csv(path)

    # Normalise column names to lowercase / underscores for safety
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

    required = {"model", "split", "object_name"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{REQUIRED_FILE} is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Ensure numeric columns
    for col in ("iou50", "iou50_95", "confidence_threshold", "object_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_ap(path: Path) -> pd.DataFrame | None:
    """
    Load per_class_ap.csv.
    Expected columns:
        model, split, confidence, object_name, object_id, AP50, AP50-95
    """
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
        for col in ("ap50", "ap50_95", "confidence"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"  [debug] Could not load {AP_FILE}: {e} — AP columns will be blank.")
        return None


def load_labels(path: Path) -> pd.DataFrame | None:
    """
    Load per_label_results.csv.
    Expected columns:
        model, split, file_name, object_name, class_id,
        width_px, height_px, gt_width_px, gt_height_px, iou50,
        % area, gt % area
    """
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        # Keep original column names (% area has special chars)
        df.columns = [c.strip() for c in df.columns]
        for col in ("width_px", "height_px", "% area", "gt % area", "iou50"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"  [debug] Could not load {LABEL_FILE}: {e} — "
              "label/area columns will be blank.")
        return None


# ─────────────────────────────────────────────────────────────────
#  SUMMARY BUILDER
# ─────────────────────────────────────────────────────────────────

def build_summary(
    raw_df:    pd.DataFrame,
    ap_df:     pd.DataFrame | None,
    label_df:  pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Produce one summary row per (model, split, confidence_threshold,
    object_name, object_id).
    """

    # ── Determine group keys ───────────────────────────────────────────────
    conf_col = next(
        (c for c in ("confidence_threshold", "confidence") if c in raw_df.columns),
        None,
    )
    id_col = next(
        (c for c in ("object_id", "class_id") if c in raw_df.columns),
        None,
    )

    group_keys = ["model", "split"]
    if conf_col:
        group_keys.append(conf_col)
    group_keys.append("object_name")
    if id_col:
        group_keys.append(id_col)

    # ── Base aggregation from raw ──────────────────────────────────────────
    records = []

    for group_vals, grp in raw_df.groupby(group_keys, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        rec: dict = {}
        for key, val in zip(group_keys, group_vals):
            # Rename to clean output column names
            out_key = {
                "confidence_threshold": "confidence_threshold",
                "confidence":           "confidence_threshold",
                "object_id":            "class_id",
                "class_id":             "class_id",
            }.get(key, key)
            rec[out_key] = val

        rec["object_count"] = len(grp)

        # avg iou50
        try:
            rec["avg_iou50"] = safe_mean(grp["iou50"])
        except Exception as e:
            warn_missing_col("iou50", REQUIRED_FILE)
            rec["avg_iou50"] = None

        # avg iou50-95
        try:
            col_95 = next(
                (c for c in ("iou50_95", "iou50-95") if c in grp.columns), None
            )
            rec["avg_iou50_95"] = safe_mean(grp[col_95]) if col_95 else None
            if col_95 is None:
                warn_missing_col("iou50_95", REQUIRED_FILE)
        except Exception as e:
            warn_missing_col("iou50_95", REQUIRED_FILE)
            rec["avg_iou50_95"] = None

        records.append(rec)

    summary = pd.DataFrame(records)

    # ── Join AP columns ────────────────────────────────────────────────────
    if ap_df is not None:
        try:
            ap_df = ap_df.copy()
            # Normalise column names to lowercase/underscores
            ap_df.columns = [c.strip().lower().replace("-", "_")
                              for c in ap_df.columns]

            # Rename common variants
            if "object_id" in ap_df.columns and "class_id" not in ap_df.columns:
                ap_df = ap_df.rename(columns={"object_id": "class_id"})
            if "name" in ap_df.columns and "object_name" not in ap_df.columns:
                ap_df = ap_df.rename(columns={"name": "object_name"})

            # Normalise object_name strings for matching (strip whitespace)
            if "object_name" in ap_df.columns:
                ap_df["object_name"] = ap_df["object_name"].astype(str).str.strip()
            summary["object_name"] = summary["object_name"].astype(str).str.strip()

            # Drop any pre-existing blank AP columns so the merge doesn't dupe them
            for _col in ("AP50", "AP50_95", "ap50", "ap50_95"):
                if _col in summary.columns:
                    summary = summary.drop(columns=[_col])

            ap_cols = [c for c in ("ap50", "ap50_95") if c in ap_df.columns]
            if not ap_cols:
                print(f"  [debug] No ap50/ap50_95 columns found in {AP_FILE}.")
                print(f"  [debug] Columns present: {list(ap_df.columns)}")
            else:
                # Try progressively looser join keys until we get at least one match.
                # AP is one value per class per model/split — never join on confidence.
                key_attempts = [
                    ["model", "split", "object_name", "class_id"],
                    ["model", "split", "object_name"],
                    ["model", "object_name"],
                    ["object_name"],
                ]
                merged = None
                used_keys = None
                for keys in key_attempts:
                    valid_keys = [k for k in keys
                                  if k in ap_df.columns and k in summary.columns]
                    if len(valid_keys) < len(keys):
                        continue   # a required key is missing from one side
                    ap_sub = ap_df[valid_keys + ap_cols].drop_duplicates(subset=valid_keys)
                    candidate = summary.merge(ap_sub, on=valid_keys, how="left")
                    filled = candidate["ap50"].notna().sum()
                    if filled > 0:
                        merged    = candidate
                        used_keys = valid_keys
                        break

                if merged is None or merged["ap50"].isna().all():
                    print(f"  [debug] AP merge produced all-NaN across all key attempts.")
                    print(f"  [debug] AP file object_name sample: "
                          f"{ap_df['object_name'].unique()[:5] if 'object_name' in ap_df.columns else 'N/A'}")
                    print(f"  [debug] Summary object_name sample: "
                          f"{summary['object_name'].unique()[:5]}")
                    print(f"  [debug] Dropping AP columns from output.")
                    # Do not add AP columns at all — cleaner than all-NaN
                else:
                    print(f"  AP joined on keys: {used_keys} "
                          f"({merged['ap50'].notna().sum()} rows filled)")
                    summary = merged.rename(columns={"ap50": "AP50", "ap50_95": "AP50_95"})

        except Exception as e:
            print(f"  [debug] Failed to join AP data: {e} -- dropping AP columns.")

    # ── Join label / area columns ──────────────────────────────────────────
    if label_df is not None:
        try:
            label_df = label_df.copy()

            # Normalise model / split / object_name for joining
            for col in ("model", "split", "object_name"):
                if col not in label_df.columns:
                    warn_missing_col(col, LABEL_FILE)

            # class_id column in label file may be 'class_id'
            if "class_id" not in label_df.columns and "object_id" in label_df.columns:
                label_df = label_df.rename(columns={"object_id": "class_id"})

            # Compute pixel area column
            area_computed = False
            if "width_px" in label_df.columns and "height_px" in label_df.columns:
                label_df["_area_px"] = (
                    pd.to_numeric(label_df["width_px"],  errors="coerce") *
                    pd.to_numeric(label_df["height_px"], errors="coerce")
                )
                area_computed = True
            else:
                print(f"  [debug] width_px / height_px missing from {LABEL_FILE} — "
                      "area_px columns will be blank.")

            # Check % area columns
            has_pct_area    = "% area"    in label_df.columns
            has_gt_pct_area = "gt % area" in label_df.columns
            has_iou50       = "iou50"     in label_df.columns
            if not has_pct_area:
                print(f"  [debug] '% area' column missing from {LABEL_FILE} — "
                      "min/avg % area will be blank.")
            if not has_iou50:
                print(f"  [debug] 'iou50' column missing from {LABEL_FILE} — "
                      "% area stats will NOT be filtered to TP labels only.")

            # Group label_df by the same keys as summary
            label_group_keys = ["model", "split", "object_name"]
            if "confidence_threshold" in label_df.columns and \
               "confidence_threshold" in summary.columns:
                label_group_keys.append("confidence_threshold")
            if "class_id" in label_df.columns and "class_id" in summary.columns:
                label_group_keys.append("class_id")

            label_agg_rows = []
            for grp_keys, grp in label_df.groupby(
                label_group_keys, dropna=False
            ):
                if not isinstance(grp_keys, tuple):
                    grp_keys = (grp_keys,)

                agg: dict = {}
                for k, v in zip(label_group_keys, grp_keys):
                    agg[k] = v

                # TP subset: iou50 >= 0.5 — keeps FP boxes from
                # corrupting % area min/avg with arbitrarily-sized predictions.
                # Note: iou50 column stores the raw IoU value (0.0–1.0),
                # not a binary flag, so we threshold at 0.5 not == 1.0.
                if has_iou50:
                    tp_mask = pd.to_numeric(grp["iou50"], errors="coerce") >= 0.5
                    grp_tp  = grp[tp_mask]
                else:
                    grp_tp  = grp   # fallback: use all rows

                if area_computed:
                    agg["min_area_px"] = safe_min(grp["_area_px"])
                    agg["max_area_px"] = safe_max(grp["_area_px"])
                    agg["avg_area_px"] = safe_mean(grp["_area_px"])
                else:
                    agg["min_area_px"] = None
                    agg["max_area_px"] = None
                    agg["avg_area_px"] = None

                if has_pct_area:
                    try:
                        # Convert 0-100 to 0.0-1.0; use TP-only rows.
                        pct = pd.to_numeric(grp_tp["% area"], errors="coerce") / 100.0

                        # min_pct_area: TP labels with area > 0.10
                        min_mask = pct > 0.10
                        agg["min_pct_area"] = safe_min(pct[min_mask])

                        # avg_pct_area: TP labels between 0.01 and 0.99 exclusive
                        avg_mask = (pct > 0.01) & (pct < 0.99)
                        cropped  = pct[avg_mask]
                        agg["avg_pct_area"] = safe_mean(cropped) if len(cropped) > 0 else 0.0
                    except Exception as e:
                        print(f"  [debug] Error computing % area stats: {e}")
                        agg["min_pct_area"] = None
                        agg["avg_pct_area"] = None
                else:
                    agg["min_pct_area"] = None
                    agg["avg_pct_area"] = None

                # gt % area stats — ground-truth crop fractions for TP labels
                if has_gt_pct_area:
                    try:
                        gt_pct = pd.to_numeric(grp_tp["gt % area"], errors="coerce") / 100.0

                        # min_gt_pct_area: TP GT labels with area > 0.10
                        gt_min_mask = gt_pct > 0.10
                        agg["min_gt_pct_area"] = safe_min(gt_pct[gt_min_mask])

                        # avg_gt_pct_area: TP GT labels between 0.01 and 0.99
                        gt_avg_mask = (gt_pct > 0.01) & (gt_pct < 0.99)
                        gt_cropped  = gt_pct[gt_avg_mask]
                        agg["avg_gt_pct_area"] = safe_mean(gt_cropped) if len(gt_cropped) > 0 else 0.0
                    except Exception as e:
                        print(f"  [debug] Error computing gt % area stats: {e}")
                        agg["min_gt_pct_area"] = None
                        agg["avg_gt_pct_area"] = None
                else:
                    agg["min_gt_pct_area"] = None
                    agg["avg_gt_pct_area"] = None

                label_agg_rows.append(agg)

            label_agg = pd.DataFrame(label_agg_rows)
            summary   = summary.merge(label_agg, on=label_group_keys, how="left")

        except Exception as e:
            print(f"  [debug] Failed to join label/area data: {e} — "
                "area columns will be blank.")
            for col in ("min_area_px", "max_area_px", "avg_area_px",
                        "min_pct_area", "avg_pct_area",
                        "min_gt_pct_area", "avg_gt_pct_area"):
                summary[col] = None

    return summary


# ─────────────────────────────────────────────────────────────────
#  COLUMN ORDERING
# ─────────────────────────────────────────────────────────────────

COLUMN_ORDER = [
    "model",
    "split",
    "confidence_threshold",
    "object_name",
    "class_id",
    "object_count",
    "avg_iou50",
    "avg_iou50_95",
    "AP50",
    "AP50_95",
    "min_area_px",
    "max_area_px",
    "avg_area_px",
    "min_pct_area",
    "avg_pct_area",
    "min_gt_pct_area",
    "avg_gt_pct_area",
]


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    present  = [c for c in COLUMN_ORDER if c in df.columns]
    leftover = [c for c in df.columns  if c not in present]
    return df[present + leftover]


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        raise SystemExit(1)

    # ── Detect files ───────────────────────────────────────────────────────
    files = detect_files(directory)

    if files["raw"] is None:
        print(f"[ERROR] Required file not found: {directory / REQUIRED_FILE}")
        print("        Cannot produce a summary without it.")
        raise SystemExit(1)

    # ── Load files ─────────────────────────────────────────────────────────
    print("── Loading files ───────────────────────────────────────────")
    try:
        raw_df = load_raw(files["raw"])
        print(f"  Loaded {REQUIRED_FILE}: {len(raw_df)} rows")
    except Exception as e:
        print(f"[ERROR] Could not load {REQUIRED_FILE}: {e}")
        raise SystemExit(1)

    ap_df    = load_ap(files["ap"])
    label_df = load_labels(files["label"])

    if ap_df is not None:
        print(f"  Loaded {AP_FILE}: {len(ap_df)} rows")
    if label_df is not None:
        print(f"  Loaded {LABEL_FILE}: {len(label_df)} rows")

    # ── Build summary ──────────────────────────────────────────────────────
    print("\n── Building summary ────────────────────────────────────────")
    summary = build_summary(raw_df, ap_df, label_df)
    summary = reorder_columns(summary)

    # ── Write output ───────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else directory / "summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"\n✅ Summary CSV → {out_path}  ({len(summary)} rows, "
          f"{len(summary.columns)} columns)")
    print(f"   Columns: {list(summary.columns)}")


if __name__ == "__main__":
    main()