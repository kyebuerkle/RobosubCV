#!/usr/bin/env python3
"""
RoboSub Dataset Utility (menu-driven)
  - Extraction is 15 FPS
  - Only 640x480 frames are saved
  - Cap-to-500 action keeps frames in existing sequence order:
      extras = total_frames - 500
      split ordered frames into `extras` consecutive groups
      remove the blurriest frame in each group

Folder layout
  <dataset_root>/
    raw_videos/
    renamed_sorted/
      <class>/
        *.mp4 / *.mov / *.m4v

    640x480_frames/
      <class>/
        *.jpg

    rejected_frames/
      <class>/
        640x480/
          *.jpg
        
    all_objects/
        <class>__*.jpg  (copied from 640x480_frames after pruning/capping)

Requires:
  python 3.x
  opencv-python (cv2)
  pillow

Windows Powershell Package Installation:
    py -m pip install opencv-python
    py -m pip install pillow
"""

import re
import shutil
import uuid
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

OUT_SIZE = (640, 480)  # width, height

BLUR_THRESHOLD_640 = 60.0
EXTRACT_FPS = 15
MAX_FRAMES_PER_CLASS_640 = 500

PRUNED_MASS_FOLDER_NAME = "all_objects"


# -----------------------------
# UI / prompts
# -----------------------------
def ask_menu() -> int:
    print("\n=== RoboSub Dataset Menu ===")
    print("  1) Rename videos")
    print("  2) Extract frames")
    print("  3) Rename and extract frames")
    print("  4) Remove blurry frames (move to rejected_frames)")
    print("  5) Restore rejected frames")
    print(f"  6) Cap frames to {MAX_FRAMES_PER_CLASS_640} per class")
    print("  7) Delete all extracted frames")
    print("  8) Quit")

    while True:
        c = input("Choose [1-8]: ").strip()
        if c in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            return int(c)
        print("Enter 1, 2, 3, 4, 5, 6, 7, or 8.")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    while True:
        s = input(prompt + suffix).strip().lower()
        if not s:
            return default_yes
        if s in {"y", "yes"}:
            return True
        if s in {"n", "no"}:
            return False
        print("Enter y or n.")


def ask_class_scope(action_label: str, class_names: list[str]) -> list[str]:
    if not class_names:
        return []

    print(f"\nChoose classes for: {action_label}")
    print("  1) Run on all classes")
    print("  2) Choose which classes to include")
    print("  3) Choose which classes to skip")

    while True:
        choice = input("Choose [1-3]: ").strip()
        if choice in {"1", "2", "3"}:
            break
        print("Enter 1, 2, or 3.")

    if choice == "1":
        return class_names[:]

    print("\nAvailable classes:")
    for i, cname in enumerate(class_names, start=1):
        print(f"  {i}) {cname}")

    while True:
        raw = input("Enter class numbers separated by commas: ").strip()
        if not raw:
            print("Enter at least one class number.")
            continue

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        selected_indices = []
        valid = True

        for p in parts:
            if not p.isdigit():
                valid = False
                break
            idx = int(p)
            if idx < 1 or idx > len(class_names):
                valid = False
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

        if valid and selected_indices:
            break

        print("Enter valid class numbers like: 1,3,4")

    selected = [class_names[i - 1] for i in selected_indices]

    if choice == "2":
        return selected

    skipped = set(selected)
    return [cname for cname in class_names if cname not in skipped]


# -----------------------------
# Helpers
# -----------------------------
def norm_class(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch in {"_", "-"})[:64]


def list_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: (p.stat().st_mtime, p.name.lower()),
    )


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name.lower(),
    )


def format_hms(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:d}:{m:02d}:{s:02d}"


def video_duration_seconds(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    if fps > 1e-6 and frame_count > 0:
        duration = float(frame_count) / float(fps)
    else:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

    cap.release()
    if duration < 0 or duration > 1e9:
        return 0.0
    return duration


def compute_class_durations(renamed_sorted: Path):
    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )
    out = []

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue

        vids = list_videos(class_dir)
        if not vids:
            continue

        secs = 0.0
        for v in vids:
            secs += video_duration_seconds(v)

        out.append({
            "cname": cname,
            "class_dir": class_dir,
            "videos": vids,
            "seconds": secs,
        })

    return out


def print_duration_report(report):
    if not report:
        print("\nNo videos found to compute durations.")
        return

    total_secs = sum(r["seconds"] for r in report)
    total_vids = sum(len(r["videos"]) for r in report)

    print("\n=== Video Time Per Class ===")
    for r in report:
        secs = r["seconds"]
        vids = len(r["videos"])
        avg = (secs / vids) if vids else 0.0
        print(f"  {r['cname']}: {format_hms(secs)} total  ({vids} videos, avg {format_hms(avg)})")

    print(f"\nOVERALL: {format_hms(total_secs)} total across {total_vids} videos")


def print_fixed_frame_estimates(report, fps_out: float):
    if not report:
        return

    total_secs = sum(r["seconds"] for r in report)

    print(f"\n=== Estimated 640x480 Frames at {fps_out:g} FPS ===")
    header = "  class".ljust(22) + "time".rjust(10) + "est_frames".rjust(14)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in report:
        secs = r["seconds"]
        est = int(round(secs * fps_out))
        print(f"  {r['cname'][:20].ljust(22)}{format_hms(secs).rjust(10)}{est:14,}")

    total_est = int(round(total_secs * fps_out))
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL'.ljust(22)}{format_hms(total_secs).rjust(10)}{total_est:14,}")


def is_already_renamed(p: Path, cname: str) -> bool:
    pat = re.compile(rf"^{re.escape(cname)}_\d{{4}}\.(mp4|mov|m4v)$", re.IGNORECASE)
    return pat.match(p.name) is not None


def safe_inplace_rename(videos, cname: str):
    raw = [p for p in videos if not is_already_renamed(p, cname)]
    already = sorted([p for p in videos if is_already_renamed(p, cname)], key=lambda p: p.name.lower())

    if not raw:
        return already

    temp = []
    for p in raw:
        tmp = p.with_name(f"__tmp__{uuid.uuid4().hex}{p.suffix.lower()}")
        p.rename(tmp)
        temp.append(tmp)

    max_idx = 0
    idx_pat = re.compile(rf"^{re.escape(cname)}_(\d{{4}})\.(mp4|mov|m4v)$", re.IGNORECASE)
    for p in already:
        m = idx_pat.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    newly = []
    k = max_idx
    for p in sorted(temp, key=lambda x: x.name.lower()):
        k += 1
        final = p.with_name(f"{cname}_{k:04d}{p.suffix.lower()}")
        while final.exists():
            k += 1
            final = p.with_name(f"{cname}_{k:04d}{p.suffix.lower()}")
        p.rename(final)
        newly.append(final)

    return sorted(already + newly, key=lambda p: p.name.lower())


def next_frame_index_for_video(out_folder: Path, video_stem: str) -> int:
    if not out_folder.exists():
        return 1

    pat = re.compile(rf"^{re.escape(video_stem)}_(\d{{6}})\.(jpg|jpeg|png)$", re.IGNORECASE)
    max_idx = 0
    for p in out_folder.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def extract_frames_time_based_640_only(video: Path, fps_out: float, out_640: Path, start_index: int = 1) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("    !! can't open:", video.name)
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_out > native_fps:
        print(f"    note: requested fps {fps_out} > native {native_fps:.2f}; using {native_fps:.2f}")
        fps_out = native_fps

    interval_ms = 1000.0 / fps_out
    next_t = 0.0
    saved = 0
    idx = start_index

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t + 1e-6 >= next_t:
            fname = f"{video.stem}_{idx:06d}.jpg"
            small = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_640 / fname), small)

            saved += 1
            idx += 1
            next_t += interval_ms

    cap.release()
    return saved


def blur_score_laplacian(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def move_blurry_640_only(*, out_640: Path, rejected_root: Path, threshold: float) -> tuple[int, int]:
    rej_640 = rejected_root / "640x480"
    rej_640.mkdir(parents=True, exist_ok=True)

    kept = 0
    moved = 0

    for img_path in list_images(out_640):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        score = blur_score_laplacian(img)
        if score < threshold:
            moved += 1
            shutil.move(str(img_path), str(rej_640 / img_path.name))
        else:
            kept += 1

    return kept, moved


def restore_rejected_640_only(*, resized_root: Path, rejected_root: Path, selected_classes=None) -> int:
    if not rejected_root.exists():
        return 0

    selected_set = set(selected_classes) if selected_classes is not None else None
    moved = 0
    class_dirs = sorted([p for p in rejected_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    for cdir in class_dirs:
        cname = cdir.name
        if selected_set is not None and cname not in selected_set:
            continue

        src_640 = cdir / "640x480"
        if not src_640.exists():
            continue

        dst_640 = resized_root / cname
        dst_640.mkdir(parents=True, exist_ok=True)

        for p in list_images(src_640):
            dest = dst_640 / p.name
            if dest.exists():
                dest = dst_640 / f"{p.stem}__restored{p.suffix.lower()}"
            shutil.move(str(p), str(dest))
            moved += 1

    return moved


def count_images(folder: Path) -> int:
    return len(list_images(folder))


def count_videos(folder: Path) -> int:
    return len(list_videos(folder))


def collect_class_640_images(resized_root: Path, cname: str):
    """
    Collect all 640x480 images for a class from its single class folder.
    This preserves the existing extracted sequence as represented by filenames
    returned from list_images().
    """
    folder = resized_root / cname
    return list_images(folder)


def split_into_consecutive_groups(items, group_count: int):
    """
    Split items into group_count consecutive groups while preserving existing order.
    Group sizes differ by at most 1.
    """
    n = len(items)
    if group_count <= 0:
        return [items]

    base = n // group_count
    rem = n % group_count

    groups = []
    start = 0
    for i in range(group_count):
        size = base + (1 if i < rem else 0)
        end = start + size
        groups.append(items[start:end])
        start = end

    return groups


def cap_class_total_640_frames(*, resized_root: Path, rejected_root: Path, cname: str, max_keep: int) -> tuple[int, int]:
    """
    Cap TOTAL 640x480 frames for one class.

    Method:
      - Keep frames in existing sequence order
      - extras = total_before - max_keep
      - Split the ordered frames into `extras` consecutive groups
      - Remove the blurriest frame from each group

    This removes exactly enough frames to leave max_keep.
    """
    all_imgs = collect_class_640_images(resized_root, cname)
    total_before = len(all_imgs)

    if total_before <= max_keep:
        return total_before, 0

    extras = total_before - max_keep
    groups = split_into_consecutive_groups(all_imgs, extras)

    moved = 0

    for group in groups:
        if not group:
            continue

        worst_score = None
        worst_path = None

        for p in group:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                score = -1e18
            else:
                score = blur_score_laplacian(img)

            if worst_score is None or score < worst_score:
                worst_score = score
                worst_path = p

        if worst_path is not None and worst_path.exists():
            rej_640 = rejected_root / cname / "640x480"
            rej_640.mkdir(parents=True, exist_ok=True)

            dest = rej_640 / worst_path.name
            if dest.exists():
                dest = rej_640 / f"{worst_path.stem}__capped{worst_path.suffix.lower()}"

            shutil.move(str(worst_path), str(dest))
            moved += 1

    total_after = len(collect_class_640_images(resized_root, cname))
    return total_after, moved


def delete_all_extracted_640_frames(resized_root: Path, rejected_root: Path, mass_root: Path, selected_classes=None) -> int:
    removed = 0
    selected_set = set(selected_classes) if selected_classes is not None else None

    if resized_root.exists():
        for class_dir in [p for p in resized_root.iterdir() if p.is_dir()]:
            if selected_set is not None and class_dir.name not in selected_set:
                continue
            for img in list_images(class_dir):
                img.unlink()
                removed += 1

    if rejected_root.exists():
        for class_dir in [p for p in rejected_root.iterdir() if p.is_dir()]:
            if selected_set is not None and class_dir.name not in selected_set:
                continue
            rej_640 = class_dir / "640x480"
            if not rej_640.exists():
                continue

            for img in list_images(rej_640):
                img.unlink()
                removed += 1

    if mass_root.exists() and selected_set is None:
        for img in list_images(mass_root):
            img.unlink()
            removed += 1

    return removed


def rebuild_pruned_mass_folder(*, resized_root: Path, mass_root: Path) -> int:
    """
    Rebuild a single combined folder containing all currently kept 640x480
    frames across every class after pruning/capping.

    Files are copied (not moved) and renamed with their class prefix to avoid
    name collisions:
      <class>__<original_name>.jpg
    """
    if mass_root.exists():
        shutil.rmtree(mass_root)
    mass_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    class_dirs = sorted([p for p in resized_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue

        for img_path in list_images(class_dir):
            dest = mass_root / f"{cname}__{img_path.name}"
            shutil.copy2(str(img_path), str(dest))
            copied += 1

    return copied


def print_dataset_summary(*, renamed_sorted: Path, resized_root: Path, rejected_root: Path):
    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    rows = []
    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue

        videos_n = count_videos(class_dir)
        resized_frames_n = count_images(resized_root / cname)
        rejected_frames_n = count_images(rejected_root / cname / "640x480")

        if videos_n == 0 and resized_frames_n == 0 and rejected_frames_n == 0:
            continue

        rows.append((cname, videos_n, resized_frames_n, rejected_frames_n))

    print("\n=== Dataset Summary (per class) ===")
    if not rows:
        print("  (no data found)")
        return

    header = (
        "  class".ljust(22)
        + "videos".rjust(8)
        + "640_frames".rjust(12)
        + "rejected_640".rjust(14)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    tot_v = tot_640 = tot_r = 0
    for cname, v, f640, r in rows:
        tot_v += v
        tot_640 += f640
        tot_r += r
        print(
            f"  {cname[:20].ljust(22)}"
            f"{v:8d}"
            f"{f640:12d}"
            f"{r:14d}"
        )

    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL'.ljust(22)}"
        f"{tot_v:8d}"
        f"{tot_640:12d}"
        f"{tot_r:14d}"
    )


# -----------------------------
# Actions
# -----------------------------
def action_rename(renamed_sorted: Path):
    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )
    if not class_dirs:
        print("No class subfolders found in:", renamed_sorted)
        return

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue

        vids = list_videos(class_dir)
        if not vids:
            print(f"\nClass: {cname}  (no videos found)")
            continue

        print(f"\nClass: {cname}  ({len(vids)} videos found)")
        renamed_vids = safe_inplace_rename(vids, cname)
        print(f"  renamed/kept: {len(renamed_vids)} videos total")


def action_extract(renamed_sorted: Path, resized_root: Path):
    report = compute_class_durations(renamed_sorted)
    print_duration_report(report)
    print_fixed_frame_estimates(report, EXTRACT_FPS)

    wipe_frames = ask_yes_no("Wipe existing extracted 640x480 frames each run?", default_yes=True)
    if not report:
        return

    selected_classes = ask_class_scope(
        "extract frames",
        [r["cname"] for r in report],
    )
    if not selected_classes:
        print("No classes selected.")
        return

    selected_set = set(selected_classes)

    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue
        if cname not in selected_set:
            continue

        vids_sorted = sorted(list_videos(class_dir), key=lambda p: p.name.lower())
        if not vids_sorted:
            continue

        print(f"\nClass: {cname}  ({len(vids_sorted)} videos)")

        out_640 = resized_root / cname

        if wipe_frames and out_640.exists():
            shutil.rmtree(out_640)

        out_640.mkdir(parents=True, exist_ok=True)

        class_total = 0
        for v in vids_sorted:
            print(f"  video: {v.name}")

            if wipe_frames:
                start_idx = 1
            else:
                start_idx = next_frame_index_for_video(out_640, v.stem)

            n = extract_frames_time_based_640_only(
                v, EXTRACT_FPS, out_640, start_index=start_idx
            )
            class_total += n
            print(f"    frames saved: {n}")

        print(f"  ==> TOTAL 640x480 frames extracted for class '{cname}': {class_total}")


def action_rename_and_extract(renamed_sorted: Path, resized_root: Path):
    report = compute_class_durations(renamed_sorted)
    print_duration_report(report)
    print_fixed_frame_estimates(report, EXTRACT_FPS)

    wipe_frames = ask_yes_no("Wipe existing extracted 640x480 frames each run?", default_yes=True)

    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )
    if not class_dirs:
        print("No class subfolders found in:", renamed_sorted)
        return

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue

        vids = list_videos(class_dir)
        if not vids:
            continue

        print(f"\nClass: {cname}  ({len(vids)} videos found)")
        renamed_vids = safe_inplace_rename(vids, cname)
        print(f"  renamed/kept: {len(renamed_vids)} videos total")

        out_640 = resized_root / cname

        if wipe_frames and out_640.exists():
            shutil.rmtree(out_640)

        out_640.mkdir(parents=True, exist_ok=True)

        class_total = 0
        for v in renamed_vids:
            print(f"  video: {v.name}")

            if wipe_frames:
                start_idx = 1
            else:
                start_idx = next_frame_index_for_video(out_640, v.stem)

            n = extract_frames_time_based_640_only(
                v, EXTRACT_FPS, out_640, start_index=start_idx
            )
            class_total += n
            print(f"    frames saved: {n}")

        print(f"  ==> TOTAL 640x480 frames extracted for class '{cname}': {class_total}")


def action_blur_clean_640_only(renamed_sorted: Path, resized_root: Path, rejected_root: Path, mass_root: Path):
    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )
    if not class_dirs:
        print("No class subfolders found in:", renamed_sorted)
        return

    class_names = [norm_class(p.name) for p in class_dirs if norm_class(p.name)]
    selected_classes = ask_class_scope("remove blurry frames", class_names)
    if not selected_classes:
        print("No classes selected.")
        return

    selected_set = set(selected_classes)

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue
        if cname not in selected_set:
            continue

        out_640 = resized_root / cname
        print(f"\nClass: {cname}")

        if not out_640.exists():
            print("  no 640x480 frames folder found; skipping.")
            continue

        kept, moved = move_blurry_640_only(
            out_640=out_640,
            rejected_root=rejected_root / cname,
            threshold=BLUR_THRESHOLD_640,
        )
        print(f"  ==> Blur filter done (640 only): kept={kept}, moved_to_rejected={moved}")

    copied = rebuild_pruned_mass_folder(resized_root=resized_root, mass_root=mass_root)
    print(f"\nCombined kept-frames folder rebuilt: {mass_root}")
    print(f"  total files copied: {copied}")


def action_restore_rejected_640(resized_root: Path, rejected_root: Path):
    if not rejected_root.exists():
        print("No rejected frames folder found.")
        return

    class_names = sorted([p.name for p in rejected_root.iterdir() if p.is_dir()], key=str.lower)
    if not class_names:
        print("No rejected class folders found.")
        return

    selected_classes = ask_class_scope("restore rejected frames", class_names)
    if not selected_classes:
        print("No classes selected.")
        return

    moved = restore_rejected_640_only(
        resized_root=resized_root,
        rejected_root=rejected_root,
        selected_classes=selected_classes,
    )
    print(f"\nRestored rejected 640x480 frames moved back: {moved}")


def action_cap_640_frames(renamed_sorted: Path, resized_root: Path, rejected_root: Path, mass_root: Path):
    class_dirs = sorted(
        [p for p in renamed_sorted.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )
    if not class_dirs:
        print("No class subfolders found in:", renamed_sorted)
        return

    class_names = [norm_class(p.name) for p in class_dirs if norm_class(p.name)]
    selected_classes = ask_class_scope(f"cap frames to {MAX_FRAMES_PER_CLASS_640}", class_names)
    if not selected_classes:
        print("No classes selected.")
        return

    selected_set = set(selected_classes)

    print(f"\nCapping policy: keep {MAX_FRAMES_PER_CLASS_640} 640x480 frames TOTAL per class.")

    for class_dir in class_dirs:
        cname = norm_class(class_dir.name)
        if not cname:
            continue
        if cname not in selected_set:
            continue

        folder = resized_root / cname
        if not folder.exists():
            print(f"\nClass: {cname}")
            print("  no extracted frames folder found; skipping.")
            continue

        total_before = len(list_images(folder))
        if total_before == 0:
            print(f"\nClass: {cname}")
            print("  no extracted frames found; skipping.")
            continue

        if total_before <= MAX_FRAMES_PER_CLASS_640:
            print(f"\nClass: {cname}")
            print(f"  total={total_before} (<= {MAX_FRAMES_PER_CLASS_640}); nothing to do.")
            continue

        after, moved = cap_class_total_640_frames(
            resized_root=resized_root,
            rejected_root=rejected_root,
            cname=cname,
            max_keep=MAX_FRAMES_PER_CLASS_640,
        )

        print(f"\nClass: {cname}")
        print(f"  before={total_before}, after={after}, moved={moved}")

    copied = rebuild_pruned_mass_folder(resized_root=resized_root, mass_root=mass_root)
    print(f"\nCombined pruned folder rebuilt: {mass_root}")
    print(f"  total files copied: {copied}")


def action_delete_all_extracted_640(resized_root: Path, rejected_root: Path, mass_root: Path):
    class_names = sorted(
        {
            p.name for p in resized_root.iterdir() if p.is_dir()
        }.union(
            p.name for p in rejected_root.iterdir() if p.is_dir()
        ),
        key=str.lower,
    )

    if not class_names:
        print("No extracted or rejected class folders found.")
        return

    selected_classes = ask_class_scope("delete extracted frames", class_names)
    if not selected_classes:
        print("No classes selected.")
        return

    deleting_all = len(selected_classes) == len(class_names)
    if deleting_all:
        prompt = "Delete ALL extracted, rejected, and combined mass-folder 640x480 frames for all classes?"
    else:
        prompt = "Delete extracted and rejected 640x480 frames for the selected classes?"

    if not ask_yes_no(prompt, default_yes=False):
        print("Cancelled.")
        return

    removed = delete_all_extracted_640_frames(
        resized_root,
        rejected_root,
        mass_root,
        selected_classes=None if deleting_all else selected_classes,
    )

    if deleting_all:
        print(f"\nDeleted extracted/rejected/combined 640x480 frames: {removed}")
    else:
        copied = rebuild_pruned_mass_folder(resized_root=resized_root, mass_root=mass_root)
        print(f"\nDeleted extracted/rejected 640x480 frames for selected classes: {removed}")
        print(f"Rebuilt combined kept-frames folder: {mass_root}")
        print(f"  total files copied: {copied}")


# -----------------------------
# main loop
# -----------------------------
def main():
    dataset_root = Path(input('\nDataset root directory: ').strip().strip('"')).resolve()

    if not dataset_root.exists():
        print("Folder not found:", dataset_root)
        return

    renamed_sorted = dataset_root / "renamed_sorted"
    resized_root = dataset_root / "640x480_frames"
    rejected_root = dataset_root / "rejected_frames"
    mass_root = dataset_root / PRUNED_MASS_FOLDER_NAME

    if not renamed_sorted.exists():
        print('\nERROR: Could not find "renamed_sorted" inside:')
        print(dataset_root)
        return

    resized_root.mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)
    mass_root.mkdir(parents=True, exist_ok=True)

    print("\nDataset root   :", dataset_root)
    print("renamed_sorted :", renamed_sorted)
    print("640x480_frames :", resized_root)
    print("rejected_frames:", rejected_root)
    print("all_objects :", mass_root)

    while True:
        mode = ask_menu()

        if mode == 8:
            print("Quitting.")
            break

        if mode == 1:
            action_rename(renamed_sorted)
        elif mode == 2:
            action_extract(renamed_sorted, resized_root)
        elif mode == 3:
            action_rename_and_extract(renamed_sorted, resized_root)
        elif mode == 4:
            action_blur_clean_640_only(renamed_sorted, resized_root, rejected_root, mass_root)
        elif mode == 5:
            action_restore_rejected_640(resized_root, rejected_root)
        elif mode == 6:
            action_cap_640_frames(renamed_sorted, resized_root, rejected_root, mass_root)
        elif mode == 7:
            action_delete_all_extracted_640(resized_root, rejected_root, mass_root)

        print_dataset_summary(
            renamed_sorted=renamed_sorted,
            resized_root=resized_root,
            rejected_root=rejected_root,
        )

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()