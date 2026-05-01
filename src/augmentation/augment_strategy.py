#	@file: augment_strategy.py
#	@brief: Controls HOW augmentations are applied to a dataset directory.
#
#	Three modes (mirrors --augment CLI argument):
#
#	  all           — legacy behaviour: one output image per augmentation value,
#	                  per augmentation type, chained sequentially.
#	                  N values × M types = N^M total images per original.
#
#	  random NUM    — NUM augmented images per original image.
#	                  For each output image, every aug type independently gets
#	                  a randomly chosen value from its list (including 1.0 / no-op
#	                  values if present).  Combos that are all-neutral (every value
#	                  == 1.0, i.e. identical to the original) are discarded and
#	                  redrawn.  Duplicate combos are also discarded.  The RNG is
#	                  seeded from the source filename so results are reproducible
#	                  for the same dataset.
#
#	  calc NUM      — NUM augmented images per original, chosen to maximise
#	                  variation in a deterministic, repeatable way.
#	                  Each possible combination (one value per aug type) is scored
#	                  by the sum of squared deviations from 1.0 (the neutral point).
#	                  Combos are sorted highest-score first; the first NUM are used.
#	                  If NUM > total unique non-neutral combos the list wraps.
#
#	All modes write both the augmented image AND the corresponding label file
#	(copied for photometric augs, transformed for geometric augs like resize).
#
#	Public API used by augment_dataset.py:
#
#	  apply_augmentations_to_dir(
#	      images_dir, images_out_dir, labels_dir, labels_out_dir,
#	      aug_specs, mode, num,
#	  )
#
#	  where aug_specs is a list of AugSpec named tuples.

from __future__ import annotations

import itertools
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import augmentation.config as config

#	Supported image extensions (must match augment_dataset.py)
_IMAGE_EXTS = {".jpg", ".png", ".jpeg"}


# ──────────────────────────────────────────────────────────────────────────────
#  AugSpec: describes one augmentation type and its parameter list
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AugSpec:
	"""
	Describes a single augmentation type.

	:param name:         Short token used in output filenames (e.g. 'exp', 'con')
	:param func:         The photometric/geometric function with signature
	                     func(image_file, out_file, amount) -> out_file
	:param values:       List of amount values for this augmentation
	:param label_func:   Optional function to transform the label file alongside
	                     the image.  Signature:
	                     label_func(label_file, out_label, amount, img_file)
	                     If None, the label is simply copied.
	"""
	name:       str
	func:       Callable
	values:     List[float]
	label_func: Optional[Callable] = None


# ──────────────────────────────────────────────────────────────────────────────
#  Mode constants
# ──────────────────────────────────────────────────────────────────────────────

MODE_ALL    = "all"
MODE_RANDOM = "random"
MODE_CALC   = "calc"


# ──────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def apply_augmentations_to_dir(
	images_dir:     str | Path,
	images_out_dir: str | Path,
	labels_dir:     str | Path,
	labels_out_dir: str | Path,
	aug_specs:      List[AugSpec],
	mode:           str  = MODE_ALL,
	num:            int  = 1,
):
	"""
	Apply augmentations to all images in images_dir and write results to
	images_out_dir, copying / transforming matching labels in parallel.

	:param images_dir:     Source image directory
	:param images_out_dir: Destination image directory
	:param labels_dir:     Source label directory (may not exist for unlabelled splits)
	:param labels_out_dir: Destination label directory
	:param aug_specs:      List of AugSpec objects (one per active augmentation type)
	:param mode:           'all' | 'random' | 'calc'
	:param num:            Number of augmented images per original (used by random/calc)
	"""
	images_dir     = Path(images_dir).resolve()
	images_out_dir = Path(images_out_dir).resolve()
	labels_dir     = Path(labels_dir).resolve() if labels_dir else None
	labels_out_dir = Path(labels_out_dir).resolve() if labels_out_dir else None

	if not images_dir.is_dir():
		print(f"[augment_strategy] {images_dir} is not a directory, skipping.")
		return
	if not aug_specs:
		print("[augment_strategy] No augmentation specs provided, skipping.")
		return

	#	Filter out specs with empty value lists
	aug_specs = [s for s in aug_specs if s.values]
	if not aug_specs:
		print("[augment_strategy] All aug specs have empty value lists, skipping.")
		return

	replace_mode = (images_dir == images_out_dir)

	if replace_mode:
		temp_img_dir = images_dir.parent / f"{images_dir.name}_temp"
		temp_img_dir.mkdir(exist_ok=True)
		working_img_out = temp_img_dir
		if labels_dir and labels_out_dir and labels_dir == labels_out_dir:
			temp_lbl_dir = labels_dir.parent / f"{labels_dir.name}_temp"
			temp_lbl_dir.mkdir(exist_ok=True)
			working_lbl_out = temp_lbl_dir
		else:
			working_lbl_out = labels_out_dir
	else:
		images_out_dir.mkdir(parents=True, exist_ok=True)
		working_img_out = images_out_dir
		if labels_out_dir:
			labels_out_dir.mkdir(parents=True, exist_ok=True)
		working_lbl_out = labels_out_dir

	image_files = sorted(
		f for f in images_dir.iterdir()
		if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
	)

	if mode == MODE_ALL:
		_apply_all(image_files, working_img_out, labels_dir, working_lbl_out, aug_specs)
	elif mode == MODE_RANDOM:
		_apply_random(image_files, working_img_out, labels_dir, working_lbl_out, aug_specs, num)
	elif mode == MODE_CALC:
		_apply_calc(image_files, working_img_out, labels_dir, working_lbl_out, aug_specs, num)
	else:
		print(f"[augment_strategy] Unknown mode '{mode}', falling back to 'all'.")
		_apply_all(image_files, working_img_out, labels_dir, working_lbl_out, aug_specs)

	#	Always copy the original image (and its label) into the output directory.
	#	In replace_mode the originals are already in place — only needed when
	#	writing to a separate output directory.
	if not replace_mode:
		for img_file in image_files:
			shutil.copy2(str(img_file), str(working_img_out / img_file.name))
			label_file = _find_label(img_file, labels_dir)
			if label_file and working_lbl_out:
				shutil.copy2(str(label_file), str(working_lbl_out / label_file.name))

	#	Swap temp dirs back into place for replace_mode
	if replace_mode:
		_swap_replace(images_dir, temp_img_dir, image_files)
		if labels_dir and working_lbl_out != labels_out_dir:
			label_files = [labels_dir / f"{img.stem}.txt" for img in image_files
			               if (labels_dir / f"{img.stem}.txt").exists()]
			_swap_replace(labels_dir, temp_lbl_dir, label_files)

	total = len(image_files)
	augmented = total * (len(list(itertools.product(*[s.values for s in aug_specs]))) - 1
	                      if mode == MODE_ALL else num)
	if config.VERBOSE:
		print(f"[augment_strategy] mode={mode}  originals={total}  "
		      f"augmented={augmented}  total_output={total + augmented}  "
		      f"specs={[s.name for s in aug_specs]}")


# ──────────────────────────────────────────────────────────────────────────────
#  Mode: all
# ──────────────────────────────────────────────────────────────────────────────

def _apply_all(
	image_files:     List[Path],
	img_out_dir:     Path,
	labels_dir:      Optional[Path],
	labels_out_dir:  Optional[Path],
	aug_specs:       List[AugSpec],
):
	"""
	Legacy all-combos mode.  One output image for every combination of values
	across all aug specs.  Combos where every value == 1.0 (identical to the
	original) are skipped.
	"""
	value_lists = [s.values for s in aug_specs]
	all_combos  = list(itertools.product(*value_lists))
	neutral     = tuple(1.0 for _ in aug_specs)

	for img_file in image_files:
		if config.VVERBOSE:
			print(f"  [all] {img_file.name}")
		label_file = _find_label(img_file, labels_dir)

		for ind, combo in enumerate(all_combos):
			if combo == neutral:
				continue	#	don't write a copy of the original

			out_name, out_lbl_name = _combo_filename(img_file, aug_specs, combo, ind)
			_write_combo(
				img_file, img_out_dir / out_name,
				label_file, labels_out_dir / out_lbl_name if (labels_out_dir and label_file) else None,
				aug_specs, combo,
			)


# ──────────────────────────────────────────────────────────────────────────────
#  Mode: random
# ──────────────────────────────────────────────────────────────────────────────

def _apply_random(
	image_files:     List[Path],
	img_out_dir:     Path,
	labels_dir:      Optional[Path],
	labels_out_dir:  Optional[Path],
	aug_specs:       List[AugSpec],
	num:             int,
):
	"""
	Random mode.  Produces NUM augmented images per original.

	For each output image every aug type independently gets a randomly chosen
	value from its list.  Combos where all values == 1.0 (neutral / identical
	to the original) are discarded and redrawn.  Duplicate combos within the
	same source image are also discarded.

	The RNG is seeded from the source filename so results are reproducible for
	the same dataset but differ across images (avoiding correlated augmentations).
	"""
	neutral     = tuple(1.0 for _ in aug_specs)
	value_lists = [s.values for s in aug_specs]
	max_unique  = len(list(itertools.product(*value_lists))) - 1	#	exclude neutral

	if num > max_unique:
		if config.VERBOSE:
			print(f"[augment_strategy] random: NUM={num} > unique non-neutral combos "
			      f"({max_unique}), clamping to {max_unique}.")
		num = max_unique

	for img_file in image_files:
		if config.VVERBOSE:
			print(f"  [random] {img_file.name}")
		label_file = _find_label(img_file, labels_dir)

		#	Seed from filename so same dataset -> same combos every run
		rng   = random.Random(_filename_seed(img_file))
		seen  = set()
		ind   = 0
		tries = 0

		while ind < num and tries < num * 20:
			tries += 1
			combo = tuple(rng.choice(vals) for vals in value_lists)
			if combo == neutral or combo in seen:
				continue
			seen.add(combo)

			out_name, out_lbl_name = _combo_filename(img_file, aug_specs, combo, ind)
			_write_combo(
				img_file, img_out_dir / out_name,
				label_file, labels_out_dir / out_lbl_name if (labels_out_dir and label_file) else None,
				aug_specs, combo,
			)
			ind += 1


# ──────────────────────────────────────────────────────────────────────────────
#  Mode: calc
# ──────────────────────────────────────────────────────────────────────────────

def _apply_calc(
	image_files:     List[Path],
	img_out_dir:     Path,
	labels_dir:      Optional[Path],
	labels_out_dir:  Optional[Path],
	aug_specs:       List[AugSpec],
	num:             int,
):
	"""
	Calculated mode.  Produces NUM augmented images per original, chosen to
	maximise variation in a deterministic, repeatable way.

	Scoring:  each combo (one value per aug type) is scored as the sum of
	squared deviations from 1.0 (the neutral / no-change point).  Higher score
	= further from the original = more training value.  Combos are sorted
	highest-score first.  The first NUM are used; if NUM exceeds the number of
	unique non-neutral combos, the list wraps (repeating combos in order).

	The same config always produces the same output — no randomness involved.
	"""
	value_lists = [s.values for s in aug_specs]
	neutral     = tuple(1.0 for _ in aug_specs)

	all_combos = [
		c for c in itertools.product(*value_lists)
		if c != neutral
	]
	#	Sort: most extreme (highest deviation from 1.0) first
	all_combos.sort(key=_combo_score, reverse=True)

	if not all_combos:
		print("[augment_strategy] calc: no non-neutral combos available, skipping.")
		return

	for img_file in image_files:
		if config.VVERBOSE:
			print(f"  [calc] {img_file.name}")
		label_file = _find_label(img_file, labels_dir)

		for ind in range(num):
			combo = all_combos[ind % len(all_combos)]
			#	If wrapping, append a repeat counter so filenames stay unique
			repeat = ind // len(all_combos)
			out_name, out_lbl_name = _combo_filename(
				img_file, aug_specs, combo, ind, repeat=repeat
			)
			_write_combo(
				img_file, img_out_dir / out_name,
				label_file, labels_out_dir / out_lbl_name if (labels_out_dir and label_file) else None,
				aug_specs, combo,
			)


# ──────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _combo_score(combo: tuple) -> float:
	"""Sum of squared deviations from 1.0 — higher = more extreme."""
	return sum((v - 1.0) ** 2 for v in combo)


def _filename_seed(img_file: Path) -> int:
	"""Deterministic integer seed derived from the filename."""
	return hash(img_file.name) & 0xFFFFFFFF


def _find_label(img_file: Path, labels_dir: Optional[Path]) -> Optional[Path]:
	"""Returns the label .txt file for img_file, or None if not found."""
	if labels_dir is None:
		return None
	label = labels_dir / f"{img_file.stem}.txt"
	return label if label.exists() else None


def _combo_filename(
	img_file:  Path,
	aug_specs: List[AugSpec],
	combo:     tuple,
	ind:       int,
	repeat:    int = 0,
) -> Tuple[str, str]:
	"""
	Build output filenames for an image + label combo.

	Format:  {stem}_{name0}{val0}_{name1}{val1}_..._{ind}[_r{repeat}]{ext}

	Example: img001_exp0.615_con0.7_mblur1.5_0.png
	"""
	tokens = "".join(
		f"_{spec.name}{val:.3f}".rstrip("0").rstrip(".")
		for spec, val in zip(aug_specs, combo)
	)
	repeat_tag = f"_r{repeat}" if repeat > 0 else ""
	stem       = f"{img_file.stem}{tokens}_{ind}{repeat_tag}"
	img_name   = stem + img_file.suffix
	lbl_name   = stem + ".txt"
	return img_name, lbl_name


def _apply_array(img: np.ndarray, spec: "AugSpec", val: float) -> np.ndarray:
	"""
	Apply a single photometric augmentation entirely in memory.

	Avoids the imread/imwrite round-trip that the underlying module functions
	use, which is the main source of slowness when stacking multiple augs.
	Geometric specs (those with a label_func) are not handled here — they
	still need a file path for the label transform and use a temp file.

	All operations mirror the implementations in photometric_module.py and
	photometric_module_2.py exactly so results are identical.
	"""
	if abs(val - 1.0) < 1e-9:
		return img  # neutral — nothing to do

	name = spec.name

	if name == "exp" or name == "saturation":
		#	change_exposure: multiply all channels and clip
		if name == "saturation":
			#	change_saturation: scale S channel in HSV
			img_f = img.astype(np.float32) / 255.0
			hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
			hsv[:, :, 1] = np.clip(hsv[:, :, 1] * val, 0.0, 1.0)
			result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			return np.clip(result * 255.0, 0, 255).astype(np.uint8)
		else:
			return np.clip(img.astype(np.float32) * val, 0, 255).astype(np.uint8)

	elif name == "con":
		#	contrast: out = mean + (pixel - mean) * val
		img_f = img.astype(np.float32)
		mean  = img_f.mean(axis=(0, 1), keepdims=True)
		return np.clip(mean + (img_f - mean) * val, 0, 255).astype(np.uint8)

	elif name == "gblur":
		#	gaussian_blur: sigma = 10 * val
		sigma = 10.0 * val
		if sigma <= 0:
			return img
		ksize = int(sigma * 6)
		ksize = ksize if ksize % 2 == 1 else ksize + 1
		ksize = max(1, ksize)
		return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

	elif name == "mblur":
		#	motion_blur: streak length = 20 * val, default horizontal
		length = int(round(20.0 * val))
		if length <= 0:
			return img
		kernel = np.zeros((length, length), dtype=np.float32)
		kernel[length // 2, :] = 1.0 / length
		total = kernel.sum()
		if total > 0:
			kernel /= total
		return cv2.filter2D(img, -1, kernel)

	elif name == "hue":
		#	hue_shift: shift H channel in uint8 HSV (0-179), 1 unit = 2 degrees
		shift = int(round((val - 1.0) * 90.0))
		if shift == 0:
			return img
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
		hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
		return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

	else:
		#	Unknown photometric aug — fall back to file-based path
		import tempfile
		tmp = Path(tempfile.mktemp(suffix=".png"))
		try:
			cv2.imwrite(str(tmp), img)
			spec.func(str(tmp), str(tmp), val)
			return cv2.imread(str(tmp))
		finally:
			if tmp.exists():
				tmp.unlink()


def _write_combo(
	img_file:      Path,
	img_out:       Path,
	label_file:    Optional[Path],
	label_out:     Optional[Path],
	aug_specs:     List[AugSpec],
	combo:         tuple,
):
	"""
	Apply a combo of augmentation values to one image (and its label if present).

	Photometric augmentations are applied entirely in memory (read once, chain
	array operations, write once) to avoid the imread/imwrite overhead of the
	underlying module functions.  Geometric augmentations (those with a
	label_func, i.e. resize) still use a single temp file because they also
	need to transform the label file.
	"""
	import tempfile

	#	Separate photometric from geometric specs for this combo
	geo_spec = None
	geo_val  = 1.0
	for spec, val in zip(aug_specs, combo):
		if spec.label_func is not None:
			geo_spec = spec
			geo_val  = val
			break

	#	If there is a non-neutral geometric aug, apply it via temp file first
	#	so the label transform can also run, then read the result into memory.
	geo_tmp = None
	if geo_spec and abs(geo_val - 1.0) > 1e-9:
		try:
			geo_tmp = Path(tempfile.mktemp(suffix=img_file.suffix))
			geo_spec.func(str(img_file), str(geo_tmp), geo_val)
			img = cv2.imread(str(geo_tmp))
		except Exception as e:
			print(f"  [warn] geometric aug failed for {img_file.name}: {e}")
			img = cv2.imread(str(img_file))
	else:
		img = cv2.imread(str(img_file))

	if img is None:
		print(f"  [warn] could not read {img_file}, skipping combo")
		if geo_tmp and geo_tmp.exists():
			geo_tmp.unlink()
		return

	#	Apply all photometric augmentations in memory
	for spec, val in zip(aug_specs, combo):
		if spec.label_func is not None:
			continue  # already handled above
		if abs(val - 1.0) < 1e-9:
			continue  # neutral, skip
		img = _apply_array(img, spec, val)

	#	Write the final image once
	cv2.imwrite(str(img_out), img)

	#	Clean up geometric temp file
	if geo_tmp and geo_tmp.exists():
		geo_tmp.unlink()

	#	Handle the label
	if label_file and label_out:
		if geo_spec and abs(geo_val - 1.0) > 1e-9:
			geo_spec.label_func(str(label_file), str(label_out), geo_val, str(img_file))
		else:
			shutil.copy2(str(label_file), str(label_out))


def _swap_replace(original_dir: Path, temp_dir: Path, original_files: List[Path]):
	"""Delete originals, move temp contents back, remove temp dir."""
	for f in original_files:
		if f.exists():
			f.unlink()
	for temp_file in temp_dir.iterdir():
		shutil.move(str(temp_file), str(original_dir / temp_file.name))
	temp_dir.rmdir()