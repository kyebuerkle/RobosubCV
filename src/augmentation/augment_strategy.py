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
		      f"augmented≈{augmented}  specs={[s.name for s in aug_specs]}")


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

	Each aug is applied in sequence on the same intermediate file, so the
	augmentations stack (e.g. exposure then contrast then blur on one image).
	A temporary file is used for intermediate steps.
	"""
	#	Chain augmentations through a temp file
	current_img = img_file
	import tempfile, os

	tmp_files = []
	try:
		for spec, val in zip(aug_specs, combo):
			if abs(val - 1.0) < 1e-9 and spec.label_func is None:
				#	Value is neutral AND not a geometric aug — skip writing a temp
				continue
			tmp = Path(tempfile.mktemp(suffix=img_file.suffix))
			tmp_files.append(tmp)
			spec.func(str(current_img), str(tmp), val)
			current_img = tmp

		#	Copy final result to destination
		shutil.copy2(str(current_img), str(img_out))

	finally:
		for tmp in tmp_files:
			if tmp.exists():
				tmp.unlink()

	#	Handle the label
	if label_file and label_out:
		#	Find the first geometric spec (one with a label_func) and its value
		geo_spec  = None
		geo_val   = 1.0
		for spec, val in zip(aug_specs, combo):
			if spec.label_func is not None:
				geo_spec = spec
				geo_val  = val
				break

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