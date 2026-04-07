#	@file: test_augment_strategy.py
#	@brief: Spec tests for augment_strategy.py
#	If you use the --save argument, it will save all the images in this test
#
#	Section 1 — AugSpec dataclass boundary tests (no images needed)
#	Section 2 — apply_augmentations_to_dir integration tests using tmp_image / output_dir

import itertools
import pytest
import cv2
import numpy as np
import shutil
from pathlib import Path

from augmentation.augment_strategy import (
	AugSpec,
	apply_augmentations_to_dir,
	_combo_score,
	_combo_filename,
	_filename_seed,
	_find_label,
	MODE_ALL, MODE_RANDOM, MODE_CALC,
)
from augmentation.photometric_module import change_exposure
from augmentation.photometric_module_2 import contrast, motion_blur


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_images(directory: Path) -> int:
	"""Returns the number of image files in a directory."""
	return len([f for f in directory.iterdir()
	            if f.is_file() and f.suffix.lower() in {".jpg", ".png", ".jpeg"}])

def debug_strategy(images_dir, output_dir, mode, num, aug_specs):
	"""Print a summary of what apply_augmentations_to_dir produced."""
	out_files = sorted(output_dir.iterdir()) if output_dir.exists() else []
	print(f"\n--- strategy debug (mode={mode}, num={num}) ---")
	print(f"  specs: {[s.name for s in aug_specs]}")
	print(f"  input images: {count_images(images_dir)}")
	print(f"  output files: {len(out_files)}")
	for f in out_files:
		print(f"    {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 1 — AugSpec dataclass and helper boundary tests
#  (pure logic, no images required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAugSpec:
	"""Boundary tests for the AugSpec dataclass."""

	def test_minimal_valid_spec(self):
		"""AugSpec with only required fields creates correctly."""
		spec = AugSpec(name="exp", func=change_exposure, values=[0.5, 1.0, 1.5])
		assert spec.name == "exp"
		assert spec.func is change_exposure
		assert spec.values == [0.5, 1.0, 1.5]
		assert spec.label_func is None

	def test_spec_with_label_func(self):
		"""AugSpec with a label_func stores it correctly."""
		dummy_label_func = lambda lf, lo, amt, img: None
		spec = AugSpec(name="res", func=change_exposure, values=[1.0], label_func=dummy_label_func)
		assert spec.label_func is dummy_label_func

	def test_spec_empty_values_list(self):
		"""AugSpec accepts an empty values list (filtered out at apply time)."""
		spec = AugSpec(name="con", func=contrast, values=[])
		assert spec.values == []

	def test_spec_single_value(self):
		"""AugSpec with a single value stores correctly."""
		spec = AugSpec(name="mb", func=motion_blur, values=[1.5])
		assert spec.values == [1.5]

	def test_spec_neutral_only_values(self):
		"""AugSpec whose only value is 1.0 is valid (all combos will be neutral)."""
		spec = AugSpec(name="exp", func=change_exposure, values=[1.0])
		assert spec.values == [1.0]

	def test_spec_negative_value(self):
		"""AugSpec accepts negative amounts (no clamping at dataclass level)."""
		spec = AugSpec(name="exp", func=change_exposure, values=[-0.5, 1.0])
		assert -0.5 in spec.values

	def test_spec_large_value(self):
		"""AugSpec accepts amounts > 2.0."""
		spec = AugSpec(name="exp", func=change_exposure, values=[3.0])
		assert spec.values == [3.0]

	def test_spec_many_values(self):
		"""AugSpec handles a long value list."""
		vals = [round(0.1 * i, 1) for i in range(1, 21)]
		spec = AugSpec(name="con", func=contrast, values=vals)
		assert len(spec.values) == 20

	@pytest.mark.parametrize("name", ["exp", "con", "mblur", "gblur", "hue", "res", "sat", "x" * 50])
	def test_spec_various_names(self, name):
		"""AugSpec stores any string name."""
		spec = AugSpec(name=name, func=change_exposure, values=[1.0])
		assert spec.name == name

	def test_spec_equality(self):
		"""Two AugSpecs with the same fields are equal (dataclass default)."""
		a = AugSpec(name="exp", func=change_exposure, values=[0.5, 1.5])
		b = AugSpec(name="exp", func=change_exposure, values=[0.5, 1.5])
		assert a == b

	def test_spec_inequality_different_values(self):
		"""Two AugSpecs with different value lists are not equal."""
		a = AugSpec(name="exp", func=change_exposure, values=[0.5])
		b = AugSpec(name="exp", func=change_exposure, values=[1.5])
		assert a != b

	def test_spec_inequality_different_name(self):
		"""Two AugSpecs with different names are not equal."""
		a = AugSpec(name="exp", func=change_exposure, values=[1.0])
		b = AugSpec(name="con", func=change_exposure, values=[1.0])
		assert a != b


class TestComboScore:
	"""Boundary tests for _combo_score."""

	def test_neutral_combo_scores_zero(self):
		"""All-neutral combo (all values 1.0) has score 0.0."""
		assert _combo_score((1.0, 1.0, 1.0)) == pytest.approx(0.0)

	def test_single_extreme_value(self):
		"""Single value deviating from 1.0 gives (v-1)^2."""
		assert _combo_score((0.5,)) == pytest.approx(0.25)
		assert _combo_score((1.5,)) == pytest.approx(0.25)

	def test_symmetric_deviations_score_equally(self):
		"""Values equally far above and below 1.0 score the same."""
		assert _combo_score((0.7,)) == pytest.approx(_combo_score((1.3,)))

	def test_larger_deviation_scores_higher(self):
		"""A more extreme value gets a higher score."""
		assert _combo_score((0.5,)) > _combo_score((0.8,))

	def test_multi_spec_score_is_additive(self):
		"""Score for a multi-spec combo is the sum of each spec's squared deviation."""
		expected = (0.5 - 1.0) ** 2 + (1.5 - 1.0) ** 2
		assert _combo_score((0.5, 1.5)) == pytest.approx(expected)

	def test_zero_amount_score(self):
		"""amount=0 gives squared deviation of 1.0."""
		assert _combo_score((0.0,)) == pytest.approx(1.0)

	def test_score_ordering_matches_extremity(self):
		"""Combos are ordered by extremity when sorted by score descending."""
		combos = [(1.3,), (0.5,), (1.0,), (0.7,), (1.5,)]
		sorted_combos = sorted(combos, key=_combo_score, reverse=True)
		#	Most extreme first: 0.5 and 1.5 tie at 0.25, then 0.7/1.3 at 0.09, then 1.0 at 0
		assert sorted_combos[-1] == (1.0,)		#	neutral always last
		assert _combo_score(sorted_combos[0]) >= _combo_score(sorted_combos[1])


class TestComboFilename:
	"""Boundary tests for _combo_filename."""

	def _make_specs(self, names):
		return [AugSpec(name=n, func=change_exposure, values=[1.0]) for n in names]

	def test_single_spec_filename_format(self, tmp_path):
		"""Single spec produces expected stem format."""
		img = tmp_path / "img001.png"
		specs = self._make_specs(["exp"])
		combo = (0.615,)
		name, lbl = _combo_filename(img, specs, combo, 0)
		assert name.startswith("img001_exp")
		assert name.endswith(".png")
		assert lbl.endswith(".txt")
		assert "_0" in name		#	index appended

	def test_multi_spec_filename_contains_all_tokens(self, tmp_path):
		"""Multi-spec combo contains a token for every spec."""
		img = tmp_path / "test.jpg"
		specs = self._make_specs(["exp", "con", "mblur"])
		combo = (0.615, 0.7, 1.5)
		name, _ = _combo_filename(img, specs, combo, 2)
		assert "_exp" in name
		assert "_con" in name
		assert "_mblur" in name

	def test_repeat_tag_absent_when_zero(self, tmp_path):
		"""No _r0 tag when repeat=0."""
		img = tmp_path / "img.png"
		specs = self._make_specs(["exp"])
		name, _ = _combo_filename(img, specs, (0.5,), 0, repeat=0)
		assert "_r" not in name

	def test_repeat_tag_present_when_nonzero(self, tmp_path):
		"""_r{repeat} tag is included when repeat > 0."""
		img = tmp_path / "img.png"
		specs = self._make_specs(["exp"])
		name, _ = _combo_filename(img, specs, (0.5,), 5, repeat=2)
		assert "_r2" in name

	def test_index_changes_filename(self, tmp_path):
		"""Different ind values produce different filenames."""
		img = tmp_path / "img.png"
		specs = self._make_specs(["con"])
		name0, _ = _combo_filename(img, specs, (0.7,), 0)
		name1, _ = _combo_filename(img, specs, (0.7,), 1)
		assert name0 != name1

	def test_label_name_matches_image_stem(self, tmp_path):
		"""Label filename has the same stem as the image filename."""
		img = tmp_path / "img.png"
		specs = self._make_specs(["exp"])
		img_name, lbl_name = _combo_filename(img, specs, (0.5,), 0)
		assert Path(img_name).stem == Path(lbl_name).stem

	def test_neutral_value_trailing_zeros_stripped(self, tmp_path):
		"""Trailing zeros are stripped from value tokens (e.g. 1.500 -> 1.5)."""
		img = tmp_path / "img.png"
		specs = self._make_specs(["exp"])
		name, _ = _combo_filename(img, specs, (1.5,), 0)
		assert "1.500" not in name
		assert "1.5" in name


class TestFilenameSeed:
	"""Tests for _filename_seed reproducibility and uniqueness."""

	def test_same_filename_same_seed(self, tmp_path):
		"""Same filename always produces the same seed."""
		f = tmp_path / "image.png"
		assert _filename_seed(f) == _filename_seed(f)

	def test_different_filenames_different_seeds(self, tmp_path):
		"""Different filenames (very likely) produce different seeds."""
		a = tmp_path / "imageA.png"
		b = tmp_path / "imageB.png"
		#	Not guaranteed by hash, but virtually certain for different names
		assert _filename_seed(a) != _filename_seed(b)

	def test_seed_is_non_negative(self, tmp_path):
		"""Seed is always a non-negative integer (masked with 0xFFFFFFFF)."""
		f = tmp_path / "any_image.jpg"
		assert _filename_seed(f) >= 0


class TestFindLabel:
	"""Tests for _find_label."""

	def test_returns_none_when_labels_dir_is_none(self, tmp_path):
		"""Returns None when no labels directory is provided."""
		img = tmp_path / "img.png"
		assert _find_label(img, None) is None

	def test_returns_label_when_exists(self, tmp_path):
		"""Returns the label path when the matching .txt file exists."""
		labels_dir = tmp_path / "labels"
		labels_dir.mkdir()
		label = labels_dir / "img.txt"
		label.write_text("0 0.5 0.5 0.2 0.2\n")
		img = tmp_path / "img.png"
		result = _find_label(img, labels_dir)
		assert result == label

	def test_returns_none_when_label_missing(self, tmp_path):
		"""Returns None when the matching .txt file does not exist."""
		labels_dir = tmp_path / "labels"
		labels_dir.mkdir()
		img = tmp_path / "img.png"
		assert _find_label(img, labels_dir) is None


# ─────────────────────────────────────────────────────────────────────────────
#  Section 2 — apply_augmentations_to_dir integration tests
#  Uses tmp_image and output_dir fixtures from conftest.py
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_dir(tmp_path: Path, tmp_image: Path) -> Path:
	"""
	Copy tmp_image into a fresh subdirectory and return that directory.
	apply_augmentations_to_dir operates on directories, not single files.
	"""
	img_dir = tmp_path / "images"
	img_dir.mkdir()
	shutil.copy2(str(tmp_image), str(img_dir / tmp_image.name))
	return img_dir

def _basic_specs():
	"""A small set of AugSpecs that cover all photometric functions used in tests."""
	return [
		AugSpec(name="exp",  func=change_exposure, values=[0.7, 1.3]),
		AugSpec(name="con",  func=contrast,         values=[0.8, 1.2]),
	]

def _single_spec():
	"""One AugSpec with two non-neutral values."""
	return [AugSpec(name="exp", func=change_exposure, values=[0.7, 1.3])]

def _neutral_spec():
	"""One AugSpec whose only value is 1.0 — everything is a neutral combo."""
	return [AugSpec(name="exp", func=change_exposure, values=[1.0])]


# ── Mode: all ─────────────────────────────────────────────────────────────────

def test_all_mode_correct_image_count(tmp_image, output_dir, tmp_path, error_csv_writer, request):
	"""all mode produces one image per non-neutral combo."""
	img_dir  = _make_image_dir(tmp_path, tmp_image)
	out_dir  = output_dir / "all_basic"
	specs    = _basic_specs()

	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_ALL)

	#	2 specs × 2 values each = 4 combos total, minus 1 neutral (1.0,1.0) = 3
	all_combos   = list(itertools.product(*[s.values for s in specs]))
	neutral      = tuple(1.0 for _ in specs)
	expected_out = len([c for c in all_combos if c != neutral])

	actual = count_images(out_dir)
	debug_strategy(img_dir, out_dir, MODE_ALL, None, specs)

	if actual != expected_out:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         test node: {request.node.name}")

	error_csv_writer(str(tmp_image.name), "all_mode_count", expected_out, actual)
	assert actual == expected_out

def test_all_mode_neutral_only_produces_no_output(tmp_image, output_dir, tmp_path, request):
	"""all mode with only-neutral values produces zero output images."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "all_neutral"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _neutral_spec(), mode=MODE_ALL)

	actual = count_images(out_dir) if out_dir.exists() else 0
	debug_strategy(img_dir, out_dir, MODE_ALL, None, _neutral_spec())

	if actual != 0:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         test node: {request.node.name}")

	assert actual == 0

def test_all_mode_output_images_readable(tmp_image, output_dir, tmp_path, request):
	"""all mode output images can be read back by cv2."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "all_readable"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _single_spec(), mode=MODE_ALL)

	if out_dir.exists():
		for f in out_dir.iterdir():
			if f.suffix.lower() in {".jpg", ".png", ".jpeg"}:
				img = cv2.imread(str(f))
				if img is None:
					print(f"\n  [FAIL] could not read: {f}")
					print(f"         source image: {tmp_image}")
					print(f"         test node: {request.node.name}")
				assert img is not None, f"cv2 could not read output: {f}"

def test_all_mode_output_same_shape_as_input(tmp_image, output_dir, tmp_path, request):
	"""all mode output images have the same shape as the source image."""
	img_dir     = _make_image_dir(tmp_path, tmp_image)
	out_dir     = output_dir / "all_shape"
	source_img  = cv2.imread(str(tmp_image))
	source_shape = source_img.shape

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _single_spec(), mode=MODE_ALL)

	if out_dir.exists():
		for f in out_dir.iterdir():
			if f.suffix.lower() in {".jpg", ".png", ".jpeg"}:
				out_img = cv2.imread(str(f))
				if out_img.shape != source_shape:
					print(f"\n  [FAIL] shape mismatch: {f}")
					print(f"         expected {source_shape}, got {out_img.shape}")
					print(f"         test node: {request.node.name}")
				assert out_img.shape == source_shape


# ── Mode: random ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num", [1, 2, 3])
def test_random_mode_correct_image_count(tmp_image, output_dir, tmp_path, num, error_csv_writer, request):
	"""random mode produces exactly NUM output images per original."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / f"random_{num}"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _basic_specs(), mode=MODE_RANDOM, num=num)

	actual = count_images(out_dir)
	debug_strategy(img_dir, out_dir, MODE_RANDOM, num, _basic_specs())

	if actual != num:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         expected {num}, got {actual}")
		print(f"         test node: {request.node.name}")

	error_csv_writer(str(tmp_image.name), "random_mode_count", num, actual)
	assert actual == num

def test_random_mode_no_neutral_combos(tmp_image, output_dir, tmp_path, request):
	"""random mode never writes a copy identical to the original (all-neutral combo)."""
	img_dir     = _make_image_dir(tmp_path, tmp_image)
	out_dir     = output_dir / "random_no_neutral"
	source_img  = cv2.imread(str(tmp_image))
	specs       = _basic_specs()

	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_RANDOM, num=3)

	if out_dir.exists():
		for f in out_dir.iterdir():
			if f.suffix.lower() in {".jpg", ".png", ".jpeg"}:
				out_img = cv2.imread(str(f))
				diff    = np.abs(source_img.astype(np.float32) - out_img.astype(np.float32))
				is_identical = diff.mean() < 0.01
				if is_identical:
					print(f"\n  [FAIL] output is identical to original: {f}")
					print(f"         test node: {request.node.name}")
				assert not is_identical, f"Output is identical to original: {f}"

def test_random_mode_reproducible(tmp_image, output_dir, tmp_path, request):
	"""random mode produces the same set of output filenames when run twice."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_a   = output_dir / "random_repro_a"
	out_b   = output_dir / "random_repro_b"

	apply_augmentations_to_dir(img_dir, out_a, None, None, _basic_specs(), mode=MODE_RANDOM, num=3)
	apply_augmentations_to_dir(img_dir, out_b, None, None, _basic_specs(), mode=MODE_RANDOM, num=3)

	names_a = sorted(f.name for f in out_a.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})
	names_b = sorted(f.name for f in out_b.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})

	if names_a != names_b:
		print(f"\n  [FAIL] outputs differ between runs")
		print(f"         run A: {names_a}")
		print(f"         run B: {names_b}")
		print(f"         test node: {request.node.name}")

	assert names_a == names_b

def test_random_mode_no_duplicate_outputs(tmp_image, output_dir, tmp_path, request):
	"""random mode produces distinct output images (no duplicates)."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "random_unique"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _basic_specs(), mode=MODE_RANDOM, num=3)

	names = [f.name for f in out_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
	if len(names) != len(set(names)):
		print(f"\n  [FAIL] duplicate filenames in output: {names}")
		print(f"         test node: {request.node.name}")

	assert len(names) == len(set(names))

def test_random_mode_clamps_num_to_max_combos(tmp_image, output_dir, tmp_path, request):
	"""random mode clamps NUM to the number of unique non-neutral combos."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "random_clamp"
	specs   = _single_spec()		#	values=[0.7, 1.3] -> 2 non-neutral combos

	#	Request 99, but only 2 unique non-neutral combos exist
	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_RANDOM, num=99)

	actual = count_images(out_dir)
	debug_strategy(img_dir, out_dir, MODE_RANDOM, 99, specs)

	if actual > 2:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         expected <= 2, got {actual}")
		print(f"         test node: {request.node.name}")

	assert actual <= 2


# ── Mode: calc ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num", [1, 2, 3])
def test_calc_mode_correct_image_count(tmp_image, output_dir, tmp_path, num, error_csv_writer, request):
	"""calc mode produces exactly NUM output images per original."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / f"calc_{num}"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, _basic_specs(), mode=MODE_CALC, num=num)

	actual = count_images(out_dir)
	debug_strategy(img_dir, out_dir, MODE_CALC, num, _basic_specs())

	if actual != num:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         expected {num}, got {actual}")
		print(f"         test node: {request.node.name}")

	error_csv_writer(str(tmp_image.name), "calc_mode_count", num, actual)
	assert actual == num

def test_calc_mode_deterministic(tmp_image, output_dir, tmp_path, request):
	"""calc mode produces identical output filenames on repeated runs."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_a   = output_dir / "calc_det_a"
	out_b   = output_dir / "calc_det_b"

	apply_augmentations_to_dir(img_dir, out_a, None, None, _basic_specs(), mode=MODE_CALC, num=3)
	apply_augmentations_to_dir(img_dir, out_b, None, None, _basic_specs(), mode=MODE_CALC, num=3)

	names_a = sorted(f.name for f in out_a.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})
	names_b = sorted(f.name for f in out_b.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})

	if names_a != names_b:
		print(f"\n  [FAIL] calc is not deterministic")
		print(f"         run A: {names_a}")
		print(f"         run B: {names_b}")
		print(f"         test node: {request.node.name}")

	assert names_a == names_b

def test_calc_mode_first_output_is_most_extreme(tmp_image, output_dir, tmp_path, request):
	"""calc mode: the first output combo has the highest _combo_score."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "calc_extreme"
	specs   = _basic_specs()

	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_CALC, num=3)

	#	Recompute the expected ranked combos the same way _apply_calc does
	value_lists = [s.values for s in specs]
	neutral     = tuple(1.0 for _ in specs)
	all_combos  = [c for c in itertools.product(*value_lists) if c != neutral]
	all_combos.sort(key=_combo_score, reverse=True)

	expected_first_score = _combo_score(all_combos[0])

	#	The first output file (lowest ind in filename) should correspond to the highest score
	out_files = sorted(
		[f for f in out_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}],
		key=lambda f: int(f.stem.rsplit("_", 1)[-1])	#	sort by trailing index
	)

	if not out_files:
		print(f"\n  [FAIL] no output files found")
		print(f"         test node: {request.node.name}")
		assert False, "No output files found"

	#	The combo embedded in the first filename should be the most extreme one
	first_file = out_files[0].name
	first_combo = all_combos[0]
	for spec, val in zip(specs, first_combo):
		val_str = f"{val:.3f}".rstrip("0").rstrip(".")
		if val_str not in first_file:
			print(f"\n  [FAIL] most extreme combo not in first output filename")
			print(f"         first file: {first_file}")
			print(f"         expected combo: {first_combo}")
			print(f"         test node: {request.node.name}")
		assert val_str in first_file

def test_calc_mode_wraps_when_num_exceeds_combos(tmp_image, output_dir, tmp_path, request):
	"""calc mode wraps combos with _r{repeat} tag when NUM > unique combos."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "calc_wrap"
	specs   = _single_spec()	#	values=[0.7, 1.3] -> 2 non-neutral combos

	#	Request 5 images: will need to wrap after combo 2, producing _r1 files
	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_CALC, num=5)

	actual   = count_images(out_dir)
	out_names = [f.name for f in out_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
	has_repeat_tag = any("_r" in n for n in out_names)

	debug_strategy(img_dir, out_dir, MODE_CALC, 5, specs)

	if actual != 5:
		print(f"\n  [FAIL] image: {tmp_image}")
		print(f"         expected 5, got {actual}")
		print(f"         test node: {request.node.name}")
	if not has_repeat_tag:
		print(f"\n  [FAIL] no _r{{repeat}} tag found in wrapped outputs")
		print(f"         files: {out_names}")
		print(f"         test node: {request.node.name}")

	assert actual == 5
	assert has_repeat_tag


# ── Label handling ────────────────────────────────────────────────────────────

def test_labels_are_copied_for_photometric_augs(tmp_image, output_dir, tmp_path, request):
	"""Photometric augs copy the label file alongside each output image."""
	img_dir   = _make_image_dir(tmp_path, tmp_image)
	lbl_dir   = tmp_path / "labels"
	lbl_dir.mkdir()
	lbl_out   = output_dir / "labels_copy"
	img_out   = output_dir / "images_copy"

	#	Write a dummy label file matching tmp_image
	lbl_file = lbl_dir / f"{tmp_image.stem}.txt"
	lbl_file.write_text("0 0.5 0.5 0.2 0.2\n")

	apply_augmentations_to_dir(
		img_dir, img_out, lbl_dir, lbl_out,
		_single_spec(), mode=MODE_CALC, num=2
	)

	img_count = count_images(img_out)
	lbl_count = len(list(lbl_out.glob("*.txt"))) if lbl_out.exists() else 0

	if img_count != lbl_count:
		print(f"\n  [FAIL] image count {img_count} != label count {lbl_count}")
		print(f"         source image: {tmp_image}")
		print(f"         test node: {request.node.name}")

	assert img_count == lbl_count, f"Image count {img_count} != label count {lbl_count}"

def test_no_labels_dir_does_not_crash(tmp_image, output_dir, tmp_path, request):
	"""apply_augmentations_to_dir works fine when labels_dir=None."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "no_labels"

	#	Should not raise
	apply_augmentations_to_dir(img_dir, out_dir, None, None, _single_spec(), mode=MODE_ALL)

	assert count_images(out_dir) > 0


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_aug_specs_produces_no_output(tmp_image, output_dir, tmp_path, request):
	"""Passing an empty aug_specs list produces no output."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "empty_specs"

	apply_augmentations_to_dir(img_dir, out_dir, None, None, [], mode=MODE_CALC, num=3)

	actual = count_images(out_dir) if out_dir.exists() else 0
	assert actual == 0

def test_all_empty_value_lists_produces_no_output(tmp_image, output_dir, tmp_path, request):
	"""AugSpecs with all-empty value lists are filtered and produce no output."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "empty_values"
	specs   = [
		AugSpec(name="exp", func=change_exposure, values=[]),
		AugSpec(name="con", func=contrast,         values=[]),
	]

	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode=MODE_CALC, num=3)

	actual = count_images(out_dir) if out_dir.exists() else 0
	assert actual == 0

def test_unknown_mode_falls_back_to_all(tmp_image, output_dir, tmp_path, request):
	"""An unrecognised mode string falls back to 'all' without raising."""
	img_dir = _make_image_dir(tmp_path, tmp_image)
	out_dir = output_dir / "unknown_mode"
	specs   = _single_spec()

	apply_augmentations_to_dir(img_dir, out_dir, None, None, specs, mode="banana")

	#	Should have run as 'all': 2 non-neutral combos from values=[0.7, 1.3]
	actual = count_images(out_dir)
	assert actual == 2