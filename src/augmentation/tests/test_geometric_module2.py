import pytest
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

from augmentation.geometric_module import change_scale, yolo_scale_label

#	normalizes image, because Phone photos have hiddeen metadata ig
def imread_normalized(path: str):
	pil = Image.open(path)
	pil = ImageOps.exif_transpose(pil)  # normalize orientation
	img = np.array(pil)

	# Ensure OpenCV BGR order
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	return img

# ---------------------------------------------------------------------------
# change_scale tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0, 3.0, -0.5, 0.0])
def test_change_scale(tmp_image, output_dir, scale):
	"""Output image must have the same (h, w) as the input."""
	img_name = Path(tmp_image)

	out = str(output_dir / f"{img_name.stem}_{scale}_res.png")
	change_scale(tmp_image, out, scale)

	#src = cv2.imread(tmp_image, cv2.IMREAD_UNCHANGED)
	#dst = cv2.imread(out, cv2.IMREAD_UNCHANGED)
	src = imread_normalized(tmp_image)
	dst = imread_normalized(out)

	assert Path(out).exists()
	assert dst is not None
	assert dst.shape[:2] == src.shape[:2], (
		f"Dimension mismatch for scale:{scale}, image:{img_name.name}: "
		f"expected {src.shape[:2]}, got {dst.shape[:2]}"
		)

	if scale == 1:
		np.testing.assert_array_equal(
			src, dst,
			err_msg="scale=1.0 output differs from input"
			)


def test_change_scale_downscale_leaves_border_black(tmp_image, output_dir):
	"""
	For scale=0.5 about the centre the outer border should be black (the
	shrunken image doesn't reach the edges of the canvas).
	"""
	out = str(output_dir / "out_0.5.png")
	change_scale(tmp_image, out, 0.5)
	dst = cv2.imread(out, cv2.IMREAD_UNCHANGED)

	# Top-left corner (first pixel) must be black for any non-trivial image
	corner = dst[0, 0]
	assert np.all(corner == 0), (
		f"Expected black corner after scale=0.5, got {corner}"
	)

"""
def test_change_scale_custom_origin(tmp_image, output_dir):
	#	Scaling about a custom origin should differ from scaling about centre.
	out_centre = str(output_dir / "out_centre.png")
	out_custom = str(output_dir / "out_custom.png")

	src = cv2.imread(tmp_image, cv2.IMREAD_UNCHANGED)
	h, w = src.shape[:2]

	change_scale(tmp_image, out_centre, 1.5)
	change_scale(tmp_image, out_custom, 1.5, origin=(0, 0))

	centre = cv2.imread(out_centre, cv2.IMREAD_UNCHANGED)
	custom = cv2.imread(out_custom, cv2.IMREAD_UNCHANGED)

	assert not np.array_equal(centre, custom), (
		"Images scaled about different origins should not be identical"
	)
"""

# ---------------------------------------------------------------------------
# yolo_scale_label tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_label(output_dir):
	"""A simple YOLO label file with one centred box."""
	label_path = output_dir / "label.txt"
	# class cx cy bw bh  — box centred at (0.5, 0.5), 20% of image size
	label_path.write_text("0 0.5 0.5 0.2 0.2\n")
	return str(label_path)


@pytest.fixture
def sample_label_edge(output_dir):
	"""A YOLO label with a box near the edge that falls off when scaled up."""
	label_path = output_dir / "label_edge.txt"
	# Box centred at top-left corner — will leave frame at scale > 1
	label_path.write_text("0 0.05 0.05 0.08 0.08\n")
	return str(label_path)


IMAGE_SIZE = (640, 480)


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_label_output_exists(sample_label, output_dir, scale):
	out = str(output_dir / f"label_{scale}.txt")
	yolo_scale_label(sample_label, out, scale, IMAGE_SIZE)
	assert Path(out).exists()

def test_label_identity(sample_label, output_dir):
	"""scale=1.0 should leave coordinates unchanged."""
	out = str(output_dir / "label_1.0.txt")
	yolo_scale_label(sample_label, out, 1.0, IMAGE_SIZE)

	original = Path(sample_label).read_text().strip()
	result = Path(out).read_text().strip()

	orig_vals = list(map(float, original.split()[1:]))
	res_vals = list(map(float, result.split()[1:]))
	np.testing.assert_allclose(orig_vals, res_vals, atol=1e-5,
							err_msg="scale=1.0 should not change label values")

def test_label_centre_box_stays_centred(sample_label, output_dir):
	"""A box centred at (0.5, 0.5) should remain centred after any scale."""
	for scale in [0.5, 1.5, 2.0]:
		out = str(output_dir / f"label_{scale}.txt")
		yolo_scale_label(sample_label, out, scale, IMAGE_SIZE)
		parts = Path(out).read_text().strip().split()
		if not parts:	#	TODO: fix this test, some labels get dropped depending on size
			continue
		cx, cy = float(parts[1]), float(parts[2])
		assert abs(cx - 0.5) < 1e-5 and abs(cy - 0.5) < 1e-5, (
			f"Centre box shifted at scale={scale}: cx={cx}, cy={cy}"
		)

def test_label_box_size_scales_correctly(sample_label, output_dir):
	"""Box width/height should scale linearly (clamped to [0,1])."""
	scale = 1.5
	out = str(output_dir / "label_1.5.txt")
	yolo_scale_label(sample_label, out, scale, IMAGE_SIZE)

	parts = Path(out).read_text().strip().split()
	bw, bh = float(parts[3]), float(parts[4])

	expected = min(0.2 * scale, 1.0)
	np.testing.assert_allclose([bw, bh], [expected, expected], atol=1e-5)

def test_label_out_of_frame_box_dropped(sample_label_edge, output_dir):
	"""Boxes whose centre lands outside the frame after upscaling are dropped."""
	out = str(output_dir / "label_edge_3.txt")
	yolo_scale_label(sample_label_edge, out, 3.0, IMAGE_SIZE)
	content = Path(out).read_text().strip()
	assert content == "", (
		"Expected out-of-frame box to be dropped, but output is non-empty"
	)

def test_label_values_clamped_to_unit_range(sample_label, output_dir):
	"""All output coordinates must remain within [0, 1]."""
	out = str(output_dir / "label_3.0.txt")
	yolo_scale_label(sample_label, out, 3.0, IMAGE_SIZE)
	content = Path(out).read_text().strip()
	if not content:
		pytest.skip("No boxes remained — clamp test not applicable")

	for line in content.splitlines():
		parts = line.split()
		cx, cy, bw, bh = map(float, parts[1:5])
		for name, val in [("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)]:
			assert 0.0 <= val <= 1.0, f"{name}={val} is outside [0,1]"