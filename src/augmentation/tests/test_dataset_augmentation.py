#	@file: test_dataset_augmentation.py
#	@brief: tests dataset augmentations by running them and validating the format

import pytest
from general_lib import validate_yolov8_dataset
import pytest
import yaml
import cv2
import numpy as np
from pathlib import Path
from augmentation import yolo_change_exposure, yolo_change_saturation

SPLIT_DIRS = ['train', 'valid', 'test']
@pytest.fixture(params=["random", "middle", "dataset_root"])
def tmp_dataset(request, tmp_path, dataset_root):
	"""
	Create temperary dataset

	:return dataset path: the path to the dataset
	"""
	if request.param == "dataset_root":
		if dataset_root:
			return dataset_root
		else:
			return None
	else:
		dataset_dir = tmp_path / f"tmp_dataset_{request.param}"
		dataset_dir.mkdir()
		
		# Create structure
		for split in SPLIT_DIRS:
			(dataset_dir / split / 'images').mkdir(parents=True)
			(dataset_dir / split / 'labels').mkdir(parents=True)
			
			# Create image
			if request.param == "random":
				img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
			elif request.param == "middle":
				img = np.full(shape = (100, 100, 3), fill_value = 127)
			else:
				img = np.ones((100, 100, 3))
			
			cv2.imwrite(str(dataset_dir / split / 'images' / f'tmp_{split}.png'), img)
			
			# Create label
			rand_labels = [str(f) for f in np.random.rand(4).tolist()]
			(dataset_dir / split / 'labels' / f'tmp_{split}.txt').write_text(
				f"0 {" ".join(rand_labels)}\n"
				)
		
		# data.yaml
		data_yaml = {
			'names': ['label'],
			'nc': 1,
			'path': str(dataset_dir),
			'test': '../test/images',
			'train': '../train/images',
			'val': '../valid/images'
			}
		with open(dataset_dir / 'data.yaml', 'w') as f:
			yaml.dump(data_yaml, f, sort_keys=False)
		
		# README
		(dataset_dir / 'README.md').write_text(
			"tmp readme for tmp_dataset\nfake Lincense: yay TM 2.0\n"
			)
		
		return dataset_dir

AUGMENT_VALUES = [0.5, 1.0, 1.5]
@pytest.mark.parametrize("test_args", ["exposure", "saturation", "both"])
def test_yolo_dataset_augmentations(tmp_dataset, output_dir, test_args):
	"""Tests multiple datasets and augmentations to the datasets"""
	if tmp_dataset is None:
		return
	validate_yolov8_dataset(tmp_dataset)

	output_dataset = output_dir / f"{tmp_dataset.name}-{test_args}"
	augment_num = 1
	if test_args == "exposure":
		yolo_change_exposure(
			tmp_dataset, output_dataset,
			AUGMENT_VALUES, "{file}_exp{ind}{ext}"
			)
		augment_num = len(AUGMENT_VALUES)
	elif test_args == "saturation":
		yolo_change_saturation(
			tmp_dataset, output_dataset,
			AUGMENT_VALUES, "{file}_sat{ind}{ext}"
			)
		augment_num = len(AUGMENT_VALUES)
	elif test_args == "both":
		yolo_change_exposure(
			tmp_dataset, output_dataset,
			AUGMENT_VALUES, "{file}_exp{ind}{ext}"
			)
		yolo_change_saturation(
			output_dataset, output_dataset,
			AUGMENT_VALUES, "{file}_sat{ind}{ext}"
			)
		augment_num = len(AUGMENT_VALUES) * len(AUGMENT_VALUES)
		
	validate_yolov8_dataset(output_dataset)
	for split in SPLIT_DIRS:
		split_dir_in = tmp_dataset / split / "images"
		file_count_in = len(list(split_dir_in.glob("*.png")))
		split_dir_out = output_dataset / split / "images"
		file_count_out = len(list(split_dir_out.glob("*.png")))
		#	making sure number of files match
		assert file_count_in * augment_num == file_count_out