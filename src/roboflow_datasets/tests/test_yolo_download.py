import pytest
from pathlib import Path
from general_lib import validate_yolov8_dataset

def test_yolo_dataset_format(dataset_root):
	validate_yolov8_dataset(dataset_root)