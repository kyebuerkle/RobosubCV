#	@file: augment_dataset.py
#	@brief: This runs through the dataset and uses the photometric and geometric functions
#	TODO: I didn't add label handling abilities woops

from pathlib import Path
from typing import List, Callable, Optional
import yaml
import json
import shutil

from augmentation import change_exposure, change_saturation
import augmentation.config as config

#	this is cool, its a function that you input another function as a parameter, it overrides it so that it loops through
#	the normal Dataset code 
#	NOTE: the underscore makes it static, so it can only be called by this file
""" example of name_conv = "{file}_sat_{val}-{ind}.{ext}" this will be -> file_name_sat_0.500-1.png """
def _dir_change_generic(
	input_dir,
	output_dir,
	value_list: List[float],
	augmentation_func: Callable,
	name_conv: str = ""
	):
	"""
	Generic function to apply augmentations to all images in a directory.
	
	:param input_dir: Directory containing input images
	:param output_dir: Directory to save output images
	:param value_list: List of values to apply
	:param augmentation_func: Function to call (change_exposure or change_saturation)
	:param name_conv: Format string for output names: ind = augmentation index, val = list value, file = original file name, ext = extention
	"""
	input_path = Path(input_dir).resolve()
	output_path = Path(output_dir).resolve()

	if not input_path.is_dir():
		print(f"{input_path} is not a directory!")
		return
	if not value_list:
		print(f"No input for {augmentation_func.__name__} values")
		return
	
	# delete originals and move augmented images
	replace_mode = (input_path == output_path)
	if replace_mode:
		if config.VERBOSE:
			print("replacing input dir")
		temp_dir = input_path.parent / f"{input_path.name}_temp"
		temp_dir.mkdir(exist_ok=True)
		working_output = temp_dir
	else:
		output_path.mkdir(parents=True, exist_ok=True)
		working_output = output_path
	
	image_files = [f for f in input_path.iterdir() 
				if f.is_file() and f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
	
	for ind, img_file in enumerate(image_files):
		if config.VERBOSE:
			print(f"Augmenting image {img_file.name}...")
		for vind, val in enumerate(value_list):
			if name_conv:
				file_name = name_conv.format(
					ind = vind,
					val = f"{val:.3f}",
					file = img_file.stem,
					ext = img_file.suffix
				)
			else:
				file_name = f"{vind}_{img_file.name}"
			
			output_file = working_output / file_name
			augmentation_func(img_file, output_file, val)
	
	if replace_mode:
		if config.VERBOSE:
			print("replacing original directory")
		for img_file in image_files:
			img_file.unlink()
		for temp_file in temp_dir.iterdir():
			shutil.move(str(temp_file), str(input_path / temp_file.name))
		
		temp_dir.rmdir()
	if config.VERBOSE:
		print(f"Created {len(image_files)*len(value_list)} augmented images from {len(image_files)} images")

def _yolo_change_labels(
	labels_in_dir, labels_out_dir, 
	value_list: List[float],
	name_conv: str = ""
	):
	"""
	copys all of the labels for phototmetric augmentations done to a yolo dataset
	
	:param labels_in_dir: input label dir
	:param labels_out_dir: output label dir (if they are the same all labels are replaced)
	"""
	input_path = Path(labels_in_dir).resolve()
	output_path = Path(labels_out_dir).resolve()

	if not input_path.is_dir():
		print(f"{input_path} is not a directory!")
		return
	if not value_list:
		print(f"No value list!")
		return
	
	# delete originals and move augmented images
	replace_mode = (input_path == output_path)
	if replace_mode:
		if config.VERBOSE:
			print("replacing label dir")
		temp_dir = input_path.parent / f"{input_path.name}_temp"
		temp_dir.mkdir(exist_ok=True)
		working_output = temp_dir
	else:
		output_path.mkdir(parents=True, exist_ok=True)
		working_output = output_path
	
	label_files = [f for f in input_path.iterdir() 
				if f.is_file() and f.suffix.lower() in [".txt"]]
	
	for ind, label_file in enumerate(label_files):
		for vind, val in enumerate(value_list):
			if name_conv:
				file_name = name_conv.format(
					ind = vind,
					val = f"{val:.3f}",
					file = label_file.stem,
					ext = label_file.suffix
					)
			else:
				file_name = f"{vind}_{label_file.name}"
			
			output_file = working_output / file_name
			shutil.copy2(label_file, output_file)
	
	if replace_mode:
		for label_file in label_files:
			label_file.unlink()
		for temp_file in temp_dir.iterdir():
			shutil.move(str(temp_file), str(input_path / temp_file.name))
		
		temp_dir.rmdir()

def dir_change_exposure(input_dir, output_dir, exposure_list: List[float], name_conv: str = ""):
	"""
	Apply exposure changes to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param exposure_list: list of exposure value percentages 0.0 -> 2.0
	:type exposure_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = exposure value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, exposure_list,
		change_exposure, name_conv
		)

def dir_change_saturation(input_dir, output_dir, saturation_list: List[float], name_conv: str = ""):
	"""
	Apply saturation changes to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param saturation_list: list of saturation value percentages 0.0 -> 2.0
	:type saturation_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = saturation value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, saturation_list,
		change_saturation, name_conv
		)

def yolo_change_exposure(input_dataset, output_dataset, exposure_list: List[float], name_conv: str = ""):
	"""
	Apply exposure changes to all directories in a yolov8 dataset
	
	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param exposure_list: list of values to augment
	:type exposure_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_exposure_function,
		exposure_list, name_conv,
		output_dataset
		)

def yolo_change_saturation(input_dataset, output_dataset, saturation_list: List[float], name_conv: str = ""):
	"""
	Apply saturation changes to all directories in a yolov8 dataset
	
	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param saturation_list: list of values to augment
	:type saturation_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_saturation_function,
		saturation_list, name_conv,
		output_dataset
		)

#====================================================================================================
# Formats: these functions go through the yolo and coco formats, labels, and README.md from Roboflow
#====================================================================================================

def _yolo_dataset_generic(
	yaml_path: str,
	augment_func: Callable[[str, str, str, str, List[float], str], None],
	value_list: List[float],
	name_conv: str = "",
	new_dataset: Optional[str] = None
	):
	"""
	Apply augmentation to all splits in a YOLO dataset.
	
	:param yaml_path: Path to data.yaml file or directory containing it
	:param change_func: Function to call
	:type change_func: ( img_in_dir, img_out_dir, label_in_dir, label_out_dir, values, name_conv )
	:param value_list: List of values to apply
	:param name_conv: Format string for output names
	:param new_dataset: Optional path to create new augmented dataset
	"""
	# Handle if directory is passed instead of yaml file
	yaml_path = Path(yaml_path)
	if yaml_path.is_dir():
		yaml_file = yaml_path / "data.yaml"
	else:
		yaml_file = yaml_path
	if not yaml_file.exists():
		print(f"YAML file not found: {yaml_file}")
		return
	
	with open(yaml_file, 'r') as f:
		data = yaml.safe_load(f)
	
	# Get dataset root
	dataset_root = yaml_file.parent
	if 'path' in data:
		dataset_root = Path(data['path']).resolve()
	
	# Determine if creating new dataset
	new_dataset_path = None
	new_yaml = None
	if new_dataset:
		new_dataset_path = Path(new_dataset)
		if (new_dataset_path.exists() and any(new_dataset_path.iterdir())):
			if config.VERBOSE:
				print("This directory already exists with stuff, overwriting...")
			shutil.rmtree(new_dataset_path)
		new_dataset_path.mkdir(parents=True)
		new_yaml = {}
		
	splits = ['train', 'val', 'test']
	for split in splits:
		if split not in data:
			continue
		
		# Get paths from YAML
		split_path = data[split]
		if Path(split_path).is_absolute():
			images_dir = Path(split_path)
		else:
			images_dir = (yaml_file / split_path).resolve()
		
		# Try common patterns if path doesn't exist
		if not images_dir.exists():
			possible_paths = [
				dataset_root / split / 'images',
				dataset_root / 'images' / split,
				]
			images_dir = next((p for p in possible_paths if p.exists()), None)
		if images_dir is None or not images_dir.exists():
			if config.VERBOSE:
				print(f"Skipping {split}: directory not found")
			continue
		
		# Get corresponding labels directory
		labels_dir = images_dir.parent / 'labels'
		if not labels_dir.exists():
			possible_paths = [
				dataset_root / split / 'labels',
				dataset_root / 'labels' / split,
				]
			labels_dir = next((p for p in possible_paths if p.exists()), None)
		if config.VERBOSE:
			print(f"Processing {split} images: {images_dir}, labels: {labels_dir}")
		
		# Create new dataset structure
		if new_dataset_path:
			split_dir = Path(split_path).parts[1:-1]
			new_images_dir = new_dataset_path / Path(*split_dir) / 'images'
			new_labels_dir = new_dataset_path / Path(*split_dir) / 'labels'
			new_images_dir.mkdir(parents=True, exist_ok=True)
			new_labels_dir.mkdir(parents=True, exist_ok=True)
			
			#	params (img_in_dir, img_out_dir, label_in_dir, label_out_dir, values, name_conv)
			augment_func(str(images_dir), str(new_images_dir), 
						str(labels_dir), str(new_labels_dir),
						value_list, name_conv)
			
			new_yaml[split] = split_path
			#new_yaml[split] = f"../{split}/images"
		else:
			#	params (img_in_dir, img_out_dir, label_in_dir, label_out_dir, values, name_conv)
			augment_func(str(images_dir), str(images_dir), 
						str(labels_dir), str(labels_dir),
						value_list, name_conv)
	
	# Create new dataset files
	if new_dataset_path:
		new_yaml_file = new_dataset_path / "data.yaml"
		new_yaml_data = data.copy()
		new_yaml_data.update(new_yaml)
		new_yaml_data['path'] = str(new_dataset_path)
		
		with open(new_yaml_file, 'w') as f:
			yaml.dump(new_yaml_data, f, default_flow_style=False)
		
		""" TODO: README.md edit
		# Create/update README
		readme_path = new_dataset_path / "README.md"
		dataset_name = new_dataset_path.name
		with open(readme_path, 'w') as f:
			f.write(f"# {dataset_name}\n\n")
			f.write(f"Augmented dataset created from: {dataset_root}\n\n")
			f.write(f"Augmentation applied: {change_func.__name__}\n")
			f.write(f"Values: {value_list}\n")
		"""
		if config.VERBOSE:
			print(f"\nNew dataset created at: {new_dataset_path}")

	if config.VERBOSE:
		print(f"Completed augmentation to yolov8 dataset: {dataset_root}")

def _yolo_exposure_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the exposure functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list, 
		change_exposure, name_conv
		)
	_yolo_change_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)

def _yolo_saturation_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the saturation functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list, 
		change_saturation, name_conv
		)
	_yolo_change_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)