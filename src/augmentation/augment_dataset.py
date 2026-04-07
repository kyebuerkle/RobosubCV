#	@file: augment_dataset.py
#	@brief: This runs through the dataset and uses the photometric and geometric functions
#	TODO: I didn't add label handling abilities woops

from pathlib import Path
from typing import List, Callable, Optional
import yaml
import json
import shutil

from .photometric_module import change_exposure, change_saturation
from .photometric_module_2 import gaussian_blur, motion_blur, contrast, hue_shift
from .geometric_module import change_scale, yolo_scale_label
from .augment_strategy import (
	AugSpec,
	apply_augmentations_to_dir,
	MODE_ALL, MODE_RANDOM, MODE_CALC,
)
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
		if config.VVERBOSE:
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
		if config.VVERBOSE:
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
		if config.VVERBOSE:
			print("replacing original directory")
		for img_file in image_files:
			img_file.unlink()
		for temp_file in temp_dir.iterdir():
			shutil.move(str(temp_file), str(input_path / temp_file.name))
		
		temp_dir.rmdir()
	if config.VERBOSE:
		print(f"Created {len(image_files)*len(value_list)} augmented images from {len(image_files)} images")

def _yolo_copy_labels(
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
		if config.VVERBOSE:
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

def dir_change_scale(input_dir, output_dir, saturation_list: List[float], name_conv: str = "", origin = None):
	"""
	Apply scale changes to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param saturation_list: list of scale value percentages 0.0 -> 2.0
	:type saturation_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = scale value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	partial_change_scale = lambda img, out, amount: change_scale(img, out, amount, origin)
	_dir_change_generic(
		input_dir, output_dir, saturation_list,
		partial_change_scale, name_conv
		)

def dir_gaussian_blur(input_dir, output_dir, blur_list: List[float], name_conv: str = ""):
	"""
	Apply gaussian blur to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param blur_list: list of blur amount percentages, 1.0 = base sigma 10
	:type blur_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = blur value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, blur_list,
		gaussian_blur, name_conv
		)

def dir_motion_blur(input_dir, output_dir, blur_list: List[float], name_conv: str = ""):
	"""
	Apply motion blur to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param blur_list: list of blur amount percentages, 1.0 = base 20 px streak
	:type blur_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = blur value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, blur_list,
		motion_blur, name_conv
		)

def dir_contrast(input_dir, output_dir, contrast_list: List[float], name_conv: str = ""):
	"""
	Apply contrast scaling to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param contrast_list: list of contrast amount percentages, 1.0 = no change
	:type contrast_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = contrast value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, contrast_list,
		contrast, name_conv
		)

def dir_hue_shift(input_dir, output_dir, hue_list: List[float], name_conv: str = ""):
	"""
	Apply hue shifting to all images in a directory.

	:param input_dir: input directory path (str or Path)
	:param output_dir: output directory path (str or Path)
	:param hue_list: list of hue shift amounts, 1.0 = no change, 1.5 = +90 degrees
	:type hue_list: List [ float ]
	:param name_conv: naming convention, use {} for formatting: ind = index, val = hue value, file = og file name, ext = extention
	:type name_conv: string format
	"""
	_dir_change_generic(
		input_dir, output_dir, hue_list,
		hue_shift, name_conv
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
	
def yolo_change_resize(input_dataset, output_dataset, resize_list: List[float], name_conv: str = ""):
	"""
	Apply resize changes to all directories in a yolov8 dataset
	
	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param resize_list: list of values to augment
	:type resize_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_resize_function,
		resize_list, name_conv,
		output_dataset
		)

def yolo_gaussian_blur(input_dataset, output_dataset, blur_list: List[float], name_conv: str = ""):
	"""
	Apply gaussian blur to all directories in a yolov8 dataset

	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param blur_list: list of values to augment
	:type blur_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_gaussian_blur_function,
		blur_list, name_conv,
		output_dataset
		)

def yolo_motion_blur(input_dataset, output_dataset, blur_list: List[float], name_conv: str = ""):
	"""
	Apply motion blur to all directories in a yolov8 dataset

	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param blur_list: list of values to augment
	:type blur_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_motion_blur_function,
		blur_list, name_conv,
		output_dataset
		)

def yolo_contrast(input_dataset, output_dataset, contrast_list: List[float], name_conv: str = ""):
	"""
	Apply contrast scaling to all directories in a yolov8 dataset

	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param contrast_list: list of values to augment
	:type contrast_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_contrast_function,
		contrast_list, name_conv,
		output_dataset
		)

def yolo_hue_shift(input_dataset, output_dataset, hue_list: List[float], name_conv: str = ""):
	"""
	Apply hue shifting to all directories in a yolov8 dataset

	:param input_dataset: input dataset path
	:param output_dataset: output dataset path
	:param hue_list: list of values to augment
	:type hue_list: List[float]
	:param name_conv: naming convention
	:type name_conv: str
	"""
	if not output_dataset or input_dataset == output_dataset:
		output_dataset = None
	_yolo_dataset_generic(
		input_dataset, _yolo_hue_shift_function,
		hue_list, name_conv,
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
			if config.VVERBOSE:
				print("This directory already exists with stuff, overwriting...")
			shutil.rmtree(new_dataset_path)
		new_dataset_path.mkdir(parents=True, exist_ok=True)
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
	_yolo_copy_labels(
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
	_yolo_copy_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)

def _yolo_resize_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the resize functions for the _yolo_dataset_generic input"""
	"""
	Generic function to apply augmentations to all images in a directory.
	
	:param input_dir: Directory containing input images
	:param output_dir: Directory to save output images
	:param value_list: List of values to apply
	:param augmentation_func: Function to call (change_exposure or change_saturation)
	:param name_conv: Format string for output names: ind = augmentation index, val = list value, file = original file name, ext = extention
	"""
	images_dir = Path(img_in_dir).resolve()
	images_out_dir = Path(img_out_dir).resolve()
	labels_dir = Path(labels_in_dir).resolve()
	labels_out_dir = Path(labels_out_dir).resolve()

	if not images_dir.is_dir():
		print(f"{images_dir} is not a directory!")
		return
	if not value_list:
		print(f"No input for change_resize values")
		return
	
	# delete originals and move augmented images
	replace_mode = (images_dir == images_out_dir)
	if replace_mode:
		if config.VVERBOSE:
			print("replacing input dir for resize dataset")
		temp_dir = images_dir.parent / f"{images_dir.name}_temp"
		temp_dir.mkdir(exist_ok=True)
		working_output = temp_dir

		temp_ldir = labels_dir.parent / f"{labels_dir.name}_temp"
		temp_ldir.mkdir(exist_ok=True)
		working_label_output = temp_ldir
	else:
		images_out_dir.mkdir(parents=True, exist_ok=True)
		working_output = images_out_dir

		labels_out_dir.mkdir(parents=True, exist_ok=True)
		working_label_output = labels_out_dir
	
	image_files = [f for f in images_dir.iterdir() 
				if f.is_file() and f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
	label_files = []
	
	for ind, img_file in enumerate(image_files):
		if config.VVERBOSE:
			print(f"Augmenting image {img_file.name}...")

		label_file = labels_dir / f"{img_file.stem}.txt"
		if not label_file.exists():
			print(f"No label file associated with {img_file.name}\nSkipping resize...")
			continue
		
		label_files.append(label_file)
		for vind, val in enumerate(value_list):
			if name_conv:
				file_name = name_conv.format(
					ind = vind,
					val = f"{val:.3f}",
					file = img_file.stem,
					ext = img_file.suffix
					)
				label_name = name_conv.format(
					ind = vind,
					val = f"{val:.3f}",
					file = label_file.stem,
					ext = label_file.suffix
					)
			else:
				file_name = f"{vind}_{img_file.name}"
				label_name = f"{vind}_{label_file.name}"
			
			output_file = working_output / file_name
			output_label = working_label_output / label_name
			#	TODO: get the correct origin points from design
			change_scale(img_file, output_file, val)
			yolo_scale_label(label_file, output_label, val, img_file)
	
	if replace_mode:
		if config.VVERBOSE:
			print("replacing original resize directory")
		for img_file in image_files:
			img_file.unlink()
		for label_file in label_files:
			label_file.unlink()
		for temp_file in temp_dir.iterdir():
			shutil.move(str(temp_file), str(images_dir / temp_file.name))
		for temp_label in temp_ldir.iterdir():
			shutil.move(str(temp_label), str(labels_dir / temp_label.name))
		
		temp_dir.rmdir()
		temp_ldir.rmdir()

	if config.VERBOSE:
		print(f"Created {len(image_files)*len(value_list)} resized images from {len(image_files)} images")

def _yolo_gaussian_blur_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the gaussian blur functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list,
		gaussian_blur, name_conv
		)
	_yolo_copy_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)

def _yolo_motion_blur_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the motion blur functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list,
		motion_blur, name_conv
		)
	_yolo_copy_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)

def _yolo_contrast_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the contrast functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list,
		contrast, name_conv
		)
	_yolo_copy_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)

def _yolo_hue_shift_function(
	img_in_dir, img_out_dir,
	labels_in_dir, labels_out_dir,
	value_list, name_conv
	):
	"""This is used to combine the hue shift functions for the _yolo_dataset_generic input"""
	_dir_change_generic(
		img_in_dir, img_out_dir, value_list,
		hue_shift, name_conv
		)
	_yolo_copy_labels(
		labels_in_dir, labels_out_dir,
		value_list, name_conv
		)


#====================================================================================================
#  Strategy-based augmentation  (all / random / calc)
#====================================================================================================

def yolo_augment(
	input_dataset:  str,
	output_dataset: str,
	aug_config:     dict,
	mode:           str = MODE_ALL,
	num:            int = 1,
):
	"""
	Apply augmentations to a full YOLOv8 dataset using a chosen strategy.

	This is the single entry point used by dataset_config.run_augmentations()
	when --augment is specified.  All aug types are combined into one pass per
	image so augmentations can be stacked (e.g. exposure + contrast + blur on
	the same output image), avoiding the multiplicative explosion of chaining
	separate yolo_change_* calls.

	:param input_dataset:  Path to the source YOLOv8 dataset (contains data.yaml)
	:param output_dataset: Path for the augmented output dataset
	:param aug_config:     Dict mapping aug-type names to value lists, e.g.:
	                         {
	                           'exposure':      [0.615, 1.0, 1.385],
	                           'contrast':      [0.7, 1.0, 1.3],
	                           'resize':        [0.5, 1.0, 1.5],
	                           'motion_blur':   [1.0, 1.5],
	                           'gaussian_blur': [1.0, 1.5],
	                           'hue_shift':     [0.8, 1.2],
	                           'saturation':    [0.7, 1.3],
	                         }
	                       Keys with empty lists or None are skipped.
	:param mode:           'all' | 'random' | 'calc'
	:param num:            Number of augmented images per original (random/calc only)
	"""
	#	Map aug-type name -> (photometric/geometric function, optional label function)
	_AUG_FUNCS = {
		'exposure'     : (change_exposure,  None),
		'contrast'     : (contrast,         None),
		'gaussian_blur': (gaussian_blur,     None),
		'motion_blur'  : (motion_blur,       None),
		'hue_shift'    : (hue_shift,         None),
		'saturation'   : (change_saturation, None),
		'resize'       : (change_scale,      yolo_scale_label),
	}

	#	Build AugSpec list from aug_config, preserving a sensible application order
	_ORDER = ['exposure', 'contrast', 'resize', 'motion_blur', 'gaussian_blur', 'hue_shift', 'saturation']
	aug_specs = []
	for name in _ORDER:
		vals = aug_config.get(name)
		if not vals:
			continue
		func, label_func = _AUG_FUNCS[name]
		aug_specs.append(AugSpec(name=name, func=func, values=list(vals), label_func=label_func))

	if not aug_specs:
		print("[yolo_augment] No active augmentation specs, nothing to do.")
		return

	_yolo_dataset_generic(
		yaml_path    = input_dataset,
		augment_func = _make_strategy_func(aug_specs, mode, num),
		value_list   = [1.0],		#	value_list unused by strategy func, passed as dummy
		name_conv    = "",
		new_dataset  = output_dataset if output_dataset != input_dataset else None,
	)


def _make_strategy_func(
	aug_specs: List[AugSpec],
	mode:      str,
	num:       int,
) -> Callable:
	"""
	Returns a _yolo_*_function-compatible callable that uses apply_augmentations_to_dir.

	The returned function has the same signature as the existing _yolo_*_function helpers:
	  f(img_in_dir, img_out_dir, labels_in_dir, labels_out_dir, value_list, name_conv)
	so it plugs straight into _yolo_dataset_generic.
	"""
	def _strategy_func(
		img_in_dir, img_out_dir,
		labels_in_dir, labels_out_dir,
		value_list, name_conv		#	value_list/name_conv ignored — strategy manages naming
	):
		apply_augmentations_to_dir(
			images_dir     = img_in_dir,
			images_out_dir = img_out_dir,
			labels_dir     = labels_in_dir,
			labels_out_dir = labels_out_dir,
			aug_specs      = aug_specs,
			mode           = mode,
			num            = num,
		)

	return _strategy_func