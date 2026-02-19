#	@file: augment_dataset.py
#	@brief: This runs through the dataset and uses the photometric and geometric functions
#	TODO: I didn't add label handling abilities woops

from pathlib import Path
from typing import List, Callable
import shutil

from augmentation import change_exposure, change_saturation
import augmentation.config as config

#	this is cool, its a function that you input another function as a parameter, it overrides it so that it loops through
#	the normal Dataset code 
#	NOTE: the underscore makes it static, so it can only be called by this file
""" example of name_conv = "{file}_sat_{val}-{ind}.{ext}" this will be -> file_name_sat_0.500-1.png """
def _dataset_change_generic(
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
	_dataset_change_generic(
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
	_dataset_change_generic(
		input_dir, output_dir, saturation_list,
		change_saturation, name_conv
	)