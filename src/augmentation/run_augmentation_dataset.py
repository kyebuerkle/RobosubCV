#	@file: run_augmentation_dataset.py
#	@brief: script to run dataset augmentation functions

from pathlib import Path

from general_lib import loading, Animations
import augmentation.config as config
from augmentation import yolo_change_exposure, yolo_change_saturation, yolo_change_resize
from augmentation import yolo_gaussian_blur, yolo_motion_blur, yolo_contrast, yolo_hue_shift
from augmentation._shared_module import get_args

def _do_yolo_augmentations(args, input_path, output_path):
	"""this is just to simplify main, only use case is there"""
	if args.exposure:
		if config.VERBOSE:
			print(f"\nApplying exposure augmentations: {args.exposure}")
		yolo_change_exposure(
			input_path,
			output_path,
			args.exposure,
			"{file}_exp{ind}{ext}"
			)
		input_path = output_path
	
	if args.saturation:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentations: {args.saturation}")
		yolo_change_saturation(
			input_path,
			output_path,
			args.saturation,
			"{file}_sat{ind}{ext}"
			)
		input_path = output_path

	if args.resize:
		if config.VERBOSE:
			print(f"\nApplying resize augmentation: {args.resize}")
		yolo_change_resize(
			input_path,
			output_path,
			args.resize,
			"{file}_res{ind}{ext}"
			)
		input_path = output_path

	if args.gaussian_blur:
		if config.VERBOSE:
			print(f"\nApplying gaussian blur augmentations: {args.gaussian_blur}")
		yolo_gaussian_blur(
			input_path,
			output_path,
			args.gaussian_blur,
			"{file}_gblur{ind}{ext}"
			)
		input_path = output_path

	if args.motion_blur:
		if config.VERBOSE:
			print(f"\nApplying motion blur augmentations: {args.motion_blur}")
		yolo_motion_blur(
			input_path,
			output_path,
			args.motion_blur,
			"{file}_mblur{ind}{ext}"
			)
		input_path = output_path

	if args.contrast:
		if config.VERBOSE:
			print(f"\nApplying contrast augmentations: {args.contrast}")
		yolo_contrast(
			input_path,
			output_path,
			args.contrast,
			"{file}_con{ind}{ext}"
			)
		input_path = output_path

	if args.hue_shift:
		if config.VERBOSE:
			print(f"\nApplying hue shift augmentations: {args.hue_shift}")
		yolo_hue_shift(
			input_path,
			output_path,
			args.hue_shift,
			"{file}_hue{ind}{ext}"
			)
		input_path = output_path

def main(args):
	"""Main script"""
	# Verify input directory exists
	input_path = Path(args.input_dir).resolve()
	output_path = Path(args.output_dir).resolve()

	#	determine if dataset is coco or yolov8
	yaml_path = input_path / "data.yaml"
	coco_path = input_path / "test/_annotations.coco.json"
	if yaml_path.exists():
		if config.VERBOSE:
			print("Dataset is yolov8 formating")
		_do_yolo_augmentations(args, input_path, output_path)
	elif coco_path.exists():
		if config.VERBOSE:
			print("Dataset is coco formating")
		print("Nothing to do for coco format yet")

	if config.VERBOSE:
		print("Finished augmenting dataset!")

if __name__ == "__main__":
	args = get_args()
	if config.VERBOSE:
		main(args)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(args)