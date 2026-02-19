#	@file: run_photometric_dataset.py
#	@brief: script to run dataset augmentation functions

from pathlib import Path

from general_lib import loading, Animations
import augmentation.config as config
from augmentation import yolo_change_exposure, yolo_change_saturation
from augmentation.run_photometric import get_args

def _do_yolo_augmentations(args, input_path, output_path):
	"""this is just to simplify main, only use case is there"""
	if args.exposure:
		if config.VERBOSE:
			print(f"\nApplying exposure augmentation...")
		yolo_change_exposure(
			input_path,
			output_path,
			args.exposure,
			"{file}_exp{ind}{ext}"
			)
	
	if args.saturation and args.exposure:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		yolo_change_saturation(
			output_path,
			output_path,
			args.saturation,
			"{file}sat{ind}{ext}"
			)
	elif args.saturation:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		yolo_change_saturation(
			input_path,
			output_path,
			args.saturation,
			"{file}_sat{ind}{ext}"
			)

def main(parser, args):
	# Verify input directory exists
	input_path = Path(args.input_dir).resolve()
	if not input_path.exists():
		parser.error(f"Input directory does not exist: {args.input_dir}")
	if not input_path.is_dir():
		parser.error(f"Input path is not a directory: {args.input_dir}")

	output_path = Path(args.output_dir).resolve()
	if not output_path.exists():
		if config.VERBOSE:
			print(f"Making new dir {str(output_path)}")
		output_path.mkdir(parents=True)

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
	out = get_args()
	if config.VERBOSE:
		main(*out)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(*out)
		