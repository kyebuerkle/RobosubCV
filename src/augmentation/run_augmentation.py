#	@file: run_photometric.py
#	@brief: run this script to use the photometric augmentation functions

from pathlib import Path

from general_lib import loading, Animations
from augmentation import dir_change_exposure, dir_change_saturation, dir_change_scale
import augmentation.config as config
from augmentation._shared_module import get_args

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

	# Apply augmentations
	if args.exposure:
		if config.VERBOSE:
			print(f"\nApplying exposure augmentation...")
		dir_change_exposure(
			input_path,
			output_path,
			args.exposure,
			"{file}_exp{val}{ext}"
			)
		input_path = output_path

	if args.saturation:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		dir_change_saturation(
			input_path,
			output_path,
			args.saturation,
			"{file}_sat{val}{ext}"
			)
		input_path = output_path

	if args.resize:
		if config.VERBOSE:
			print(f"\nApplying resize augmentation...")
		dir_change_scale(
			input_path,
			output_path,
			args.resize,
			"{file}_res{val}{ext}",
			args.resize_origin if args.resize_origin else None
			)
		input_path = output_path
			
	if config.VERBOSE:
		print("Augmentation complete!")

if __name__ == "__main__":
	out = get_args()
	if config.VERBOSE:
		main(*out)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(*out)