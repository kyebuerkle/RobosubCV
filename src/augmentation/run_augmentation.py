#	@file: run_augmentation.py
#	@brief: run this script to use the photometric augmentation functions

from pathlib import Path

from general_lib import loading, Animations
from augmentation import dir_change_exposure, dir_change_saturation, dir_change_scale
from augmentation import dir_gaussian_blur, dir_motion_blur, dir_contrast, dir_hue_shift
import augmentation.config as config
from augmentation._shared_module import get_args

def main(args):
	input_path = Path(args.input_dir).resolve()
	output_path = Path(args.output_dir).resolve()

	# Apply augmentations
	if args.exposure:
		if config.VERBOSE:
			print(f"\nApplying exposure augmentation...")
		dir_change_exposure(
			input_path,
			output_path,
			args.exposure,
			"{file}_exp{ind}{ext}"
			)
		input_path = output_path

	if args.saturation:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		dir_change_saturation(
			input_path,
			output_path,
			args.saturation,
			"{file}_sat{ind}{ext}"
			)
		input_path = output_path

	if args.resize:
		if config.VERBOSE:
			print(f"\nApplying resize augmentation...")
		dir_change_scale(
			input_path,
			output_path,
			args.resize,
			"{file}_res{ind}{ext}",
			args.resize_origin if args.resize_origin else None
			)
		input_path = output_path

	if args.gaussian_blur:
		if config.VERBOSE:
			print(f"\nApplying gaussian blur augmentation...")
		dir_gaussian_blur(
			input_path,
			output_path,
			args.gaussian_blur,
			"{file}_gblur{ind}{ext}"
			)
		input_path = output_path

	if args.motion_blur:
		if config.VERBOSE:
			print(f"\nApplying motion blur augmentation...")
		dir_motion_blur(
			input_path,
			output_path,
			args.motion_blur,
			"{file}_mblur{ind}{ext}"
			)
		input_path = output_path

	if args.contrast:
		if config.VERBOSE:
			print(f"\nApplying contrast augmentation...")
		dir_contrast(
			input_path,
			output_path,
			args.contrast,
			"{file}_con{ind}{ext}"
			)
		input_path = output_path

	if args.hue_shift:
		if config.VERBOSE:
			print(f"\nApplying hue shift augmentation...")
		dir_hue_shift(
			input_path,
			output_path,
			args.hue_shift,
			"{file}_hue{ind}{ext}"
			)
		input_path = output_path

	if config.VERBOSE:
		print("Augmentation complete!")

if __name__ == "__main__":
	args = get_args()
	if config.VERBOSE:
		main(args)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(args)