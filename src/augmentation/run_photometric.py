#	@file: run_photometric.py
#	@brief: run this script to use the photometric augmentation functions

import argparse
from pathlib import Path

from general_lib import loading, Animations, parse_float_list
from augmentation import dir_change_exposure, dir_change_saturation
import augmentation.config as config

def get_args():
	parser = argparse.ArgumentParser(
		description="Apply photometric augmentations (exposure/saturation) to images",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
%(prog)s ./images -e 0.8,1.2 -s 0.9,1.1
%(prog)s ./input ./output --exposure 0.5,1.0,1.5
%(prog)s ./dataset -e 0.7,1.3 -v
		"""
	)
	
	# Positional arguments
	parser.add_argument(
		'input_dir',
		type=str,
		help='Input directory containing images to augment'
	)
	parser.add_argument(
		'output_dir',
		type=str,
		nargs='?',
		default=None,
		help='Output directory for augmented images (default: same as input_dir)'
	)
	# Optional arguments
	parser.add_argument(
		'-e', '--exposure',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of exposure values (e.g., "0.8,1.2,1.5")'
	)
	parser.add_argument(
		'-s', '--saturation',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of saturation values (e.g., "0.9,1.0,1.1")'
	)
	parser.add_argument(
		'-v', '--verbose',
		action='store_true',
		help='Enable verbose output for debugging'
	)
	args = parser.parse_args()
	
	# Set output_dir to input_dir if not provided
	if args.output_dir is None:
		args.output_dir = args.input_dir
	
	# Validation
	if not args.exposure and not args.saturation:
		parser.error("At least one of --exposure or --saturation must be specified")
	
	# Verbose output
	if args.verbose:
		config.VERBOSE = True
		print(f"Input directory: {args.input_dir}")
		print(f"Output directory: {args.output_dir}")
		if args.exposure:
			print(f"Exposure values: {args.exposure}")
		if args.saturation:
			print(f"Saturation values: {args.saturation}")
	
	return (parser, args)

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
			args.exposure
			)
	
	if args.saturation and args.exposure:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		dir_change_saturation(
			output_path,
			output_path,
			args.saturation
			)
	elif args.saturation:
		if config.VERBOSE:
			print(f"\nApplying saturation augmentation...")
		dir_change_saturation(
			input_path,
			output_path,
			args.saturation
			)
			
	if config.VERBOSE:
		print("Augmentation complete!")

if __name__ == "__main__":
	out = get_args()
	if config.VERBOSE:
		main(*out)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(*out)