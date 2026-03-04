#	@file: _shared_module.py (protected)
#	@brief: shared functions for geometric and phototmetric modules

import argparse
from pathlib import Path
import augmentation.config as config
from general_lib import parse_float_list

def get_args():
	parser = argparse.ArgumentParser(
		description="Apply augmentations to images",
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
		'-r', "--resize",
		type=parse_float_list,
		metavar='LIST',
		help='Comma-seperated list of resize / scale values (e.g., "0.3,1.5,2.75")'
		)
	parser.add_argument(
		"--resize-origin",
		type=parse_float_list,
		metavar='LIST',
		help="Sets the origin point to scale from, use format x,y"
		)
	parser.add_argument(
		'-v', '--verbose',
		action='store_true',
		help='Enable verbose output for debugging'
		)
	parser.add_argument(
		'-vv', "--verbose-verbose",
		action='store_true',
		help="Enable even more verbose output for debugging"
		)
	
	args = parser.parse_args()
	
	# Set output_dir to input_dir if not provided
	if args.output_dir is None:
		args.output_dir = args.input_dir
	
	# Validation
	if not args.exposure and not args.saturation and not args.resize:
		parser.error("At least one of --exposure or --saturation or --resize must be specified")

	if args.resize_origin and len(args.resize_origin) != 2:
		parser.error("There needs to be 2 values for origin: x,y")

	# Verbose output
	if args.verbose_verbose:
		args.verbose = True
		config.VVERBOSE = True
	
	if args.verbose:
		config.VERBOSE = True
		print(f"Input directory: {args.input_dir}")
		print(f"Output directory: {args.output_dir}")
		if args.exposure:
			print(f"Exposure values: {args.exposure}")
		if args.saturation:
			print(f"Saturation values: {args.saturation}")
		if args.resize:
			print(f"Scale / resize values: {args.resize}")
		if args.resize_origin:
			print(f"Origin point: {args.resize_origin}")
	
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
	
	return args