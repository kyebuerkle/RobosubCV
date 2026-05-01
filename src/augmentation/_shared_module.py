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
%(prog)s ./images --gaussian-blur 0.5,1.0,1.5
%(prog)s ./images --motion-blur 0.8,1.2 --contrast 0.7,1.3
%(prog)s ./images --hue-shift 0.5,1.5
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

	# ── Photometric module 1 ──────────────────────────────────────────────────
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

	# ── Geometric module ──────────────────────────────────────────────────────
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

	# ── Photometric module 2 ──────────────────────────────────────────────────
	parser.add_argument(
		'-gb', '--gaussian-blur',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of gaussian blur amounts (e.g., "0.5,1.0,1.5"). 1.0 = base sigma 10'
		)
	parser.add_argument(
		'-mb', '--motion-blur',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of motion blur amounts (e.g., "0.5,1.0,1.5"). 1.0 = base 20px streak'
		)
	parser.add_argument(
		'-c', '--contrast',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of contrast amounts (e.g., "0.7,1.0,1.3"). 1.0 = no change'
		)
	parser.add_argument(
		'-hs', '--hue-shift',
		type=parse_float_list,
		metavar='LIST',
		help='Comma-separated list of hue shift amounts (e.g., "0.5,1.0,1.5"). 1.0 = no change, 1.5 = +90 degrees'
		)

	# ── Augmentation strategy ─────────────────────────────────────────────────
	parser.add_argument(
		'--augment',
		nargs='+',
		metavar=('MODE', 'NUM'),
		help=(
			'Augmentation strategy. Options:\n'
			'  --augment all            apply every combination (default, legacy)\n'
			'  --augment random 3       3 randomly combined augmentations per image\n'
			'  --augment calc   3       3 maximally-varied augmentations per image (deterministic)'
		)
		)

	# ── Verbosity ─────────────────────────────────────────────────────────────
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

	# Normalise hyphens → underscores so args.gaussian_blur etc. work
	# (argparse does this automatically for long flags, but being explicit is safer)

	# Validation
	has_any_augmentation = any([
		args.exposure,
		args.saturation,
		args.resize,
		args.gaussian_blur,
		args.motion_blur,
		args.contrast,
		args.hue_shift,
		])
	if not has_any_augmentation:
		parser.error(
			"At least one augmentation must be specified: "
			"--exposure, --saturation, --resize, "
			"--gaussian-blur, --motion-blur, --contrast, or --hue-shift"
			)

	if args.resize_origin and len(args.resize_origin) != 2:
		parser.error("There needs to be 2 values for origin: x,y")

	#	Parse --augment MODE [NUM]
	augment_mode = "all"
	augment_num  = 3
	if args.augment:
		raw = args.augment
		augment_mode = raw[0].lower()
		if augment_mode not in ("all", "random", "calc"):
			parser.error(f"--augment mode must be 'all', 'random', or 'calc', got '{augment_mode}'")
		if len(raw) >= 2:
			try:
				augment_num = int(raw[1])
				if augment_num < 1:
					raise ValueError
			except ValueError:
				parser.error(f"--augment NUM must be a positive integer, got '{raw[1]}'")
		elif augment_mode in ("random", "calc"):
			parser.error(f"--augment {augment_mode} requires a NUM argument, e.g. --augment {augment_mode} 3")
	args.augment_mode = augment_mode
	args.augment_num  = augment_num

	# Verbose output
	if args.verbose_verbose:
		args.verbose = True
		config.VVERBOSE = True
	
	if args.verbose:
		config.VERBOSE = True
		print(f"Input directory:  {args.input_dir}")
		print(f"Output directory: {args.output_dir}")
		print(f"Augment mode:     {args.augment_mode}  (num={args.augment_num})")
		if args.exposure:
			print(f"Exposure values:     {args.exposure}")
		if args.saturation:
			print(f"Saturation values:   {args.saturation}")
		if args.resize:
			print(f"Scale / resize:      {args.resize}")
		if args.resize_origin:
			print(f"Origin point:        {args.resize_origin}")
		if args.gaussian_blur:
			print(f"Gaussian blur:       {args.gaussian_blur}")
		if args.motion_blur:
			print(f"Motion blur:         {args.motion_blur}")
		if args.contrast:
			print(f"Contrast:            {args.contrast}")
		if args.hue_shift:
			print(f"Hue shift:           {args.hue_shift}")
	
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