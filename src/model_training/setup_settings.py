#	@file: setup_settings.py
#	@brief: This script will be used to setup the users config file, and roboflow login

import argparse
import sys
import roboflow
import json
from pathlib import Path

from general_lib import parse_float_list, parse_int_list
from roboflow_datasets import Config, roboflow_login

def _parse_args() -> dict:
	"""
	Parses the arguments. Returns dictionary of the arguments (use it to update without re-writing).

	Primary augmentation arguments:
	-exp/--exposure			comma separated exposure list
	-con/--contrast			comma separated contrast list  (replaces saturation as primary)
	-res/--resize			comma separated resize list
	-mb/--motion-blur		comma separated motion blur list
	-gb/--gaussian-blur		comma separated gaussian blur list
	-hs/--hue-shift			comma separated hue shift list
	-sat/--saturation		comma separated saturation list (legacy, kept for compat)

	Other arguments:
	-o/--output				output results directory
	-d/--dataset			dataset save directory
	--epochs / --patience / --device / --model
	"""
	parser = argparse.ArgumentParser(
		description="Set up the configuration file for model training",
		epilog="Use options to only change one portion of the configuration"
		)
	
	# ── Dataset / paths ───────────────────────────────────────────────────────
	parser.add_argument(
		'-o', "--output",
		metavar="DIR",
		type = str,
		help = "Output directory of the YOLO model results"
		)
	parser.add_argument(
		'-d', "--dataset",
		metavar="DIR",
		type = str,
		help = "directory to save the datasets to"
		)
	parser.add_argument(
		'-p', "--print",
		action="store_true",
		help = "prints the configuration.json file"
		)

	# ── Augmentation arguments ────────────────────────────────────────────────
	parser.add_argument(
		'-exp', "--exposure",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of exposure augmentations (e.g. 0.615,1,1.385)"
		)
	parser.add_argument(
		'-con', "--contrast",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of contrast augmentations (e.g. 0.7,1,1.3)"
		)
	parser.add_argument(
		'-res', "--resize",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of resize augmentations (e.g. 0.5,1,1.5)"
		)
	parser.add_argument(
		'-mb', "--motion-blur",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of motion blur augmentations (e.g. 0.5,1,1.5)"
		)
	parser.add_argument(
		'-gb', "--gaussian-blur",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of gaussian blur augmentations (e.g. 0.5,1,1.5)"
		)
	parser.add_argument(
		'-hs', "--hue-shift",
		metavar="LIST",
		type = parse_float_list,
		help = "comma separated list of hue shift augmentations (e.g. 0.5,1,1.5)"
		)
	parser.add_argument(
		'-sat', "--saturation",
		metavar="LIST",
		type = parse_float_list,
		help = "(legacy) comma separated list of saturation augmentations"
		)

	# ── Augmentation strategy ─────────────────────────────────────────────────
	parser.add_argument(
		'--augment',
		nargs='+',
		metavar=('MODE', 'NUM'),
		help=(
			'Augmentation strategy: '
			'"all" (default, one image per value), '
			'"random 3" (3 random combos per image), '
			'"calc 3" (3 maximally-varied combos, deterministic)'
		)
		)

	# ── Training parameters ───────────────────────────────────────────────────
	parser.add_argument(
		"--epochs",
		metavar="INT",
		type=int,
		help="epoch parameter for model training"
		)
	parser.add_argument(
		"--patience",
		metavar="INT",
		type=int,
		help="patience parameter for model training"
		)
	parser.add_argument(
		"--device",
		metavar="LIST",
		type=parse_int_list,
		help="device parameter for model creation"
		)
	parser.add_argument(
		"--model",
		metavar="YOLO MODEL",
		type=str,
		help="model parameter for model training, yolov8m.pt <- medium model, "
		     "replace the m with n for nano, s for small, l for large, and x for extra large"
		)
	
	args = parser.parse_args()
	ret = {}
	
	if args.output:
		ret["save"] = Path(args.output).resolve()
	if args.dataset:
		ret["directory"] = Path(args.dataset).resolve()
	if args.print:
		ret["print"] = args.print
	if args.exposure:
		ret["exposure"] = args.exposure
	if args.contrast:
		ret["contrast"] = args.contrast
	if args.resize:
		ret["resize"] = args.resize
	if args.motion_blur:
		ret["motion_blur"] = args.motion_blur
	if args.gaussian_blur:
		ret["gaussian_blur"] = args.gaussian_blur
	if args.hue_shift:
		ret["hue_shift"] = args.hue_shift
	if args.saturation:
		ret["saturation"] = args.saturation
	if args.augment:
		raw = args.augment
		mode = raw[0].lower()
		if mode not in ("all", "random", "calc"):
			print(f"Warning: unknown --augment mode '{mode}', defaulting to 'all'")
			mode = "all"
		ret["augment_mode"] = mode
		if len(raw) >= 2:
			try:
				ret["augment_num"] = int(raw[1])
			except ValueError:
				print(f"Warning: --augment NUM '{raw[1]}' is not an integer, defaulting to 3")
				ret["augment_num"] = 3
	if args.epochs:
		ret["epochs"] = args.epochs
	if args.patience:
		ret["patience"] = args.patience
	if args.device:
		ret["device"] = args.device
	if args.model:
		ret["model"] = args.model
	
	if not ret:
		return None
	else:
		return ret

def _check_yes(answer: str) -> bool:
	"""checks if an input is y or yes"""
	ans = answer.strip().lower()
	if ans == "y" or ans == "yes":
		return True
	else:
		return False

def _prompt_aug_list(label: str, default, default_label: str) -> list | None:
	"""
	Shared helper for the interactive augmentation prompts.

	:param label:         Display name shown to the user (e.g. 'Exposure')
	:param default:       The default list to use on empty input, or None for no default
	:param default_label: Human-readable description of the default (e.g. '-37.5%, original, +38.5%')
	:returns list | None: parsed list, or None if the user chose no augmentation
	"""
	values = input(f"\n    {label}: ")
	if not values or values.strip() == "":
		if default is not None:
			print(f"Saving defaults: {default}, {default_label}")
			return default
		else:
			print(f"No {label} augmentation")
			return None
	elif values.strip() == "0":
		print(f"No {label} augmentation, equivalent to 1.0")
		return None
	else:
		parsed = parse_float_list(values)
		print(f"Saving values: {parsed}")
		return parsed

def alturnate_main(arg_dict):
	"""
	alternate_main is used to update the existing settings, main will overwrite everything
	"""
	printing = arg_dict.pop("print", False)
	if printing:
		print("Updating configuration.json with these arguments:")
		print(json.dumps(arg_dict, indent=4))

	config = Config()
	config.config_dict(arg_dict)
	config.save_file()

	print("Saved configuration")
	if printing:
		print(json.dumps(config.to_dict(), indent=4))

def main():
	"""
	Main script
	
	This script steps through setting up the config file and roboflow login:
	1. Log into roboflow (choice of API key or browser prompt)
	2. Model results directory (optional)
	3. Dataset save directory (optional)
	4. Augmentation types (interactive, in order):
	     a. Exposure
	     b. Contrast
	     c. Resize
	     d. Motion Blur
	   (gaussian blur, hue shift, saturation available as arguments only)
	5. Training parameters (epochs, patience, device, model)
	"""
	robosubcv_dir = Path(__file__).parent.parent.parent.absolute()

	print("")
	print("Thank you for using the RobosubCV repository to train your YOLOv8 model!")
	print("Follow the next steps to setup your configuration.json file")
	print("If you don't want to save this information, use arguments in the 'train.sbatch' script in the 'sbatch' folder")
	print("(to figure out the arguments run: 'poetry run python src/train.py -h', arguments always overwrite the save file)")
	print("----------------------------------------------------------------------------------------------------------------")
	print("")

	# ── 1. Roboflow login ─────────────────────────────────────────────────────
	print("First you need to log into Roboflow in this terminal, you can either use an API key or login with Roboflow's url prompt.")
	answer = input("Do you wish to use an API key (press enter or n for Roboflow prompt)? [y/n] ")

	api_key = None
	if _check_yes(answer):
		print("To find your API key go to app.roboflow.com and go into your 'account settings' (bottom left corner with profile pic)")
		print("Under 'Workspace' in the top left, select which project has the datasets you want to train")
		print("Then go to the 'API Keys' section and copy your 'Private Key'. NOTE: the configuration.json file that this will save to is not secure, use ^C to exit and start over")
		api_key = input("Paste your API Key here: ")
		rf = roboflow_login(api_key=api_key)
	else:
		rf = roboflow_login()
	
	if rf is None:
		print("Log in failed, try again or use API")
		sys.exit(1)
	else:
		print("Successful Login!")

	try:
		workspace = rf.workspace()
		print(f"Default workspace: {workspace.name}")
	except:
		print(f"Couldn't find a default workspace with login")
	
	# ── 2. Results directory ──────────────────────────────────────────────────
	result_str = input("\nType in your preferred results directory for the YOLO output to go to (press Enter for default): ")
	result_path = Path(result_str)
	if not result_str.strip():
		result_path = robosubcv_dir / "results/"
	elif not result_path.exists():
		answer = input(f"Directory: {result_path}, does not exist. Do you want it to be created? [y/n] ")
		if _check_yes(answer):
			print(f"Creating, and saving directory {result_path}...")
			result_path.mkdir(parents=True)
		else:
			print(f"Not creating, saving default directory. Re-run script to change again.")
			result_path = robosubcv_dir / "results/"
	
	result_path = result_path.absolute().resolve()
	if not result_path.exists():
		print(f"ERROR: failed to save dataset directory at: {result_path}, exiting...")
		sys.exit(1)
	print(f"Saving models to {result_path}\n")

	# ── 3. Dataset directory ──────────────────────────────────────────────────
	dir_str = input("Type in your preferred save directory for image training datasets (press Enter for default): ")
	dir_path = Path(dir_str)
	if not dir_str.strip():
		dir_path = robosubcv_dir / "data/"
	elif not dir_path.exists():
		answer = input(f"Directory: {dir_path}, does not exist. Do you want it to be created? [y/n] ")
		if _check_yes(answer):
			print(f"Creating, and saving directory {dir_path}...")
			dir_path.mkdir(parents=True)
		else:
			print(f"Not creating, saving default directory. Re-run script to change again.")
			dir_path = robosubcv_dir / "data/"
	
	dir_path = dir_path.absolute().resolve()
	if not dir_path.exists():
		print(f"ERROR: failed to save dataset directory at: {dir_path}, exiting...")
		sys.exit(1)
	print(f"Saving datasets to {dir_path}\n")

	# ── 4. Augmentation ───────────────────────────────────────────────────────
	print("Now enter the augmentation values for each type.")
	print("Press 'Enter' for the default values, press '0' for no augmentation.")
	print("For each type, enter a comma-separated list of percentages (no spaces).")
	print("Example: '0.7,1,1.3' applies augmentation at 70% (less), 100% (no change), and 130% (more).")
	print("(gaussian blur, hue shift, and saturation are available as arguments: -gb, -hs, -sat)")

	#	a. Exposure
	exposure = _prompt_aug_list(
		"Exposure",
		default=[0.615, 1.0, 1.385],
		default_label="-38.5%, original, +38.5%  (±3 EV for RealSense camera)"
		)

	#	b. Contrast  (primary photometric aug, replaces saturation in the interactive flow)
	contrast = _prompt_aug_list(
		"Contrast",
		default=[0.7, 1.0, 1.3],
		default_label="-30%, original, +30%"
		)

	#	c. Resize
	resize = _prompt_aug_list(
		"Resize",
		default=None,
		default_label="no default resize augmentation"
		)

	#	d. Motion Blur
	motion_blur = _prompt_aug_list(
		"Motion Blur",
		default=None,
		default_label="no default motion blur augmentation"
		)

	# ── 5. Augmentation strategy ──────────────────────────────────────────────
	print("\nHow should augmentations be applied to each image?")
	print("  all        — one image per value per aug type (can oversample)")
	print("  random N   — N randomly combined augmented images per original")
	print("  calc N     — N maximally-varied augmented images per original (deterministic)")
	print("Press Enter for default (calc 3).")

	augment_mode = "calc"
	augment_num  = 3
	aug_answer = input("\n    Augment mode [all / random N / calc N]: ").strip().lower()
	if aug_answer == "" or aug_answer == "calc":
		augment_mode, augment_num = "calc", 3
		print(f"Saving default: calc 3")
	elif aug_answer == "all":
		augment_mode = "all"
		print(f"Saving: all")
	else:
		parts = aug_answer.split()
		if parts[0] in ("random", "calc"):
			augment_mode = parts[0]
			if len(parts) >= 2:
				try:
					augment_num = int(parts[1])
					print(f"Saving: {augment_mode} {augment_num}")
				except ValueError:
					print(f"Invalid number '{parts[1]}', defaulting to {augment_mode} 3")
					augment_num = 3
			else:
				print(f"No number given, defaulting to {augment_mode} 3")
		else:
			print(f"Unrecognised input '{aug_answer}', defaulting to calc 3")

	# ── 6. Saving ─────────────────────────────────────────────────────────────
	print(f"\nSaving config file at {Config.path}")
	config = Config(
		api_key      = api_key,
		save_dir     = result_path,
		dataset_dir  = dir_path,
		exposure     = exposure,
		contrast     = contrast,
		resize       = resize,
		motion_blur  = motion_blur,
		augment_mode = augment_mode,
		augment_num  = augment_num,
		)
	config.save_file()
	print("Finished setup, final file values: ")
	print(json.dumps(config.to_dict(), indent=4))

if __name__ == "__main__":
	arg_dict = _parse_args()
	if arg_dict:
		alturnate_main(arg_dict)
	else:
		main()