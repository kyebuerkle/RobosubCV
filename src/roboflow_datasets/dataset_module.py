#	dataset_module.py
#	Kye Buerkle
#	brief: module for the dataset library
#		This library manages all the Roboflow functionality
#		Uploading and Downloading

import argparse
import roboflow
import json
import os
import shutil

from general_lib import parse_float_list, parse_int_list

#	@brief: logs into roboflow with api, or user
#	@param: api_key, when none login with user
#	@return: roboflow.Roboflow -> rf for roboflow login
def roboflow_login(api_key = None):
	try:
		if (api_key != None):
			rf = roboflow.Roboflow(api_key = api_key)
		else:
			roboflow.login()
			rf = roboflow.Roboflow()
		return rf

	except Exception as e:
		print(f"Failed to log into Roboflow, exception: {e}")
		return None

#	@brief: argument parser for anything roboflow related
#	@return: argument dictionary, non on fail
#	NOTE: I used this in 3 scripts so it belonged as a funciton
def robo_arg_parse():
	parser = argparse.ArgumentParser(
		prog = "download_dataset.py",
		description = "downloads a dataset from roboflow",
		argument_default = argparse.SUPPRESS,
		epilog="""
	If you want to use these arguments in the 'train.sbatch' script instead of the configuration.json file settup
	I recomeng using this format:
	'''train.sbatch
	#	Past the URL to your dataset here
	ROBOFLOW_URL=[ROBOFLOW_URL]

	#	argument values
	SAVE_DIRECTORY=../saving/here
	CONTRAST=0.7,1,1.3
	EXPOSURE=2,1,0.5
	MOTION_BLUR=0.5,1,1.5

	poetry install
	poetry run python $HOME/RobosubCV/src/model_training/train.py -u $ROBOFLOW_URL -s $SAVE_DIRECTORY -con $CONTRAST -exp $EXPOSURE -mb $MOTION_BLUR
	'''
		"""
		)
	
	parser.add_argument('-u', "--url",       help = "Input the URL of the roboflow project")
	parser.add_argument('-d', "--directory", help = "output directory")
	parser.add_argument('-w', "--workspace", help = "roboflow workspace")
	parser.add_argument('-p', "--project",   help = "Project ID from roboflow")
	parser.add_argument('-v', "--version",   help = "versions of project, default 1", type=int)
	parser.add_argument('-k', "--key",       help = "api key for Roboflow login")
	parser.add_argument('-f', "--format",    help = "model format of images")
	parser.add_argument('-y', "--yes",       help = "accepts the overwrite without waiting for user input", action="store_true")
	parser.add_argument('-s', "--save",      help = "save directory for the training model")

	#	── Augmentation arguments ──────────────────────────────────────────────
	parser.add_argument('-exp', "--exposure",
		help = "exposure augmentation values, comma separated eg: 0.5,1,1.75"
		)
	parser.add_argument('-con', "--contrast",
		help = "contrast augmentation values, comma separated eg: 0.7,1,1.3"
		)
	parser.add_argument('-res', "--resize",
		help = "scale / resize augmentation values, comma separated eg: 0.5,1,1.75,3"
		)
	parser.add_argument('-mb', "--motion_blur",
		help = "motion blur augmentation values, comma separated eg: 0.5,1,1.5"
		)
	parser.add_argument('-gb', "--gaussian_blur",
		help = "gaussian blur augmentation values, comma separated eg: 0.5,1,1.5"
		)
	parser.add_argument('-hs', "--hue_shift",
		help = "hue shift augmentation values, comma separated eg: 0.5,1,1.5"
		)
	parser.add_argument('-sat', "--saturation",
		help = "(legacy) saturation augmentation values, comma separated eg: 0.5,1,1.75"
		)
	parser.add_argument('--augment',
		nargs='+',
		metavar=('MODE', 'NUM'),
		help = "Augmentation strategy: 'all', 'random 3', or 'calc 3'"
		)

	#	── Training parameters ─────────────────────────────────────────────────
	parser.add_argument("--epochs",   help = "epoch parameter for model training")
	parser.add_argument("--patience", help = "patience parameter for model training")
	parser.add_argument("--imgsz",    help = "imgsz parameter for model training")
	parser.add_argument("--seed",     help = "seed parameter for model creation")
	parser.add_argument("--device",   help = "device parameter for model creation")
	parser.add_argument("--model",
		help = "model parameter for model training, yolov8m.pt <- medium model, "
		       "replace the m with n for nano, s for small, l for large, and x for extra large"
		)

	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()
	arg_dict = vars(args)

	#	deletes key if nothing was done to it
	key_temp = arg_dict.get("key", None)
	if (not key_temp is None) and (key_temp == "[API_KEY]"):
		arg_dict.pop("key")

	#	parsing float lists — all augmentation args
	for key in ["exposure", "contrast", "resize", "motion_blur", "gaussian_blur", "hue_shift", "saturation"]:
		if key in arg_dict:
			val = arg_dict.get(key)
			arg_dict[key] = list(parse_float_list(val))

	if "device" in arg_dict:
		val = arg_dict.get("device")
		arg_dict["device"] = list(parse_int_list(val))

	#	Parse --augment MODE [NUM] into augment_mode / augment_num
	if "augment" in arg_dict and arg_dict["augment"]:
		raw  = arg_dict.pop("augment")
		mode = raw[0].lower()
		if mode not in ("all", "random", "calc"):
			print(f"Warning: unknown --augment mode '{mode}', defaulting to 'all'")
			mode = "all"
		arg_dict["augment_mode"] = mode
		if len(raw) >= 2:
			try:
				arg_dict["augment_num"] = int(raw[1])
			except ValueError:
				print(f"Warning: --augment NUM '{raw[1]}' is not an integer, defaulting to 3")
				arg_dict["augment_num"] = 3
	else:
		arg_dict.pop("augment", None)

	return arg_dict