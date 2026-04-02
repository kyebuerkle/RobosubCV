#	dataset_module.py
#	Kye Buerkle
#	brief: module for the dataset library
#		This library manages all the Roboflow functionality
#		Uploading and Downloading
#	TODO: It is a little stupid to put all my vars in a dictionary
#		I should make it into a Config class, that way it has functions
#		relating to it (but I'll do this after designing the minimum model)

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
	SATURATINO=0.5,1
	EXPOSURE=2,1,0.5

	poetry install
	poetry run python $HOME/RobosubCV/src/model_training/train.py -u $ROBOFLOW_URL -s $SAVE_DIRECTORY -sat $SATURATION -exp $EXPOSURE
	'''
		"""
		)
	
	parser.add_argument('-u', "--url", help = "Input the URL of the roboflow project")
	parser.add_argument('-d', "--directory", help = "output directory")
	parser.add_argument('-w', "--workspace", help = "roboflow workspace")
	parser.add_argument('-p', "--project", help = "Project ID from rboflow")
	parser.add_argument('-v', "--version", help = "versions of project, default 1", type=int)
	parser.add_argument('-k', "--key", help = "api key for Roboflow login")
	parser.add_argument('-f', "--format", help = "model format of images")
	parser.add_argument('-y', "--yes", help = "accepts the overwrite without waiting for user input", action="store_true")
	parser.add_argument('-s', "--save", help="save directory for the training model")
	parser.add_argument('-sat', "--saturation", help="saturation augmentation values, comma seperated eg: 0.5,1,1.75")
	parser.add_argument('-exp', "--exposure", help="exposure augmentation values, comma seperated eg: 0.5,1,1.75")
	parser.add_argument('-res', "--resize", help="scale / resize augmentation values, comma seperated eg: 0.5,1,1.75,3")
	parser.add_argument("--epochs", help="epoch paramter for model training")
	parser.add_argument("--patience", help="patience paramter for model training")
	parser.add_argument("--imgsz", help="imgsz parameter for model training")
	parser.add_argument("--seed", help="seed paramter for model creation")
	parser.add_argument("--device", help="device paramter for model creation")
	parser.add_argument("--model", help="model paramter for model training, yolov8m.pt <- medium model, replace the m with n for nano, s for small, l for large, and x for extra large")
	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()
	arg_dict = vars(args)

	#	deletes key if nothing was done to it
	key_temp = arg_dict.get("key", None)
	if (not key_temp is None) and (key_temp == "[API_KEY]"):
		arg_dict.pop("key")

	#	parsing lists
	for key in ["exposure", "saturation", "resize"]:
		if key in arg_dict:
			val = arg_dict.get(key)
			arg_dict[key] = list(parse_float_list(val))
	if "device" in arg_dict:
		val = arg_dict.get("device")
		arg_dict["device"] = list(parse_int_list(val))

	return arg_dict