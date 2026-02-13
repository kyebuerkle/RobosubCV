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
		argument_default = argparse.SUPPRESS
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
	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()
	arg_dict = vars(args)

	#	deletes key if nothing was done to it
	key_temp = arg_dict.get("key", None)
	if (not key_temp is None) and (key_temp == "[API_KEY]"):
		arg_dict.pop("key")

	return arg_dict