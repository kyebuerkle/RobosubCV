#	dataset_module.py
#	Kye Buerkle
#	brief: module for the dataset library
#		This library manages all the Roboflow functionality
#		Uploading and Downloading

import roboflow
import json
import os
import sys

# .json settings file path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roboflow_config.json")

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, directory, workspace, project, version):
	with open(path, "w") as f:
		json.dump(
			{
				"directory": directory,
				"workspace": workspace,
				"project": project,
				"version": version
			},
			f,
			indent = 2
			)

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
		sys.exit(1)
		return None