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

# .json settings file path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roboflow_config.json")

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_json(**dict):
	if not dict.get("config_path"):
		print(f"Failed to save json file with config: {dict}")
		return
	with open(dict.get("config_path"), "w") as f:
		json.dump(dict, f, indent = 2)

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

#	@brief: configures the config dict
#	@params: argv dictionary {directory, dataset, workspace, project, version, key, format, yes}
#	@returns: None on failure, config dictionary on success
def config_dict(config_path, **argv):
	dir_path = os.path.dirname(os.path.abspath(__file__))
	directory_r = os.path.abspath(f"{dir_path}/../../data")
	config = {
			"directory" : directory_r,
			"dataset" : "",
			"workspace" : "",
			"project" : "",
			"version" : 1,
			"key" : None,
			"format" : "coco",
			"yes" : False,
			"verbose": False
		}

	if os.path.exists(config_path):
		json_file = load_json(config_path)
		config.update(json_file)

	#	parsing URL argument
	if "url" in argv:
		temp_space = argv.pop("url").split("/")
		try:
			version_temp = int(temp_space[-1])
		except:
			return None
		
		argv.update({
			"workspace": temp_space[-3], 
			"project": temp_space[-2], 
			"version": version_temp
			})

	config.update(argv)

	#	check args		
	if (config.get("workspace") is None) or (config.get("workspace") == ""):
		print("No 'workspace' in config, use -w to determine the workspace")
		return None
	if (config.get("project") is None) or (config.get("project") == ""):
		print("No 'project' in config, use -p to determine the project")
		return None

	config.update(dataset = f"{config.get("project")}-v{config.get("version")}")
	config.update(config_path = config_path)
	return config

#	@brief: download dataset from roboflow
#	@param: config -> config dictionary
def roboflow_download(config):
	output_dir = os.path.join(config.get("directory"), config.get("dataset"))
	if os.path.exists(output_dir):
		print("Dataset already exists, Do you wish to overwrite?")
		if config.get("yes"):
			response = "y"
		else:
			response = input("[y/n] ")
		if response.lower() == "y":
			try:
				shutil.rmtree(output_dir)
			except Exception as e:
				print(f"Failed to delete previous dataset at: {output_dir}\n{e}")
				return False
			
	#	login to roboflow & download dataset
	try:
		rf = roboflow_login(api_key= config.get("key"))
		project = rf.workspace(config.get("workspace")).project(config.get("project"))
		version = project.version(config.get("version"))
		dataset = version.download(model_format = config.get("format"), location = output_dir)
	except Exception as e:
		print(f"Failed to download dataset from config file: {config.get("config_path", "no path")}\n{e}")
		return False
	#	saving new json config
	config_save = {k: config.get(k) for k in ("directory", "workspace", "project", "version")}
	save_json(**config_save)
	return dataset

#	@brief: uploads dataset to roboflow
#	@param: config -> config dictionary
def roboflow_upload(config):
	input_dir = os.path.join(config.get("directory"), config.get("dataset"))
	if not os.path.exists(input_dir):
		print(f"No path exists to upload: {input_dir}")
		return
	
	try:
		rf = roboflow_login(api_key=config.get("key"))
		workspace = rf.workspace(config.get("workspace"))
		workspace.upload_dataset(
			input_dir,
			config.get("project"),
			#dataset_format = config.get("format"),
			project_license = "MIT",
    		project_type = "object-detection"
			)
	except Exception as e:
		print(f"Failed to upload dateset from config file: {config.get("config_path", "no path")}")
		return
	
	#	saving new json config
	config_save = {k: config.get(k) for k in ("directory", "workspace", "project", "version")}
	save_json(**config_save)

#	@brief: argument parser for anything roboflow related
#	@return: config dictionary, nOne if failed
#	NOTE: I used this in 3 scripts so it belonged as a funciton
def robo_arg_parse(config_path = CONFIG_PATH):
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
	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()
	arg_dict = vars(args)

	config = config_dict(config_path, **arg_dict)
	if config is None:
		print("Failed to configure arguments")
		return None
	
	return config