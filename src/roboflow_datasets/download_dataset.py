#	download_dataset.py
#	Kye Buerkle
#	brief: downloads a roboflow dataset
#	usage: download_dataset.py [options]

import argparse
import sys
import os
import shutil
from dataset_module import CONFIG_PATH, load_json, roboflow_login, save_json

# --------- Main Function ----------
def roboflow_download():
	if os.path.exists(CONFIG_PATH):
		config = load_json(CONFIG_PATH)
	else:
		config = {}

	# parse the arguments
	parser = argparse.ArgumentParser(
		prog = "download_dataset.py",
		description = "downloads a dataset from roboflow"
		)

	parser.add_argument('-d', "--directory", help = "output directory", default = config.get("directory"))
	parser.add_argument('-w', "--workspace", help = "roboflow workspace", default = config.get("workspace"))
	parser.add_argument('-p', "--project", help = "Project ID from rboflow", default = config.get("project"))
	parser.add_argument('-v', "--version", help = "versions of project, default 1", type=int, default = config.get("version"))
	parser.add_argument('-k', "--key", help = "api key for Roboflow login", default=None)
	parser.add_argument('-f', "--format", help = "model format of images", default = "coco")
	parser.add_argument('-y', "--yes", help = "accepts the overwrite without waiting for user input", action="store_true")
	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()

	#	check parsed args
	directory_r = args.directory
	workspace_r = args.workspace
	project_r = args.project
	version_r = args.version
	# defaults if not in config or argument
	if (directory_r is None) or (directory_r == ""):
		dir_path = os.path.dirname(os.path.abspath(__file__))
		directory_r = os.path.abspath(f"{dir_path}/../../data")
	if (workspace_r is None) or (workspace_r == ""):
		print("No 'workspace' in config, use -w to determine the workspace")
		sys.exit(1)
	if (project_r is None) or (project_r == ""):
		print("No 'project' in config, use -p to determine the project")
		sys.exit(1)
	if (version_r is None) or (version_r == 0):
		version_r = 1

	output_dir = os.path.join(directory_r, f"{project_r}-{version_r}")
	if os.path.exists(output_dir):
		print("Dataset already exists, Do you wish to overwrite?")
		if args.yes:
			response = "y"
		else:
			response = input("[y/n] ")
		if response.lower() == "y":
			try:
				shutil.rmtree(output_dir)
			except Exception as e:
				print(f"Failed to delete previous dataset at: {output_dir}\n{e}")
				sys.exit(1)

	#	login to roboflow & download dataset
	try:
		rf = roboflow_login(api_key= args.key)
		project = rf.workspace(workspace_r).project(project_r)
		version = project.version(version_r)
		dataset = version.download(model_format = args.format, location = output_dir)
		print(f"Path to dataset: {dataset.location}")
		print(f"Dataset name: {dataset.name}")
		print(f"Dataset version: {dataset.version}")
	except Exception as e:
		print(f"Failed to download dataset from config file: {CONFIG_PATH}\n{e}")
		sys.exit(1)

	#	saving new json config
	save_json(CONFIG_PATH, directory_r, workspace_r, project_r, version_r)
	return dataset

if __name__ == "__main__":
	roboflow_download()