#	download_dataset.py
#	Kye Buerkle
#	brief: downloads a roboflow dataset
#	usage: download_dataset.py [options]

import roboflow
import argparse
import sys
import os

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

if __name__=="__main__":
	#	default data path
	dir_path = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.abspath(f"{dir_path}/../../data")

	# parse the arguments
	parser = argparse.ArgumentParser(
		prog = "download_dataset.py",
		description = "downloads a dataset from roboflow"
		)

	parser.add_argument('-k', "--key", help = "api key for Roboflow login", default=None)
	parser.add_argument('-f', "--format", help = "model format of images", default = "coco")
	parser.add_argument('-d', "--directory", help = "output directory", default = data_path)
	parser.add_argument('-w', "--workspace", help = "roboflow workspace")
	parser.add_argument("project", help = "Project ID from rboflow")
	parser.add_argument("version", help = "versions of project, default 1", type=int)
	args = parser.parse_args()

	#	check parsed args
	workspace = args.workspace.replace(" ", "-")
	output_dir = os.path.join(args.directory, f"{args.project}-{args.version}")

	#	login to roboflow
	rf = roboflow_login(api_key= args.key)
	project = rf.workspace(workspace).project(args.project)
	version = project.version(args.version)
	dataset = version.download(model_format = args.format, location = output_dir)
	print(f"Path to dataset: {dataset.location}")
	print(f"Dataset name: {dataset.name}")
	print(f"Dataset version: {dataset.version}")