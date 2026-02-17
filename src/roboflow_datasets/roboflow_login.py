#	@file: roboflow_login.py
#	@brief: Logs the user into roboflow to save it for training

import sys
from roboflow_datasets import roboflow_login

if __name__ == "__main__":
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
		print(f"Couldn't find a default workspace with loggin")