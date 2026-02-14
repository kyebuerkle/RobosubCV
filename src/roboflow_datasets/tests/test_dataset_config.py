#	@file: test_dataset_module.py
#	@brief: unit test for the dataset_module

import pytest
import os
import shutil
from roboflow_datasets.dataset_config import Config

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_test.json")

class TestDatasetModule:

	@pytest.fixture
	def directory_in(self):
		dir_path = os.path.dirname(os.path.abspath(__file__))
		return os.path.abspath(f"{dir_path}/../../../data")

	#	tests if there is no config file
	def test_no_file(self, directory_in):
		# must remove file first
		if (os.path.exists(CONFIG_PATH)):
			os.remove(CONFIG_PATH)

		input_args = {
			"workspace": "",
			"project": "",
			"path": CONFIG_PATH
			}
		results = Config(input_args)
		assert results.workspace == ""
		assert results.project == ""

		input_args = {
			"workspace": "test-workspace",
			"project": "",
			"path": CONFIG_PATH
			}
		results = Config(input_args)
		assert results.workspace == "test-workspace"

		input_args = {
			"workspace": "",
			"project": "test-project",
			"path": CONFIG_PATH
			}
		results = Config(input_args)
		assert results.project == "test-project"

		input_args = {
			"workspace": "test-workspace",
			"project": "test-project",
			"path": CONFIG_PATH
			}
		results = Config(input_args)

		assert results.workspace == "test-workspace"
		assert results.project == "test-project"
		assert results.path == CONFIG_PATH
		assert results.version == 1
		assert results.dataset_dir == directory_in
		assert results.get_dataset() == os.path.join(directory_in, "test-project-v1")

		results.save_file()

	def test_with_file(self, directory_in):
		if (not os.path.exists(CONFIG_PATH)):
			pytest.fail(f"Config file ({CONFIG_PATH}) doesn't exist")
			return
		
		results = Config(path=CONFIG_PATH)
		assert results.workspace == "test-workspace"
		assert results.project == "test-project"
		assert results.path == CONFIG_PATH
		assert results.version == 1
		assert results.dataset_dir == directory_in
		assert results.get_dataset() == os.path.join(directory_in, "test-project-v1")

		results.save_file()
		
	def test_url_parse(self, directory_in):
		if (not os.path.exists(CONFIG_PATH)):
			pytest.fail(f"Config file ({CONFIG_PATH}) doesn't exist")
			return
		
		input_args = {
			"url": "https://app.roboflow.com/workspace-url/project-id/3",
			"path": CONFIG_PATH
			}
		results = Config(input_args)
		assert results.workspace == "workspace-url"
		assert results.project == "project-id"
		assert results.path == CONFIG_PATH
		assert results.version == 3
		assert results.dataset_dir == directory_in
		assert results.get_dataset() == os.path.join(directory_in, "project-id-v3")

		results.save_file()
		
	def test_cleanup(self):
		if os.path.exists(CONFIG_PATH):
			os.remove(CONFIG_PATH)

		assert not os.path.exists(CONFIG_PATH)