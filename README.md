#		RobosubCV

Robosub Computer Vision is used for the Montana State University RoboCat's AUV team. This program is designed to use Roboflow and Tempest to train a YOLOv8 computer vision model to detect objects during their RoboSub competion. 

##    Usage

1. You have to setup your environment if you are using Tempest. Go through the **Model Creation: Setup** portion of the RoboSubCV Procedure.
   1. This will also setup your git and poetry. More notes on that are down below in the **Git & Poetry** section
2. Configure your settings. Run this command:
   ```
   poetry run python src/model_training/setup_settings.py
   ```
   1. This will take you through a series of steps to configure your settings (I recomend pressing 'Enter' all the way through as it will keep the defaults)
3. Go to your [roboflow](https://app.roboflow.com) account in a browser, and select which project and version of dataset you want to train off of. Then copy the URL.
4. **IF** you are using Tempest, go to the `/sbatch/train.sbatch` script, and paste your roboflow URL in the *ROBOFLOW_URL* variable. Then run `sbatch sbatch/train.sbatch` to queue your Tempest job.
4. **IF** you aren't using Tempest, type: `/src/train.py -u [URL]` and paste your URL in the *[URL]* spot. Run this script
   1. NOTE: You are in charge of managing your GPU's, this script is currently set up for 2, so you will have to edit that in the script or add it as a parameter.

##		Git & Poetry

Overview of how to set up the Git repository, and python Poetry environment.
(Note: Poetry works on tempest, but as of now we've always used Anaconda).

Everytime you use this repo:
1. `git pull`
   1. gets latest changes to repo
2. `poetry install`
   1. installs poetry dependencies from .lock file

###		Git Setup

Setup the repository on your personal computer by using 
```
git clone git@github.com:kyebuerkle/RobosubCV.git
```
The key is found in [GitHub](https://github.com/kyebuerkle/RobosubCV) -> Code -> Code (Green button)
Select either HTTPS or SSH, depending on your method of authentication.

get feature branches by doing
```
git branch -a
git checkout -t origin/[branch name]
```
go to [usage document](./git_usage.md) for feature making, merging and reviewing.


NOTE: git branch -a | grep "brach name" keyword search in branches
NOTE: git checkout [local name] origin/[remote name] use this for custom branch names on your local device

###		Poetry Setup

Install the poetry dependencies: (python poetry must be installed first)
```
poetry install
```

Run in poetry environment:
```
poetry run <command>

#	Example1:
poetry run py script.py

#	Example2:
poetry run ./app.exe
```
Go to: https://python-poetry.org/docs/basic-usage/ to install and read through simple poetry commands.

To install poetry run: `pipx install poetry` or `pip install poetry`
(pipx is prefered as it isolates the poetry environment from other dependencies you have installed)

##    Verifications

This repo has unit tests utilizing pytest. To verify them run:
```
poetry run pytest
```

this will run through all unit tests in any directory under 'src'
use `-v` for verbose and get more information.