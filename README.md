#		RobosubCV

Robosub Computer Vision is used for the Montana State University RoboCat's AUV team. This program is designed to use Roboflow and Tempest to train a YOLOv8 computer vision model to detect objects during their RoboSub competion. 

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
git clone [key]
```
[key] is found in [GitHub](https://github.com/kyebuerkle/RobosubCV) -> Code -> Code (Green button)
Select either HTTPS or SSH, depending on your method of authentication.

get feature branches by doing
```
git branch -a
git checkout -t origin/[branch name]
```
go to [usage document](./git_usage.md) for feature making, merging and reviewing.

> ![NOTE]
> git branch -a | grep "brach name" 
> keyword search in branches

> ![NOTE]
> git checkout [local name] origin/[remote name]
> use this for custom branch names on your local device

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
