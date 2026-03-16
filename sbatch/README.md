#		Sbatch

Tempest uses Slurm requests to run programs in their queue. Sbatch -> Slurm Batch, is batch scripts that work with Tempest.

##		train.sbatch

This is the main training program. It runs `train.py` with certain options to run in Tempest properly. This program will both download a dataset, and train the YOLO model

###		Usage

To use this, first login you must be in the Tempest web: [Tempest](https://tempest-web.msu.montana.edu/). And go to the terminal (Files > Home Directory > *button* >_Open in Terminal). Before using this script, follow the steps in `../src/model_training/setup_settings.py` to configure the settings file this script relies on.

1. `git pull` : update the git repo, if you don't have it then follow [README](../README.md) instructions to clone it
   1. make sure you are in the right branch *framework/training*
2. `nano train.sbatch` : opens the file so you can edit
   1. save by pressing *ctrl + s* (^S)
   2. exit by pressing *ctrl + x* (^X)
3. For the variable `ROBOFLOW_URL` paste in your Roboflow project URL
   1. In (roboflow)[app.roboflow.com], navigate to the project you want to download, and just copy the URL at the top
4. save and exit the nano file (^S & ^X)
5. `sbatch train.sbatch` to queue your job
   1. terminal should say: `Submitted batch job ######` on success

> ![NOTE]
> Make sure to follow the Environment setup **FIRST** : [tempest_setup](./tempest_setup.md)

To double check you have the environment run: `conda env list` and one of the environments should have *Training*. 

###      Other Usage

The `train.py` script has a variety of arguments if you don't want to use the configuration.json file from the setup_settings.py script, look at the [README](../src/model_training/README.md) to configure these arguments.
Use variable format `VARIABLE=val` where there is no space, and call the variable via `$VARIABLE` (normal batch script format).

###		Roboflow API & URL

You can use the API key to sign in, or the URL prompt when running the `setup_settings.py` script.
To access your API Key go to [app.roboflow.com](https://app.roboflow.com) and sign in.

1. Go to your *Account Settings* 
   1. press your profile in the bottom left corner
   2. select *Account Settings*
2. Select your current Workspace
   1. This is the workspace that your project is in
3. Under your Workspace, select *API Keys*
4. Copy your *Private API Key*

To get your URL:

1. Go to your Workspace in the top left corner of Roboflow
2. Go to the project you wish to train the model on
3. Make sure the sected version is the one you want to train
4. Copy the URL at the top
   1. should look like: `https://app.roboflow.com/WORKSPACE-NAME/PROJECT-NAME/VERSION`

##    tests

The *tests* directory is used for other Slurm Batch scripts for testing Specifications and Requirements during the Verification stage of Capstone.

Discriptions on how to use them are in the Final Project Report, under the Verifications chapter.

The specific reqs and specs that this repo covers is:
- Spec 1.3.1
- Spec 1.3.2
- Req 1.3
- Spec 1.2.1
- Req 1.2
- Req 1.4
- Spec 1.1.1
- Req 1.1
- Obj 1
  
(Chronologically in the order the Capstone team tested the specs)