#		Sbatch

Tempest uses Slurm requests to run programs in their queue. Sbatch -> Slurm Batch, is batch scripts that work with Tempest.

##		train.sbatch

This is the main training program. It runs `train.py` with certain options to run in Tempest properly. This program will both download a dataset, and train the YOLO model

###		Usage

To use this, first login you must be in the Tempest web: [Tempest](https://tempest-web.msu.montana.edu/). And go to the terminal (Files > Home Directory > *button* >_Open in Terminal). Do the following steps in the terminal.

1. `git pull` : update the git repo, if you don't have it then follow [README](../README.md) instructions to clone it
   1. make sure you are in the right branch *framework/training*
2. `nano train.sbatch` : opens the file so you can edit
   1. save by pressing *ctrl + s* (^S)
   2. exit by pressing *ctrl + x* (^X)
3. For the variable `ROBOFLOW_API` paste in your Roboflow Api key
   1. read below how to access that
4. For the variable `ROBOFLOW_URL` paste in your Roboflow project URL
   1. In (roboflow)[app.roboflow.com], navigate to the project you want to download, and just copy the URL at the top
5. save and exit the nano file (^S & ^X)
6. `sbatch train.sbatch` to queue your job
   1. terminal should say: `Submitted batch job ######` on success

> ![NOTE]
> Make sure to follow the Environment setup **FIRST** : [tempest_setup](./tempest_setup.md)

To double check you have the environment run: `conda env list` and one of the environments should have *Training* at the end

###		Roboflow API & URL

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

Note from Kye: This process is only this tedius for our **Minimal Working System**. Just so that we can integrate the other subsystems without the middle portions like the augmentation. I will try to find simpler ways of doing this (like saving the API key, or logging into Roboflow some other way).
The URL will be neccassary no matter what, as there isn't a simpler way of selecting the project someone wants to train.
The API and loggin, in the future, will be a One Time settup before competition. 