#		Tempest Setup

Tempest is a High Performance Computer (HPC) research cluster, created and owned by Montana State University. Link: https://www.montana.edu/uit/rci/tempest/

##		Log In

In order to access Tempest, each user needs permissions. If you are a student at MSU you get free permissions automatically. To gain more permissions, talk to Dr. Bradley Whitaker, Dr. Trevor Venoy, or contact the Research Cyberinfrastructure team: [406-994-1777](tel:4069941777), <rci-support@montana.edu>.

Use this link: https://montana.qualtrics.com/jfe/form/SV_54uG1LqwrDnDWJw, if you don't have an account. All other questions, go to the main [Tempest](https://www.montana.edu/uit/rci/tempest/) website.

###		Security

Tempest can only be accessed from a **MSU-Secure** internet connection. If you are on campus, then log into the MSU-Secure WiFi with your student account information. 
If you are not on campus, then you will need a VPN, MSU uses Cisco. Follow the instructions here: https://www.montana.edu/uit/computing/desktop/vpn/. 

###		Tempest Web

Once you have an account, and secured internet, you can access Tempest website: https://tempest-web.msu.montana.edu/. This is your Tempest Homepage. (There are other ways to access Tempest, but this is the way we will use for the RoboSub competition).

Setup:
1. Go to the terminal in Tempest
   1. Files > Tempest > _Open in Terminal
2. Setup Environment
   1. More details below 
3. Follow steps in [../README.md](../README.md) to clone repo

###		Setup Environment

Follow these steps to setup Tempest to run the repository.
```
module load Anaconda3/2024.02-1
cd $HOME/RobosubCV
bash env_setup.sh
```

--- *Legacy* ---
```
cd $HOME
module load Anaconda3/2024.02-1
conda create -n Training python=3.12 -y
source activate Training
```
--- *Legacy* ---

This creates our *Training* environment. Use `conda env list` to list your environments to pick from. 
If `conda activate Training` asks for `conda init`. Run these steps:
```
conda init
source ~/.bashrc
conda activate Training

# or use deprecated option instead
source activate Training
```
And you should see (Training) to the left of your Tempest login.

--- *Legacy: env_setup.sh does this for you* ---
In *Training* we need to install torchvision, to manage the GPUs in Tempest, and Poetry, to manage our repository dependencies.
```
pip3 install --upgrade torch torchvision torchaudio
pip install poetry
# disable poetry virtual environment
poetry config virtualenvs.create false
```
--- *Legacy* ---

Use `conda deactivate` to exit the environment.

This environment is used in the (slurm) sbatch scripts when we send jobs to the Tempest queue. 

###      Roboflow

OPTIONAL: you can log into roboflow so you don't need an API Key

1. be in your environment: `source activate Training`
2. run: `poetry run python src/roboflow_datasets/roboflow_login.py`
3. Follow the instructions to authenticate your login
   1. read the README.md to understand usage

Now you can delete the `ROBOFLOW_API` in the train.sbatch script (or just leave it, the script checks)

##		Validate

After setting up Tempest with the [Repository](../README.md) cloned, there are a couple validation scripts. These validate that the environment is set up correctly, that the scripts can run in Tempest jobs, and that the team can get a working YOLO model.

...TODO...