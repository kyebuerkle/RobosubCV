#		Windows Usage

This repository in primarily ment for Linux systems, specifically Tempest. I (Kye Buerkle) used a WSL on my Windows device to program and test everything, as well as the Tempest terminal. But some scripts (like `src/live-stream/yolo_streamer.py`) need access to the hardware on your device, so if you have a Windows, you can't use a WSL. 
If you settup a VM it should work, as long as it is Linux (I recomend Ubuntu).

I did find a way to run this on a Windows device.

##		Environment in Windows

Tempest uses 'Anaconda' and you can download it on windows as well.

1. Go to this website: https://www.anaconda.com/download/success to download Anaconda
   1. I recomend using 'Miniconda' as it is a lightweight alternative, and we don't need most of the libraries in Anaconda, just Python (Poetry deals with all other dependencies)
2. Open the Anaconda Promt (Anaconda Powershell works too)
3. Setup your conda environment. If you have access to the Model Creation Procedure, go to step 19, and run everythin in you Anaconda Prompt
   1. run `./env_setup.ps1/` and it should automatically make your env

pip and pip3 might not work... if they don't, run through the **Install Poetry** steps below.

From here, all you need to do is follow normal usage steps! Go to the main [README](./README.md) to see how to `git clone`, `git pull`, and use poetry. 

Whenever you come back to use the Windows Anaconda environment make sure to activate the Training environment:
```conda
conda activate Training
```

Then the classic:
```
git pull
poetry install
```

###		Install Poetry

Run through these to get poetry on your Windows device

In command prompt or conda run:
```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

When this is finished, it should give you steps to add to your environment variable. Those didn't work for me, so the next steps show how to add that to your Path. First copy the path to your poetry, it should look simular to this: `C:\Users\yourname\AppData\Roaming\Python\Scripts\poetry`
(A way to double check this is the right path is type `C:\Users\yourname\AppData\Roaming\Python\Scripts\poetry --version` and it will give you the current poetry version installed).

Then you need to set Poetry to your PATH (environment variable). To do this on Windows, right click the windows startup menu on your task bar (or press 'Win + x'). Select system (or press 'y') and scroll down to *Device Specification*. Select the *Advanced System Settings* button. Then press the *Environment Variables...* button at the bottom. In the Environment Variable, select the Path variable, and add a new path. Add the path that you copied that leads to your poetry executable. 
To check that this worked, save, and go back to your prompt / terminal. type `poetry --version` and it should tell you the Poetry version installed.

###      Legacy env setup

```conda
conda create -n Training python=3.12 -y
conda activate Training
pip install poetry

# disable poetry virtual environment
poetry config virtualenvs.create false
```