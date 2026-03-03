#!/bin/bash

## priority-bradwhitaker or group-bradwhitaker
## priority faster for gpupriority (2 A100 or 2 A40); H100 nextgen-gpu (4 H100 and 3 nodes) faster, but wait longer
#SBATCH --account=priority-bradwhitaker
#SBATCH --partition=gpupriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:2
## 1000G max for H100
## 500GB for Quad A100
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=RobosubCV
## %x is the job name and %j and the job number
#SBATCH --output=./stream-out/%x-%j.out
#SBATCH --error=./stream-out/%x-%j.err

module load Anaconda3/4.0.0
source activate $HOME/.conda/envs/Training

#	Past the URL to your dataset here
ROBOFLOW_URL=[ROBOFLOW_URL]

#	Augmentation parameters, comma seperated no spaces
EXPOSURE=0.615,1,1.385
SATURATION=0.7,1,1.3

poetry install
poetry run python $HOME/RobosubCV/src/model_training/train.py -u $ROBOFLOW_URL 