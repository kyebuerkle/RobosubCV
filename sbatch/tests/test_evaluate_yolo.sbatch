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
#SBATCH --job-name=RobosubCV-Obj-2
## %x is the job name and %j and the job number
#SBATCH --output=./stream-out/%x-%j.out
#SBATCH --error=./stream-out/%x-%j.err

module load Anaconda3/4.0.0
source activate $HOME/.conda/envs/Training
cd $HOME/RobosubCV/src/model_training/tests

MODELS=(
)
DATASET=
SPLIT=(
test
train
valid
)

#	device is, cpu= 1 cpu, 0= 1 GPU, 0,1= 2 GPU
DEVICE=0,1

poetry install
poetry run python evaluate_yolo_labels.py -m "${MODELS[@]}" -d $DATASET --device $DEVICE --split "${SPLIT[@]}"

cd $HOME/RobosubCV/sbatch

