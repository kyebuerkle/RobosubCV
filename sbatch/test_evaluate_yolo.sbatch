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
#SBATCH --job-name=RobosubCV-spec-1-3
## %x is the job name and %j and the job number
#SBATCH --output=./stream-out/%x-%j.out
#SBATCH --error=./stream-out/%x-%j.err

module load Anaconda3/4.0.0
source activate $HOME/.conda/envs/Training
cd $HOME/RobosubCV/src/model_training/tests

MODELS=(
../../../results/1-3-models/both-45.pt
../../../results/1-3-models/exposure-45.pt
../../../results/1-3-models/saturation-45.pt
)
DATASET=$HOME/RobosubCV/data/2025-robosub-on-site-2-v3-tests/data.yaml
CONFIDENCE=0.5
SPLIT=(
test
train
valid
all
)

#	device is, cpu= 1 cpu, 0= 1 GPU, 0,1= 2 GPU
DEVICE=0,1

poetry install
poetry run python test_evaluate_yolo.py -m "${MODELS[@]}" -d $DATASET --device $DEVICE --conf $CONFIDENCE --split "${SPLIT[@]}"

cd $HOME/RobosubCV/sbatch

