#!/bin/bash

## priority-bradwhitaker or group-bradwhitaker
## priority faster for gpupriority (2 A100 or 2 A40); H100 nextgen-gpu (4 H100 and 3 nodes) faster, but wait longer
#SBATCH --account=priority-bradwhitaker
#SBATCH --partition=gpupriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=a40:1
## 1000G max for H100
## 500GB for Quad A100
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --job-name=RobosubCV-env
## %x is the job name and %j and the job number
#SBATCH --output=./stream-out/%x-%j.out
#SBATCH --error=./stream-out/%x-%j.err

module load Anaconda3/4.0.0
bash $HOME/RobosubCV/env_setup.sh