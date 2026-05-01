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
#SBATCH --time=0-12:00:00
#SBATCH --job-name=RobosubCV
## %x is the job name and %j and the job number
#SBATCH --output=./stream-out/%x-%j.out
#SBATCH --error=./stream-out/%x-%j.err

module load Anaconda3/4.0.0
source activate $HOME/.conda/envs/Training

#	Past the URL to your dataset here
ROBOFLOW_URL=[ROBOFLOW_URL]

poetry install

# Debug block - remove after fixing
echo "=== ENV DEBUG ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"
nvidia-smi
echo "================="

poetry run python $HOME/RobosubCV/src/model_training/train.py -u $ROBOFLOW_URL
