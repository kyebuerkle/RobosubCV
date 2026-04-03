#!/bin/bash

set -e  # exit on error

DEBUG=1   # 0 = silent (errors only), 1 = verbose
ENV_NAME="Training"
PYTHON_VERSION="3.11"

# log functions if DEBUG = 1
log() 
{
    if [ "$DEBUG" -eq 1 ]; then
        echo "[INFO] $1"
    fi
}

error() 
{
	echo "[ERROR] $1"
}

log "========== ENV SETUP START =========="

# go to Anaconda
if ! command -v conda &> /dev/null; then
    error "Conda not found. Please install Anaconda/Miniconda."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# create and activate environment
log "Checking for conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    log "Environment '$ENV_NAME' exists."
else
    log "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y || {
        error "Failed to create conda environment"
        exit 1
    }
fi

log "Activating environment..."
conda activate $ENV_NAME || {
    error "Failed to activate environment"
    exit 1
}

# checking pip and torch
log "Upgrading pip..."
pip install --upgrade pip

log "Checking for existing torch installation..."

python - <<EOF
import importlib.util
exit(0 if importlib.util.find_spec("torch") else 1)
EOF

TORCH_EXISTS=$?

if [ $TORCH_EXISTS -eq 0 ]; then
    log "Torch already installed. Skipping installation."
else
    log "Torch not found. Detecting CUDA version..."

    CUDA_VERSION=""

    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log "Detected CUDA version: $CUDA_VERSION"
    else
        echo "[WARN] nvidia-smi not found. Cannot detect CUDA."
    fi

    # Install correct torch version
    if [[ "$CUDA_VERSION" == 12* ]]; then
        log "Installing PyTorch for CUDA 12.x (cu121)..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121 || {
            error "Failed to install cu121 torch"
            exit 1
        }

    elif [[ "$CUDA_VERSION" == 11* ]]; then
        log "Installing PyTorch for CUDA 11.x (cu118)..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 || {
            error "Failed to install cu118 torch"
            exit 1
        }

    else
        echo "[WARN] Unknown or no CUDA detected. Installing default torch..."
        pip install torch torchvision torchaudio || {
            error "Failed to install default torch"
            exit 1
        }
    fi
fi

# Verify torch works
log "Verifying torch installation..."

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# Install poetry
log "Ensuring Poetry is installed..."

if ! command -v poetry &> /dev/null; then
    log "Installing Poetry..."
    pip install poetry || {
        error "Failed to install Poetry"
        exit 1
    }
else
    log "Updating Poetry..."
    pip install --upgrade poetry
fi

# Install poetry dependencies
log "Running poetry install..."

poetry install || {
    error "Poetry install failed"
    exit 1
}

log "========== ENV SETUP COMPLETE =========="