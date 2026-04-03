#!/bin/bash

# ==============================================================================
#  env_setup.sh - Cross-platform environment setup entry point
#
#  On Windows (WSL or Git Bash): delegates to env_setup.ps1 via powershell.exe
#  On Linux / macOS:             runs natively
#
#  Usage:  bash env_setup.sh [-v|--verbose] [-q|--quiet] [--force-cpu] [--env NAME]
#
#  Windows users: just run  .\env_setup.ps1  directly in PowerShell instead.
# ==============================================================================

set -e

# -- Default config ------------------------------------------------------------
VERBOSE=0
ENV_NAME="Training"
PYTHON_VERSION="3.11"
FORCE_CPU=0

# -- Argument parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=1 ;;
        -q|--quiet)   VERBOSE=0 ;;
        --force-cpu)  FORCE_CPU=1 ;;
        --env)        ENV_NAME="$2"; shift ;;
        *) echo "[WARN]  Unknown argument: $1" ;;
    esac
    shift
done

# -- Logging helpers -----------------------------------------------------------
log()  { if [ "$VERBOSE" -eq 1 ]; then echo "[INFO]  $1"; fi; }
warn() { echo "[WARN]  $1"; }
err()  { echo "[ERROR] $1"; }

[ "$VERBOSE" -eq 1 ] && set -x
log "========== ENV SETUP START =========="

# -- OS detection --------------------------------------------------------------
# There are three "bash running on Windows" scenarios, all of which should
# delegate to PowerShell rather than try to find a Windows conda from bash:
#
#   Git Bash / MSYS2  -> $WINDIR is set, uname says "Linux" or MINGW*
#   WSL               -> uname says "Linux", /proc/version has "microsoft"
#   Cygwin            -> uname says CYGWIN*
#
detect_os() {
    # Git Bash / MSYS2
    if [ -n "$WINDIR" ] || [ -n "$SYSTEMROOT" ]; then
        echo "windows"; return
    fi
    # WSL (kernel string contains Microsoft or WSL)
    if [ -f /proc/version ] && grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
        echo "wsl"; return
    fi
    case "$(uname -s)" in
        Darwin*)              echo "mac" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        Linux*)               echo "linux" ;;
        *)                    echo "unknown" ;;
    esac
}

OS=$(detect_os)
log "Detected OS: $OS"

# -- Windows / WSL: delegate to PowerShell ------------------------------------
if [ "$OS" = "windows" ] || [ "$OS" = "wsl" ]; then
    if [ "$OS" = "wsl" ]; then
        warn "Running inside WSL. Delegating to Windows PowerShell so conda"
        warn "can find your Windows Anaconda installation."
        warn "Tip: you can also just run  .\\env_setup.ps1  directly in PowerShell."
    else
        log "Windows (Git Bash) detected. Delegating to env_setup.ps1..."
    fi

    # Locate env_setup.ps1 relative to this script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PS_SCRIPT="$SCRIPT_DIR/env_setup.ps1"

    if [ ! -f "$PS_SCRIPT" ]; then
        err "env_setup.ps1 not found in $SCRIPT_DIR"
        err "Both env_setup.sh and env_setup.ps1 must be in the same directory."
        exit 1
    fi

    # Convert the Unix path to a Windows path for powershell.exe
    if command -v cygpath &>/dev/null; then
        WIN_PATH=$(cygpath -w "$PS_SCRIPT")
    elif [ "$OS" = "wsl" ]; then
        # WSL path: /home/... or /mnt/c/... -> use wslpath if available
        if command -v wslpath &>/dev/null; then
            WIN_PATH=$(wslpath -w "$PS_SCRIPT")
        else
            WIN_PATH=$(echo "$PS_SCRIPT" | sed -E 's|^/mnt/([a-zA-Z])/|\1:/|; s|/|\\|g')
        fi
    else
        WIN_PATH=$(echo "$PS_SCRIPT" | sed -E 's|^/([a-zA-Z])/|\1:/|; s|/|\\|g')
    fi

    log "PowerShell script path: $WIN_PATH"

    # Build optional flag array cleanly (safe with set -e)
    PS_EXTRA_ARGS=()
    [ "$VERBOSE"   -eq 1 ] && PS_EXTRA_ARGS+=("-Verbose")
    [ "$FORCE_CPU" -eq 1 ] && PS_EXTRA_ARGS+=("-ForceCpu")

    powershell.exe -ExecutionPolicy Bypass -File "$WIN_PATH" \
        -EnvName "$ENV_NAME" \
        -PythonVersion "$PYTHON_VERSION" \
        "${PS_EXTRA_ARGS[@]}"

    exit $?
fi

# ==============================================================================
#  Native Linux / macOS
# ==============================================================================

# -- Locate and initialise conda -----------------------------------------------
init_conda() {
    if command -v conda &>/dev/null; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
    else
        local candidates=(
            "$HOME/anaconda3"
            "$HOME/miniconda3"
            "$HOME/opt/anaconda3"
            "$HOME/opt/miniconda3"
            "/opt/anaconda3"
            "/opt/miniconda3"
        )
        for dir in "${candidates[@]}"; do
            if [ -f "$dir/etc/profile.d/conda.sh" ]; then
                CONDA_BASE="$dir"; break
            fi
        done
        if [ -z "${CONDA_BASE:-}" ]; then
            err "Conda not found. Please install Anaconda or Miniconda."
            exit 1
        fi
    fi
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    log "Conda initialised from: $CONDA_BASE"
}

init_conda

# -- Create / reuse environment ------------------------------------------------
log "Checking for conda environment: $ENV_NAME"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    log "Environment '$ENV_NAME' already exists - skipping creation."
else
    log "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y || {
        err "Failed to create conda environment."; exit 1
    }
fi

log "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME" || {
    err "Failed to activate environment. Try: conda init bash"; exit 1
}

# -- pip -----------------------------------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip

# -- PyTorch -------------------------------------------------------------------
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
TORCH_EXISTS=$?

if [ "$TORCH_EXISTS" -eq 0 ]; then
    log "PyTorch already installed - skipping."
else
    CUDA_VERSION=""
    if [ "$FORCE_CPU" -eq 1 ]; then
        warn "--force-cpu flag set. Installing CPU-only PyTorch."
    else
        log "Detecting CUDA version..."
        if command -v nvidia-smi &>/dev/null; then
            CUDA_VERSION=$(nvidia-smi 2>/dev/null \
                | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
            log "nvidia-smi reports CUDA: ${CUDA_VERSION:-not found}"
        fi
        if [ -z "$CUDA_VERSION" ] && command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version 2>/dev/null \
                | grep -oP "release \K[0-9]+\.[0-9]+" || true)
            log "nvcc reports CUDA: ${CUDA_VERSION:-not found}"
        fi
        [ -z "$CUDA_VERSION" ] && warn "No CUDA detected - installing CPU-only PyTorch."
    fi

    CUDA_MAJOR="${CUDA_VERSION%%.*}"
    case "$CUDA_MAJOR" in
        12) pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121 ;;
        11) pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu118 ;;
        *)  pip install torch torchvision torchaudio ;;
    esac || { err "Failed to install PyTorch."; exit 1; }
fi

log "Verifying PyTorch..."
python -c "
import torch
print('  torch version :', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  CUDA version  :', torch.version.cuda)
    print('  GPU           :', torch.cuda.get_device_name(0))
"

# -- Poetry --------------------------------------------------------------------
log "Checking Poetry..."
if ! command -v poetry &>/dev/null; then
    log "Installing Poetry..."
    pip install poetry || { err "Failed to install Poetry."; exit 1; }
else
    log "Updating Poetry..."
    pip install --upgrade poetry
fi

log "Configuring Poetry to use the current environment..."
poetry config virtualenvs.create false --local 2>/dev/null || true

log "Running poetry install..."
poetry install || { err "poetry install failed."; exit 1; }

log "========== ENV SETUP COMPLETE =========="