#!/bin/bash

# ==============================================================================
#  env_setup.sh - Cross-platform environment setup entry point
#
#  On Windows (WSL or Git Bash): delegates to env_setup.ps1 via powershell.exe
#  On Linux / macOS:             runs natively
#
#  Usage:  bash env_setup.sh [-v|--verbose] [-q|--quiet] [--force-cpu] [--env NAME] [--cuda VERSION]
#
#  --cuda VERSION  Skip auto-detection and force a CUDA major version (e.g. --cuda 12)
#                  Use this on HPC login nodes where GPUs are not accessible.
#
#  Windows users: just run  .\env_setup.ps1  directly in PowerShell instead.
#
#  On HPC / slow SSH connections, run inside screen or tmux to avoid disconnects:
#    screen -S setup
#    bash env_setup.sh --cuda 13
#    Ctrl+A then D to detach
# ==============================================================================

set -e

# Prevent SSH keepalive timeouts from killing long pip/conda installs.
# Works by making the shell periodically print something if SSH is idle.
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
    # Send a no-op to stdout every 60s to keep the SSH session alive
    ( while true; do sleep 60; echo -n "." 2>/dev/null || true; done ) &
    KEEPALIVE_PID=$!
    trap 'kill $KEEPALIVE_PID 2>/dev/null || true' EXIT
fi

# -- Default config ------------------------------------------------------------
VERBOSE=0
ENV_NAME="Training"
PYTHON_VERSION="3.11"
FORCE_CPU=0
FORCE_CUDA=""

# -- Argument parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=1 ;;
        -q|--quiet)   VERBOSE=0 ;;
        -vv|--vverbose) VERBOSE=2 ;;
        --force-cpu)  FORCE_CPU=1 ;;
        --cuda)       FORCE_CUDA="$2"; shift ;;
        --env)        ENV_NAME="$2"; shift ;;
        *) echo "[WARN]  Unknown argument: $1" ;;
    esac
    shift
done

# -- Logging helpers -----------------------------------------------------------
log()  { if [ "$VERBOSE" -ge 1 ]; then echo "[INFO]  $1"; fi; }
warn() { echo "[WARN]  $1"; }
err()  { echo "[ERROR] $1"; }

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
{ set +x; } 2>/dev/null
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    [ "$VERBOSE" -eq 2 ] && set -x
    log "Environment '$ENV_NAME' already exists - skipping creation."
else
    log "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y || {
        [ "$VERBOSE" -eq 2 ] && set -x
        err "Failed to create conda environment."; exit 1
    }
    [ "$VERBOSE" -eq 2 ] && set -x
fi

log "Activating environment '$ENV_NAME'..."
# Temporarily disable set -x around conda activate - the activation scripts
# produce thousands of trace lines that can flood and drop SSH connections.
{ set +x; } 2>/dev/null
conda activate "$ENV_NAME" || {
    err "Failed to activate environment. Try: conda init bash"; exit 1
}
# Re-enable tracing after activation is complete
[ "$VERBOSE" -eq 2 ] && set -x

# -- pip -----------------------------------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip

# -- CUDA / GPU checks --------------------------------------------------------

# 1. Check nvidia-smi for driver health
check_nvidia_smi() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found - no NVIDIA GPU or driver detected."
        return 1
    fi

    # Capture both stdout and stderr; a broken driver prints to stderr
    NVSMI_OUT=$(nvidia-smi 2>&1)
    NVSMI_EXIT=$?

    if [ $NVSMI_EXIT -ne 0 ]; then
        err "nvidia-smi failed with exit code $NVSMI_EXIT. GPU driver is broken or not loaded."
        err "Output: $NVSMI_OUT"
        err "Common causes:"
        err "  - Driver was updated without a reboot (reboot the machine)"
        err "  - Driver/library version mismatch (reinstall NVIDIA driver)"
        err "  - No GPU allocated to this job (check your Slurm --gres= flag)"
        return 1
    fi

    # Check for known error strings even on exit 0
    if echo "$NVSMI_OUT" | grep -qiE "Driver/library version mismatch|Failed to initialize NVML|error:"; then
        err "nvidia-smi reports a driver error:"
        err "$NVSMI_OUT"
        err "The GPU driver needs to be fixed by a system administrator."
        return 1
    fi

    log "nvidia-smi OK"
    return 0
}

# 2. Ensure nvcc is available, install via conda if missing
# depricated - this always crashed, better to do manually 
ensure_nvcc() {
    if command -v nvcc &>/dev/null; then
        log "nvcc found: $(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || true)"
        return 0
    fi

    warn "nvcc not found. Attempting to install cudatoolkit via conda..."
    conda install -n "$ENV_NAME" -c conda-forge cudatoolkit -y 2>/dev/null || \
    conda install -n "$ENV_NAME" cudatoolkit -y 2>/dev/null || {
        warn "Could not install cudatoolkit via conda. Will rely on nvidia-smi for CUDA version."
        return 1
    }

    # Reload PATH so nvcc is visible
    export PATH="$CONDA_PREFIX/bin:$PATH"

    if command -v nvcc &>/dev/null; then
        log "nvcc installed: $(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || true)"
        return 0
    else
        warn "nvcc still not found after install - will rely on nvidia-smi."
        return 1
    fi
}

# -- PyTorch -------------------------------------------------------------------
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" && TORCH_EXISTS=0 || TORCH_EXISTS=$?

if [[ "$TORCH_EXISTS" -eq 0 && -n "$FORCE_CUDA"]]; then
    log "PyTorch already installed - skipping."
else
    CUDA_VERSION=""
    if [ "$FORCE_CPU" -eq 1 ]; then
        warn "--force-cpu flag set. Installing CPU-only PyTorch."
    else
        log "Detecting CUDA version..."

        # If --cuda was passed, skip detection entirely
        if [ -n "$FORCE_CUDA" ]; then
            CUDA_VERSION="$FORCE_CUDA.0"
            log "--cuda flag set: forcing CUDA major version $FORCE_CUDA"

        else
            # Check driver health first - exit if broken
            if command -v nvidia-smi &>/dev/null; then
                if ! check_nvidia_smi; then
                    err "Cannot install GPU PyTorch with a broken driver. Exiting."
                    err "Fix the driver or re-run with --force-cpu to install CPU-only PyTorch."
                    err "On HPC login nodes without GPU access, use --cuda <version> instead."
                    err "  e.g.  bash env_setup.sh --cuda 12"
                    exit 1
                fi
                CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
                log "nvidia-smi reports CUDA: ${CUDA_VERSION:-not found}"
            fi

            # Try nvcc as fallback / confirmation, install if missing
            if [ -z "$CUDA_VERSION" ]; then
                #ensure_nvcc
                if command -v nvcc &>/dev/null; then
                    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || true)
                    log "nvcc reports CUDA: ${CUDA_VERSION:-not found}"
                fi
            fi

            if [ -z "$CUDA_VERSION" ]; then
                warn "No CUDA detected and --cuda not specified."
                warn "If you are on an HPC login node, re-run with --cuda <version>."
                warn "  e.g.  bash env_setup.sh --cuda 12"
                warn "Installing CPU-only PyTorch for now."
            fi
        fi
    fi

    CUDA_MAJOR="${FORCE_CUDA:-${CUDA_VERSION%%.*}}"
    case "$CUDA_MAJOR" in
        13) log "Installing PyTorch for CUDA 13.x (cu130)..."
            pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 ;;
        12) log "Installing PyTorch for CUDA 12.x (cu121)..."
            pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ;;
        11) log "Installing PyTorch for CUDA 11.x (cu118)..."
            pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ;;
        *)  log "Installing default PyTorch..."
            pip3 install --upgrade torch torchvision torchaudio ;;
    esac || { err "Failed to install PyTorch."; exit 1; }
fi

if python -c "import ultralytics" &>/dev/null; then
    VERSION=$(python -c "import ultralytics; print(ultralytics.__version__)")
    echo "Ultralytics is installed. Version: $VERSION"
else
    log "Installing Ultralytics 8.4.8..."
    pip install ultralytics==8.4.8 || { err "Failed to install Ultralytics."; exit 1; }
fi
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

# -- Final summary -------------------------------------------------------------
echo ""
echo "========== ENV SETUP COMPLETE =========="
echo ""
echo "  Environment : $ENV_NAME"
python -c "
import torch
print('  Torch version:', torch.__version__)
cuda_ok = torch.cuda.is_available()
print('  CUDA available:', cuda_ok)
if cuda_ok:
    print('  CUDA version :', torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}          : {torch.cuda.get_device_name(i)}')
else:
    print('  GPU            : none (CPU-only mode)')
"
echo ""
echo "  To activate the environment run:"
echo "    conda activate $ENV_NAME"
echo "========================================="