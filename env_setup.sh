#!/bin/bash

# ==============================================================================
#  env_setup.sh - Cross-platform environment setup entry point
#
#  On Windows (WSL or Git Bash): delegates to env_setup.ps1 via powershell.exe
#  On Linux / macOS:             runs natively
#
#  Usage:  bash env_setup.sh [OPTIONS]
#
#  Options:
#    -v, --verbose      Show info logs
#    -vv, --vverbose    Show all shell tracing (very noisy, avoid over SSH)
#    -q, --quiet        Suppress info logs (default)
#    --force-cpu        Install CPU-only PyTorch regardless of GPU
#    --cuda VERSION     Force a CUDA major version, skips auto-detection and
#                       reinstalls torch even if already present.
#                       Use from HPC login nodes or to fix a wrong torch build.
#                       e.g.  bash env_setup.sh --cuda 13
#    --env NAME         Conda environment name (default: Training)
#
#  On HPC: prefer running via sbatch so GPUs are allocated and auto-detected:
#    sbatch env_setup.sbatch
#
#  If running from an HPC login node directly, use --cuda to specify version:
#    bash env_setup.sh --cuda 13
#
#  On slow SSH connections, wrap in screen or tmux to survive disconnects:
#    screen -S setup && bash env_setup.sh
#    Ctrl+A then D to detach, reconnect with: screen -r setup
#
#  Windows users: run  .\env_setup.ps1  directly in PowerShell instead.
# ==============================================================================

set -e

# -- Default config ------------------------------------------------------------
VERBOSE=0
ENV_NAME="Training"
PYTHON_VERSION="3.11"
FORCE_CPU=0
FORCE_CUDA=""

# -- Argument parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)   VERBOSE=1 ;;
        -vv|--vverbose) VERBOSE=2 ;;
        -q|--quiet)     VERBOSE=0 ;;
        --force-cpu)    FORCE_CPU=1 ;;
        --cuda)         FORCE_CUDA="$2"; shift ;;
        --env)          ENV_NAME="$2"; shift ;;
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
# Three "bash on Windows" cases all need to delegate to PowerShell:
#   Git Bash / MSYS2  -> $WINDIR is set, uname says "Linux" or MINGW*
#   WSL               -> uname says "Linux", /proc/version has "microsoft"
#   Cygwin            -> uname says CYGWIN*
detect_os() {
    if [ -n "$WINDIR" ] || [ -n "$SYSTEMROOT" ]; then
        echo "windows"; return
    fi
    if [ -f /proc/version ] && grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
        echo "wsl"; return
    fi
    case "$(uname -s)" in
        Darwin*)               echo "mac" ;;
        CYGWIN*|MINGW*|MSYS*)  echo "windows" ;;
        Linux*)                echo "linux" ;;
        *)                     echo "unknown" ;;
    esac
}

OS=$(detect_os)
log "Detected OS: $OS"

# -- Windows / WSL: delegate to PowerShell ------------------------------------
if [ "$OS" = "windows" ] || [ "$OS" = "wsl" ]; then
    if [ "$OS" = "wsl" ]; then
        warn "Running inside WSL. Delegating to Windows PowerShell so conda"
        warn "can find your Windows Anaconda installation."
        warn "Tip: run  .\\env_setup.ps1  directly in PowerShell instead."
    else
        log "Windows (Git Bash) detected. Delegating to env_setup.ps1..."
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PS_SCRIPT="$SCRIPT_DIR/env_setup.ps1"

    if [ ! -f "$PS_SCRIPT" ]; then
        err "env_setup.ps1 not found in $SCRIPT_DIR"
        err "Both env_setup.sh and env_setup.ps1 must be in the same directory."
        exit 1
    fi

    if command -v cygpath &>/dev/null; then
        WIN_PATH=$(cygpath -w "$PS_SCRIPT")
    elif [ "$OS" = "wsl" ]; then
        if command -v wslpath &>/dev/null; then
            WIN_PATH=$(wslpath -w "$PS_SCRIPT")
        else
            WIN_PATH=$(echo "$PS_SCRIPT" | sed -E 's|^/mnt/([a-zA-Z])/|\1:/|; s|/|\\|g')
        fi
    else
        WIN_PATH=$(echo "$PS_SCRIPT" | sed -E 's|^/([a-zA-Z])/|\1:/|; s|/|\\|g')
    fi

    log "PowerShell script path: $WIN_PATH"

    PS_EXTRA_ARGS=()
    [ "$VERBOSE"   -ge 1 ] && PS_EXTRA_ARGS+=("-Verbose")
    [ "$FORCE_CPU" -eq 1 ] && PS_EXTRA_ARGS+=("-ForceCpu")
    [ -n "$FORCE_CUDA" ]   && PS_EXTRA_ARGS+=("-ForceCuda" "$FORCE_CUDA")

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
# Suppress tracing around conda calls - they produce thousands of lines
# of shell trace output that can flood and drop SSH connections.
{ set +x; } 2>/dev/null
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    [ "$VERBOSE" -ge 2 ] && set -x
    log "Environment '$ENV_NAME' already exists - skipping creation."
else
    log "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y || {
        [ "$VERBOSE" -ge 2 ] && set -x
        err "Failed to create conda environment."; exit 1
    }
    [ "$VERBOSE" -ge 2 ] && set -x
fi

log "Activating environment '$ENV_NAME'..."
{ set +x; } 2>/dev/null
conda activate "$ENV_NAME" || {
    err "Failed to activate environment. Try: conda init bash"; exit 1
}
[ "$VERBOSE" -ge 2 ] && set -x

# -- pip -----------------------------------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip

# -- CUDA / GPU health check ---------------------------------------------------
check_nvidia_smi() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found - no NVIDIA GPU or driver detected."
        return 1
    fi

    NVSMI_OUT=$(nvidia-smi 2>&1)
    NVSMI_EXIT=$?

    if [ $NVSMI_EXIT -ne 0 ]; then
        err "nvidia-smi failed (exit $NVSMI_EXIT). GPU driver is broken or not loaded."
        err "Output: $NVSMI_OUT"
        err "Common causes:"
        err "  - Driver updated without a reboot -> reboot the machine"
        err "  - Driver/library version mismatch -> reinstall NVIDIA driver"
        err "  - No GPU allocated to this job   -> check your Slurm --gres= flag"
        return 1
    fi

    if echo "$NVSMI_OUT" | grep -qiE "Driver/library version mismatch|Failed to initialize NVML|error:"; then
        err "nvidia-smi reports a driver error:"
        err "$NVSMI_OUT"
        err "The GPU driver needs to be fixed by a system administrator."
        return 1
    fi

    log "nvidia-smi OK"
    return 0
}

# -- PyTorch -------------------------------------------------------------------
# Decide whether to install / reinstall torch:
#   - Not installed           -> always install
#   - Installed + --cuda set  -> reinstall with --force-reinstall to fix wrong build
#   - Installed, no --cuda    -> skip
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" \
    && TORCH_EXISTS=0 || TORCH_EXISTS=$?

if [ "$TORCH_EXISTS" -eq 0 ] && [ -z "$FORCE_CUDA" ] && [ "$FORCE_CPU" -eq 0 ]; then
    log "PyTorch already installed and no --cuda or --force-cpu flag set - skipping."
else
    REINSTALL_FLAG=""
    [ "$TORCH_EXISTS" -eq 0 ] && [ -n "$FORCE_CUDA" ] && REINSTALL_FLAG="--force-reinstall"
    [ -n "$REINSTALL_FLAG" ] && log "Torch exists but --cuda set - forcing reinstall."

    CUDA_VERSION=""
    if [ "$FORCE_CPU" -eq 1 ]; then
        warn "--force-cpu flag set. Installing CPU-only PyTorch."

    elif [ -n "$FORCE_CUDA" ]; then
        # --cuda bypasses all detection
        log "--cuda $FORCE_CUDA set: skipping auto-detection."

    else
        log "Detecting CUDA version..."

        if command -v nvidia-smi &>/dev/null; then
            if ! check_nvidia_smi; then
                err "Cannot install GPU PyTorch with a broken driver."
                err "Options:"
                err "  Fix the driver, then re-run this script."
                err "  Use --force-cpu to install CPU-only PyTorch."
                err "  Use --cuda <ver> if on an HPC login node (e.g. --cuda 13)."
                exit 1
            fi
            CUDA_VERSION=$(nvidia-smi 2>/dev/null \
                | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
            log "nvidia-smi reports CUDA: ${CUDA_VERSION:-not found}"
        fi

        # nvcc fallback (no auto-install - mark as deprecated in comments)
        if [ -z "$CUDA_VERSION" ] && command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version 2>/dev/null \
                | grep -oP "release \K[0-9]+\.[0-9]+" || true)
            log "nvcc reports CUDA: ${CUDA_VERSION:-not found}"
        fi

        if [ -z "$CUDA_VERSION" ]; then
            warn "No CUDA detected and --cuda not specified."
            warn "If on an HPC login node, use:  bash env_setup.sh --cuda <ver>"
            warn "Or submit via sbatch so GPUs are allocated for auto-detection."
            warn "Installing CPU-only PyTorch for now."
        fi
    fi

    # Resolve CUDA major version for index URL selection
    CUDA_MAJOR="${FORCE_CUDA:-${CUDA_VERSION%%.*}}"

    case "$CUDA_MAJOR" in
        13) log "Installing PyTorch for CUDA 13.x (cu130)..."
            pip install $REINSTALL_FLAG torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu130 ;;
        12) log "Installing PyTorch for CUDA 12.x (cu121)..."
            pip install $REINSTALL_FLAG torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121 ;;
        11) log "Installing PyTorch for CUDA 11.x (cu118)..."
            pip install $REINSTALL_FLAG torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu118 ;;
        *)  log "Installing CPU-only PyTorch..."
            pip install $REINSTALL_FLAG torch torchvision torchaudio ;;
    esac || { err "Failed to install PyTorch."; exit 1; }
fi

# -- Ultralytics ---------------------------------------------------------------
if python -c "import ultralytics" &>/dev/null; then
    VERSION=$(python -c "import ultralytics; print(ultralytics.__version__)")
    log "Ultralytics already installed (version $VERSION) - skipping."
else
    log "Installing Ultralytics..."
    pip install ultralytics || { err "Failed to install Ultralytics."; exit 1; }
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
if [ -f "poetry.lock" ]; then
    rm "poetry.lock"
fi
poetry install || { err "poetry install failed."; exit 1; }

# -- Final summary -------------------------------------------------------------
echo ""
echo "========== ENV SETUP COMPLETE =========="
echo ""
echo "  Environment : $ENV_NAME"
python -c "
import torch
print('  Torch version :', torch.__version__)
cuda_ok = torch.cuda.is_available()
print('  CUDA available:', cuda_ok)
if cuda_ok:
    print('  CUDA version  :', torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}          : {torch.cuda.get_device_name(i)}')
else:
    print('  GPU            : none (CPU-only mode)')
"
echo ""
echo "  Run this to activate the environment:"
echo "    conda activate $ENV_NAME"
echo "========================================="