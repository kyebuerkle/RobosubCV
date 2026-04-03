#!/bin/bash

# =============================================================================
#  env_setup.sh — Cross-platform environment setup entry point
#  Works from:  Linux bash  |  Windows Anaconda PowerShell (via Git Bash/MSYS)
#
#  Usage:  bash env_setup.sh [--debug] [--force-cpu] [--env NAME]
# =============================================================================

set -e

# ── Default config ────────────────────────────────────────────────────────────
DEBUG=1
ENV_NAME="Training"
PYTHON_VERSION="3.11"
FORCE_CPU=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      DEBUG=1 ;;
        --quiet)      DEBUG=0 ;;
        --force-cpu)  FORCE_CPU=1 ;;
        --env)        ENV_NAME="$2"; shift ;;
        *) echo "[WARN] Unknown argument: $1" ;;
    esac
    shift
done

# ── Logging helpers ───────────────────────────────────────────────────────────
log()   { [ "$DEBUG" -eq 1 ] && echo "[INFO]  $1"; }
warn()  { echo "[WARN]  $1"; }
error() { echo "[ERROR] $1"; }

log "========== ENV SETUP START =========="

# ── OS detection ──────────────────────────────────────────────────────────────
# IMPORTANT: Check Windows env vars FIRST.
# Git Bash / MSYS2 running inside PowerShell reports `uname -s` as "Linux",
# so uname alone cannot distinguish "real Linux" from "Windows + Git Bash".
# $WINDIR and $SYSTEMROOT are always set on Windows regardless of the shell.
detect_os() {
    # Primary signal: Windows environment variables
    if [ -n "$WINDIR" ] || [ -n "$SYSTEMROOT" ]; then
        echo "windows"
        return
    fi
    # Secondary: uname (reliable on true Linux/macOS)
    case "$(uname -s)" in
        Darwin*)              echo "mac" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        Linux*)               echo "linux" ;;
        *)                    echo "unknown" ;;
    esac
}

OS=$(detect_os)
log "Detected OS: $OS"

# ── Windows path: delegate to PowerShell script ───────────────────────────────
if [ "$OS" = "windows" ]; then
    log "Windows detected — delegating to env_setup.ps1 via PowerShell..."

    # Build the PowerShell script path (same directory as this script)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PS_SCRIPT="$SCRIPT_DIR/env_setup.ps1"

    if [ ! -f "$PS_SCRIPT" ]; then
        error "env_setup.ps1 not found in $SCRIPT_DIR"
        error "Both env_setup.sh and env_setup.ps1 must be in the same directory."
        exit 1
    fi

    # Convert Unix path to Windows path for PowerShell.
    # Try cygpath first (Cygwin/MSYS2), then fall back to a sed conversion
    # that handles both /c/Users/... (Git Bash) and /mnt/c/... (WSL) style paths.
    if command -v cygpath &>/dev/null; then
        WIN_PATH=$(cygpath -w "$PS_SCRIPT")
    else
        WIN_PATH=$(echo "$PS_SCRIPT" \
            | sed -E 's|^/([a-zA-Z])/|\1:/|; s|^/mnt/([a-zA-Z])/|\1:/|; s|/|\\|g')
    fi
    log "PowerShell script path: $WIN_PATH"

    PS_EXTRA_ARGS=()
    [ "$DEBUG"     -eq 1 ] && PS_EXTRA_ARGS+=("-Debug")
    [ "$FORCE_CPU" -eq 1 ] && PS_EXTRA_ARGS+=("-ForceCpu")

    powershell.exe -ExecutionPolicy Bypass -File "$WIN_PATH" \
        -EnvName "$ENV_NAME" \
        -PythonVersion "$PYTHON_VERSION" \
        "${PS_EXTRA_ARGS[@]}"

    exit $?
fi

# =============================================================================
#  Linux / macOS path — everything below runs natively in bash
# =============================================================================

# ── Locate & initialise conda ─────────────────────────────────────────────────
init_conda() {
    if command -v conda &>/dev/null; then
        # Already on PATH — just initialise the shell functions
        CONDA_BASE=$(conda info --base 2>/dev/null)
    else
        # Search common install locations
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
                CONDA_BASE="$dir"
                break
            fi
        done
        if [ -z "$CONDA_BASE" ]; then
            error "Conda not found. Please install Anaconda or Miniconda."
            exit 1
        fi
    fi

    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    log "Conda initialised from: $CONDA_BASE"
}

init_conda

# ── Create / reuse conda environment ─────────────────────────────────────────
log "Checking for conda environment: $ENV_NAME"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    log "Environment '$ENV_NAME' already exists — skipping creation."
else
    log "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y || {
        error "Failed to create conda environment."
        exit 1
    }
fi

log "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME" || {
    error "Failed to activate environment. Try running: conda init bash"
    exit 1
}

# ── pip ───────────────────────────────────────────────────────────────────────
log "Upgrading pip..."
pip install --upgrade pip -q

# ── Torch detection & install ─────────────────────────────────────────────────
python - <<'PYEOF'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PYEOF
TORCH_EXISTS=$?

if [ "$TORCH_EXISTS" -eq 0 ]; then
    log "PyTorch is already installed — skipping."
else
    if [ "$FORCE_CPU" -eq 1 ]; then
        warn "--force-cpu flag set. Installing CPU-only PyTorch."
        CUDA_VERSION=""
    else
        log "Detecting CUDA version..."
        CUDA_VERSION=""

        # Primary: nvidia-smi
        if command -v nvidia-smi &>/dev/null; then
            CUDA_VERSION=$(nvidia-smi 2>/dev/null \
                | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
            log "nvidia-smi reports CUDA: ${CUDA_VERSION:-not found}"
        fi

        # Fallback: nvcc
        if [ -z "$CUDA_VERSION" ] && command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version 2>/dev/null \
                | grep -oP "release \K[0-9]+\.[0-9]+" || true)
            log "nvcc reports CUDA: ${CUDA_VERSION:-not found}"
        fi

        [ -z "$CUDA_VERSION" ] && warn "No CUDA detected — will install CPU-only PyTorch."
    fi

    install_torch() {
        local index_url="$1"
        local label="$2"
        log "Installing PyTorch ($label)..."
        if [ -n "$index_url" ]; then
            pip install torch torchvision torchaudio \
                --index-url "$index_url" || {
                error "Failed to install PyTorch ($label)."
                exit 1
            }
        else
            pip install torch torchvision torchaudio || {
                error "Failed to install PyTorch (CPU/default)."
                exit 1
            }
        fi
    }

    CUDA_MAJOR="${CUDA_VERSION%%.*}"   # e.g. "12" from "12.1"
    case "$CUDA_MAJOR" in
        12) install_torch "https://download.pytorch.org/whl/cu121" "CUDA 12.x → cu121" ;;
        11) install_torch "https://download.pytorch.org/whl/cu118" "CUDA 11.x → cu118" ;;
        *)  install_torch "" "CPU / default" ;;
    esac
fi

# ── Verify torch ──────────────────────────────────────────────────────────────
log "Verifying PyTorch installation..."
python - <<'PYEOF'
import torch
print(f"  torch version : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version  : {torch.version.cuda}")
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
PYEOF

# ── Poetry ────────────────────────────────────────────────────────────────────
log "Checking Poetry..."
if ! command -v poetry &>/dev/null; then
    log "Installing Poetry..."
    pip install poetry || { error "Failed to install Poetry."; exit 1; }
else
    log "Updating Poetry..."
    pip install --upgrade poetry -q
fi

# Tell Poetry to use the active conda env's Python (avoids venv-inside-venv)
log "Configuring Poetry to use the current environment..."
poetry config virtualenvs.create false --local 2>/dev/null || true

log "Running poetry install..."
poetry install || { error "poetry install failed."; exit 1; }

log "========== ENV SETUP COMPLETE =========="