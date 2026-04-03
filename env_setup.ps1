# =============================================================================
#  env_setup.ps1 — Windows / Anaconda PowerShell companion
#  Called automatically by env_setup.sh when running on Windows.
#  Can also be run directly:  .\env_setup.ps1
# =============================================================================

param(
    [string]$EnvName       = "Training",
    [string]$PythonVersion = "3.11",
    [switch]$Debug,
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"

# ── Logging helpers ───────────────────────────────────────────────────────────
function Log   { param($msg) if ($Debug) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan } }
function Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

Log "========== ENV SETUP START (PowerShell) =========="

# ── Locate conda ──────────────────────────────────────────────────────────────
function Find-CondaBase {
    # 1. Already on PATH
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $base = conda info --base 2>$null
        if ($base) { return $base.Trim() }
    }

    # 2. Common install locations
    $candidates = @(
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\anaconda3",
        "C:\miniconda3"
    )
    foreach ($dir in $candidates) {
        if (Test-Path "$dir\Scripts\conda.exe") { return $dir }
    }
    return $null
}

$CondaBase = Find-CondaBase
if (-not $CondaBase) {
    Error "Conda not found. Please install Anaconda or Miniconda."
    exit 1
}

Log "Conda base: $CondaBase"

# Initialise conda for this PowerShell session
$condaHook = "$CondaBase\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
} else {
    # Older conda layout
    $condaInit = "$CondaBase\Scripts\conda.exe"
    (& $condaInit "shell.powershell" "hook") | Out-String | Invoke-Expression
}

# ── Create / reuse conda environment ─────────────────────────────────────────
Log "Checking for conda environment: $EnvName"
$envList = conda env list 2>$null
if ($envList -match "(?m)^${EnvName}\s") {
    Log "Environment '$EnvName' already exists — skipping creation."
} else {
    Log "Creating environment '$EnvName' with Python $PythonVersion..."
    conda create -n $EnvName python=$PythonVersion -y
    if ($LASTEXITCODE -ne 0) { Error "Failed to create conda environment."; exit 1 }
}

Log "Activating environment '$EnvName'..."
conda activate $EnvName
if ($LASTEXITCODE -ne 0) { Error "Failed to activate environment."; exit 1 }

# ── pip ───────────────────────────────────────────────────────────────────────
Log "Upgrading pip..."
pip install --upgrade pip -q

# ── Torch detection & install ─────────────────────────────────────────────────
$torchCheck = python -c "import importlib.util; exit(0 if importlib.util.find_spec('torch') else 1)" 2>$null
$TorchExists = $LASTEXITCODE -eq 0

if ($TorchExists) {
    Log "PyTorch is already installed — skipping."
} else {
    $CudaVersion = ""

    if (-not $ForceCpu) {
        Log "Detecting CUDA version..."

        # Primary: nvidia-smi
        $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $smiOut = & nvidia-smi.exe 2>$null | Out-String
            if ($smiOut -match "CUDA Version:\s+(\d+\.\d+)") {
                $CudaVersion = $Matches[1]
                Log "nvidia-smi reports CUDA: $CudaVersion"
            }
        }

        # Fallback: nvcc
        if (-not $CudaVersion) {
            $nvcc = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
            if ($nvcc) {
                $nvccOut = & nvcc.exe --version 2>$null | Out-String
                if ($nvccOut -match "release (\d+\.\d+)") {
                    $CudaVersion = $Matches[1]
                    Log "nvcc reports CUDA: $CudaVersion"
                }
            }
        }

        if (-not $CudaVersion) { Warn "No CUDA detected — will install CPU-only PyTorch." }
    } else {
        Warn "--ForceCpu flag set. Installing CPU-only PyTorch."
    }

    $CudaMajor = if ($CudaVersion) { [int]($CudaVersion.Split(".")[0]) } else { 0 }

    switch ($CudaMajor) {
        12 {
            Log "Installing PyTorch for CUDA 12.x (cu121)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        11 {
            Log "Installing PyTorch for CUDA 11.x (cu118)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        }
        default {
            Log "Installing CPU-only / default PyTorch..."
            pip install torch torchvision torchaudio
        }
    }

    if ($LASTEXITCODE -ne 0) { Error "Failed to install PyTorch."; exit 1 }
}

# ── Verify torch ──────────────────────────────────────────────────────────────
Log "Verifying PyTorch installation..."
python -c @"
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version  : {torch.version.cuda}')
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
"@

# ── Poetry ────────────────────────────────────────────────────────────────────
Log "Checking Poetry..."
$poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryCmd) {
    Log "Installing Poetry..."
    pip install poetry
    if ($LASTEXITCODE -ne 0) { Error "Failed to install Poetry."; exit 1 }
} else {
    Log "Updating Poetry..."
    pip install --upgrade poetry -q
}

# Tell Poetry to use the active conda env (no nested venv)
Log "Configuring Poetry to use the current environment..."
poetry config virtualenvs.create false --local 2>$null

Log "Running poetry install..."
poetry install
if ($LASTEXITCODE -ne 0) { Error "poetry install failed."; exit 1 }

Log "========== ENV SETUP COMPLETE =========="