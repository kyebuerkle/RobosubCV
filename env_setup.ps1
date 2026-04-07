# ==============================================================================
#  env_setup.ps1 - Windows / Anaconda environment setup
#  Run directly from Anaconda PowerShell or standard PowerShell:
#      .\env_setup.ps1
#
#  Optional flags:
#      -EnvName       Name of the conda environment  (default: Training)
#      -PythonVersion Python version to use           (default: 3.11)
#      -ForceCpu      Skip CUDA detection, install CPU-only PyTorch
#      -Verbose       Print detailed progress logs
# ==============================================================================

param(
    [string]$EnvName       = "Training",
    [string]$PythonVersion = "3.11",
    [switch]$ForceCpu,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
function Log   { param($m) if ($Verbose) { Write-Host "[INFO]  $m" -ForegroundColor Cyan } }
function Warn  { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Err   { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

Log "========== ENV SETUP START =========="

# ------------------------------------------------------------------------------
# Locate conda base directory
# ------------------------------------------------------------------------------
function Find-CondaBase {
    # 1. Already on PATH
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) {
        $base = (conda info --base 2>$null)
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
    Err "Conda not found. Please install Anaconda or Miniconda."
    exit 1
}
Log "Conda base: $CondaBase"

# ------------------------------------------------------------------------------
# Initialise conda for this PowerShell session
# ------------------------------------------------------------------------------
$condaHook = "$CondaBase\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
} else {
    $condaExe = "$CondaBase\Scripts\conda.exe"
    (& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
}

# ------------------------------------------------------------------------------
# Create or reuse conda environment
# ------------------------------------------------------------------------------
Log "Checking for conda environment: $EnvName"
$envList = conda env list 2>$null | Out-String
if ($envList -match "(?m)^${EnvName}\s") {
    Log "Environment '$EnvName' already exists - skipping creation."
} else {
    Log "Creating environment '$EnvName' with Python $PythonVersion..."
    conda create -n $EnvName python=$PythonVersion -y
    if ($LASTEXITCODE -ne 0) { Err "Failed to create conda environment."; exit 1 }
}

Log "Activating environment '$EnvName'..."
conda activate $EnvName
if ($LASTEXITCODE -ne 0) { Err "Failed to activate environment."; exit 1 }

# ------------------------------------------------------------------------------
# Upgrade pip
# ------------------------------------------------------------------------------
Log "Upgrading pip..."
pip install --upgrade pip -q

# ------------------------------------------------------------------------------
# PyTorch - detect existing install
# ------------------------------------------------------------------------------
python -c "import importlib.util; exit(0 if importlib.util.find_spec('torch') else 1)" 2>$null
$TorchExists = ($LASTEXITCODE -eq 0)

if ($TorchExists) {
    Log "PyTorch is already installed - skipping."
} else {
    $CudaVersion = ""

    if (-not $ForceCpu) {
        Log "Detecting CUDA version..."

        # Primary: nvidia-smi
        $nvSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
        if ($nvSmi) {
            $smiOut = (& nvidia-smi.exe 2>$null) | Out-String
            if ($smiOut -match "CUDA Version:\s+(\d+\.\d+)") {
                $CudaVersion = $Matches[1]
                Log "nvidia-smi reports CUDA: $CudaVersion"
            }
        }

        # Fallback: nvcc
        if (-not $CudaVersion) {
            $nvcc = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
            if ($nvcc) {
                $nvccOut = (& nvcc.exe --version 2>$null) | Out-String
                if ($nvccOut -match "release (\d+\.\d+)") {
                    $CudaVersion = $Matches[1]
                    Log "nvcc reports CUDA: $CudaVersion"
                }
            }
        }

        if (-not $CudaVersion) { Warn "No CUDA detected - will install CPU-only PyTorch." }
    } else {
        Warn "-ForceCpu flag set. Installing CPU-only PyTorch."
    }

    $CudaMajor = 0
    if ($CudaVersion) { $CudaMajor = [int]($CudaVersion.Split(".")[0]) }

    if ($CudaMajor -ge 12) {
        Log "Installing PyTorch for CUDA 12.x (cu121)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    } elseif ($CudaMajor -eq 11) {
        Log "Installing PyTorch for CUDA 11.x (cu118)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    } else {
        Log "Installing default PyTorch..."
        pip install torch torchvision torchaudio
    }

    if ($LASTEXITCODE -ne 0) { Err "Failed to install PyTorch."; exit 1 }

}

# ------------------------------------------------------------------------------
# Verify PyTorch
# ------------------------------------------------------------------------------
Log "Verifying PyTorch installation..."
python -c "import torch; print('  torch version : ' + torch.__version__); print('  CUDA available: ' + str(torch.cuda.is_available()))"
if ($LASTEXITCODE -ne 0) { Err "Failed to install PyTorch."; exit 1 }

python -c "import ultralytics; print(ultralytics.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) 
{ 
	Log "Installing Ultralytics 8.4.8..."
	pip install ultralytics==8.4.8
	if ($LASTEXITCODE -ne 0) { Err "Failed to install Ultralytics."; exit 1 }
}
# ------------------------------------------------------------------------------
# Poetry
# ------------------------------------------------------------------------------
Log "Checking Poetry..."
$poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryCmd) {
    Log "Installing Poetry..."
    pip install poetry
    if ($LASTEXITCODE -ne 0) { Err "Failed to install Poetry."; exit 1 }
} else {
    Log "Updating Poetry..."
    pip install --upgrade poetry -q
}

# Use the active conda env directly - no nested virtualenv
Log "Configuring Poetry to use the current environment..."
poetry config virtualenvs.create false --local 2>$null

Log "Running poetry install..."
poetry install
if ($LASTEXITCODE -ne 0) { Err "poetry install failed."; exit 1 }

Log "========== ENV SETUP COMPLETE =========="