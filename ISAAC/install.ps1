<#
.SYNOPSIS
  Sets up Isaac Lab inside PoRacer\ISAAC\ and installs the Boy task package into it.

.DESCRIPTION
  1. Clones Isaac Lab into ISAAC\isaaclab (skipped if present).
  2. Creates ISAAC\isaaclab\.venv (Python 3.11) and pip-installs Isaac Sim into it, the way
     the Isaac Lab "pip installation" docs describe. Isaac Sim is ~10 GB of wheels.
  3. Runs isaaclab.bat --install rsl_rl, then pip install -e ISAAC\boy_tasks plus onnx,
     onnxruntime and tensorboard.

  Nothing here touches PoRacer\.venv - that one carries the ML-Agents/torch pins and must
  stay in exact parity with the C# package.

  HARDWARE NOTE: Isaac Sim's stated minimum is an RTX 3070 with 8 GB of VRAM. This machine
  reports an RTX 2060 with 6 GB. Headless training with a few hundred envs may still run,
  but it is unsupported; expect to use --num_envs 512 or lower, or train on another box
  and copy logs\rsl_rl\boy_chase_flat back here for export.

.PARAMETER IsaacLabRef
  Git ref of Isaac Lab to check out. Default: main.

.PARAMETER Python
  Python 3.11 launcher to build the venv from. Default: "py -3.11".
#>
param(
    [string]$IsaacLabRef = "main",
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Lab = Join-Path $Root "isaaclab"

Write-Host "== Boy / Isaac Lab setup in $Root"

# ---- 1. Isaac Lab checkout -------------------------------------------------------------
if (-not (Test-Path (Join-Path $Lab "isaaclab.bat"))) {
    Write-Host "-- cloning Isaac Lab ($IsaacLabRef) into $Lab"
    git clone --depth 1 --branch $IsaacLabRef https://github.com/isaac-sim/IsaacLab.git $Lab
} else {
    Write-Host "-- Isaac Lab already present at $Lab"
}

# ---- 2. venv + Isaac Sim --------------------------------------------------------------
$Venv = Join-Path $Lab ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "-- creating $Venv"
    & $Python.Split(" ")[0] $Python.Split(" ")[1..99] -m venv $Venv
}
& $Py -m pip install --upgrade pip
$hasSim = & $Py -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('isaacsim') else 1)"; $simOk = $LASTEXITCODE -eq 0
if (-not $simOk) {
    Write-Host "-- installing Isaac Sim wheels (large download)"
    & $Py -m pip install "isaacsim[all,extscache]" --extra-index-url https://pypi.nvidia.com
}

# ---- 3. Isaac Lab + rsl_rl + the Boy task ----------------------------------------------
Push-Location $Lab
try {
    $env:ISAACLAB_PYTHON = $Py
    Write-Host "-- isaaclab.bat --install rsl_rl"
    & (Join-Path $Lab "isaaclab.bat") --install rsl_rl
} finally {
    Pop-Location
}

Write-Host "-- installing boy_tasks (editable) + export deps"
& $Py -m pip install -e (Join-Path $Root "boy_tasks")
& $Py -m pip install onnx onnxruntime tensorboard usd-core

# ---- 4. rig outputs ---------------------------------------------------------------------
Write-Host "-- regenerating the rig (USD + json) with the venv's Python"
& $Py (Join-Path $Root "boy_rig\build_boy_rig.py")

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  $Lab\isaaclab.bat -p $Root\scripts\train.py --num_envs 2048"
Write-Host "  $Lab\isaaclab.bat -p $Root\scripts\export_bundle.py"
Write-Host "TensorBoard: http://localhost:6006 (train.py starts it)."
