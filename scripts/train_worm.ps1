# Trains a creature locomotion policy against a headless env build.
# Usage: .\scripts\train_worm.ps1 [-RunId worm_loco01] [-Config Config\WormLoco01.yaml] [-EnvExe Builds\WormEnv\WormEnv.exe] [-Resume]
# Step budget lives in the YAML (max_steps) - mlagents-learn has no CLI override.
# Teardown order (per project rules): trainer -> envs -> TensorBoard.
param(
    [string]$RunId = "worm_loco01",
    [string]$ConfigPath = "Config\WormLoco01.yaml",
    [string]$EnvExe = "Builds\WormEnv\WormEnv.exe",
    [switch]$Resume,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv (Unity menu: PoRacer/Build ... Training Env)" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF - parallel envs would stall. Enable it in Player Settings." }

$trainArgs = @(
    $config,
    "--run-id=$RunId",
    "--env=$resolvedEnv",
    "--no-graphics",
    "--num-envs=8",
    "--base-port=5005",
    "--time-scale=10",
    "--torch-device=cpu"
)
if ($Resume) { $trainArgs += "--resume" }
if ($Force) { $trainArgs += "--force" }

Write-Host "Training $RunId with 8 envs (ports 5005-5012), time-scale 10..."
& $mlagents @trainArgs
