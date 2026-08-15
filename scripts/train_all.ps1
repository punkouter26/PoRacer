# Unattended all-creatures training run (9 behaviors, SCN_TRAIN_ALL env).
# Usage: .\scripts\train_all.ps1 [-RunId all_loco01] [-ConfigPath Config\AllLoco01.yaml] [-Resume]
# Runs until max_steps completes for every behavior (no time box).
# Starts TensorBoard first (project rule). num-envs=4: each env instance holds 18
# articulated areas, so 4 instances saturate the CPU while leaving torch headroom.
# Teardown order per project rules: trainer -> envs; TensorBoard stays up for review.
param(
    [string]$RunId = "all_loco01",
    [string]$ConfigPath = "Config\AllLoco01.yaml",
    [string]$EnvExe = "Builds\AllEnv\AllEnv.exe",
    [switch]$Resume,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
$log = Join-Path $root "results\$RunId-console.log"

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv (Unity menu: PoRacer/Build All-Creatures Training Env)" }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF - parallel envs would stall. Enable it in Player Settings." }

# TensorBoard first (project rule). Port 6006; skip if already listening.
$tbUp = $false
try { $tbUp = (Invoke-WebRequest -Uri "http://localhost:6006" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch {}
if (-not $tbUp) {
    Start-Process -FilePath $tensorboard -ArgumentList "--logdir", (Join-Path $root "results"), "--port", "6006" -WindowStyle Hidden
    Write-Host "TensorBoard started at http://localhost:6006"
} else {
    Write-Host "TensorBoard already running at http://localhost:6006"
}

$trainArgs = @(
    $config,
    "--run-id=$RunId",
    "--env=$resolvedEnv",
    "--no-graphics",
    "--num-envs=4",
    "--base-port=5005",
    "--time-scale=10",
    "--torch-device=cpu"
)
if ($Resume) { $trainArgs += "--resume" }
if ($Force) { $trainArgs += "--force" }

Write-Host "Training $RunId until complete: 4 envs (ports 5005-5008), time-scale 10. Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$proc.WaitForExit()
Write-Host "Trainer exited with code $($proc.ExitCode)."

# Teardown order per project rules: trainer (above) -> envs.
Get-Process -Name "AllEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Brains: results\$RunId\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
