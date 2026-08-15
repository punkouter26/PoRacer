# Unattended 1-hour all-creatures training run with a timestamped run-id.
# Usage: .\scripts\train_all_1h.ps1 [-ConfigPath Config\AllLoco1h01.yaml] [-Hours 1]
# max_steps in the config is calibrated to finish near the 1 h mark; the time box
# here is the hard stop. Starts TensorBoard first (project rule).
# num-envs: this env holds 18 articulated areas per instance, so instances are
# CPU-heavy; cores/6 capped at 4 leaves torch its share (recorded per MLOps rule).
param(
    [string]$ConfigPath = "Config\AllLoco1h01.yaml",
    [string]$EnvExe = "Builds\AllEnv\AllEnv.exe",
    [double]$Hours = 1
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$RunId = "all1h_" + (Get-Date -Format "yyyyMMdd_HHmm")
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
$log = Join-Path $root "results\$RunId-console.log"

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv" }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF." }

# TensorBoard first (project rule).
$tbUp = $false
try { $tbUp = (Invoke-WebRequest -Uri "http://localhost:6006" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch {}
if (-not $tbUp) {
    Start-Process -FilePath $tensorboard -ArgumentList "--logdir", (Join-Path $root "results"), "--port", "6006" -WindowStyle Hidden
}
Write-Host "TensorBoard: http://localhost:6006"

$cores = [Environment]::ProcessorCount
$numEnvs = [Math]::Max(2, [Math]::Min(4, [int][Math]::Floor($cores / 3)))

$trainArgs = @(
    $config,
    "--run-id=$RunId",
    "--env=$resolvedEnv",
    "--no-graphics",
    "--num-envs=$numEnvs",
    "--base-port=5005",
    "--time-scale=10",
    "--torch-device=cpu"
)

Write-Host "Training $RunId for $Hours h: $numEnvs envs ($cores cores), ports 5005-$((5004 + $numEnvs)). Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$deadlineMs = [int]($Hours * 3600 * 1000)
if ($proc.WaitForExit($deadlineMs)) {
    Write-Host "Trainer finished on its own (max_steps). Exit code: $($proc.ExitCode)"
} else {
    Write-Host "Time box reached - stopping trainer (checkpoints preserved; resume via mlagents-learn --resume)."
    taskkill /PID $proc.Id | Out-Null
    if (-not $proc.WaitForExit(120000)) { taskkill /PID $proc.Id /F | Out-Null }
}

# Teardown order per project rules: trainer (above) -> envs. TensorBoard stays up.
Get-Process -Name "AllEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Brains: results\$RunId\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
