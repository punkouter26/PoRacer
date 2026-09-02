# Attended one-hour fine-tune of the creatures that cannot reach the finish
# (Config\BrokenLoco01.yaml, run-id broken_loco01). Warm-starts from the
# all8h_20260826_2244 checkpoints and pins the env to Flat-race conditions.
# Usage: .\scripts\train_broken_1h.ps1 [-Minutes 40] [-RunId broken_loco01]
# TensorBoard goes up first (project rule). Rebuild Builds\AllEnv before this
# if SCN_TRAIN_ALL changed - the env must contain exactly the eight behaviors
# the config lists, or mlagents aborts at startup.
param(
    [string]$ConfigPath = "Config\BrokenLoco01.yaml",
    [string]$EnvExe = "Builds\AllEnv\AllEnv.exe",
    [double]$Minutes = 40,
    [string]$RunId = "broken_loco01"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
$log = Join-Path $root "results\$RunId-console.log"

if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv" }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }

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
    "--torch-device=cpu",
    "--force"
)

Write-Host "Training $RunId for $Minutes min: $numEnvs envs ($cores cores), ports 5005-$((5004 + $numEnvs)). Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$deadlineMs = [int]($Minutes * 60 * 1000)
if ($proc.WaitForExit($deadlineMs)) {
    Write-Host "Trainer finished on its own (max_steps). Exit code: $($proc.ExitCode)"
} else {
    Write-Host "Time box reached - stopping trainer (checkpoints preserved)."
    taskkill /PID $proc.Id | Out-Null
    if (-not $proc.WaitForExit(120000)) { taskkill /PID $proc.Id /F | Out-Null }
}

# Teardown: trainer (above) -> envs. TensorBoard stays up.
Get-Process -Name "AllEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Brains: results\$RunId\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
