# Unattended 6-hour worm training run against the headless env build.
# Usage: .\scripts\train_worm_6h.ps1 [-RunId worm_loco03] [-ConfigPath Config\WormLoco03.yaml] [-Hours 6] [-Resume]
# Starts TensorBoard first (project rule: no training run without it), then
# mlagents-learn. After the time box: teardown order trainer -> envs; TensorBoard
# is left running so results can be reviewed when you return.
param(
    [string]$RunId = "worm_loco03",
    [string]$ConfigPath = "Config\WormLoco03.yaml",
    [string]$EnvExe = "Builds\WormEnv\WormEnv.exe",
    [double]$Hours = 6,
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
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv (Unity menu: PoRacer/Build Worm Training Env)" }
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
    "--num-envs=8",
    "--base-port=5005",
    "--time-scale=10",
    "--torch-device=cpu"
)
if ($Resume) { $trainArgs += "--resume" }
if ($Force) { $trainArgs += "--force" }

Write-Host "Training $RunId for $Hours h: 8 envs (ports 5005-5012), time-scale 10. Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$deadlineMs = [int]($Hours * 3600 * 1000)
if ($proc.WaitForExit($deadlineMs)) {
    Write-Host "Trainer exited on its own (max_steps reached or error). Exit code: $($proc.ExitCode)"
} else {
    Write-Host "Time box of $Hours h reached - stopping trainer (checkpoints save every 500k steps; resume with -Resume)."
    # Graceful close first so the final checkpoint can flush, hard kill as fallback.
    taskkill /PID $proc.Id | Out-Null
    if (-not $proc.WaitForExit(120000)) { taskkill /PID $proc.Id /F | Out-Null }
}

# Teardown order per project rules: trainer (above) -> envs. TensorBoard stays up for review.
Get-Process -Name "WormEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Latest brain: results\$RunId\Worm\*.onnx  |  TensorBoard: http://localhost:6006"
