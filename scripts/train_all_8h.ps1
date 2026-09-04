# Unattended 8-hour all-creatures training run with a timestamped run-id.
# Usage: .\scripts\train_all_8h.ps1 [-ConfigPath Config\AllLoco8h02.yaml] [-Hours 8]
# max_steps in the config (3.2M) is calibrated to finish near the 8 h mark; the
# time box here is the hard stop. Starts TensorBoard first (project rule).
# num-envs: this env holds 12 articulated areas per instance (6 creatures x 2
# track variants) since Worm and Spider were removed on 2026-09-04; instances are
# CPU-heavy, so cores/3 capped at 4 leaves torch its share (recorded per MLOps
# rule). The four .glb bipeds train separately via scripts/train_humanoids.ps1.
#
# The config must declare exactly the behaviours AllEnv contains: mlagents-learn
# aborts on the first one it reports that the config does not name.
param(
    [string]$ConfigPath = "Config\AllLoco8h02.yaml",
    [string]$EnvExe = "Builds\AllEnv\AllEnv.exe",
    [double]$Hours = 8,
    # Continue an interrupted run toward max_steps: pass its existing run-id.
    [string]$ResumeRunId = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$RunId = if ($ResumeRunId) { $ResumeRunId } else { "all8h_" + (Get-Date -Format "yyyyMMdd_HHmm") }
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
# Resume blocks get their own log so the first block's console is preserved.
$logName = if ($ResumeRunId) { "$RunId-resume-" + (Get-Date -Format "HHmm") + "-console.log" } else { "$RunId-console.log" }
$log = Join-Path $root "results\$logName"

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv" }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF." }
# The config warm-starts the coded-gait creatures from BC/GAIL demos, by exact
# filename - a stale recording session leaves "Crab_1.demo" behind while
# "Crab.demo" is missing, which mlagents only complains about minutes in. The
# four humanoids are deliberately absent: a biped's coded gait is a fall, so they
# train on extrinsic reward alone.
$demoBehaviors = @("Hexapod", "Quad", "Centipede", "Crab", "Kangaroo", "Blob")
# training\demos, NOT Assets\Demonstrations: that is where the demos actually live
# and, more to the point, it is the path the config's demo_path entries resolve to.
# Checking the other folder made this preflight throw on a tree where every demo was
# present and correct - a guard that blocked the run it was meant to protect.
$missingDemos = $demoBehaviors | Where-Object { -not (Test-Path (Join-Path $root "training\demos\$_.demo")) }
if ($missingDemos) { throw "Missing demos: $($missingDemos -join ', ') under training\demos. Re-record with Editor_RecordDemos.EnableRecorders() (delete the old .demo files first, or the new ones land as <Name>_1.demo)." }

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
if ($ResumeRunId) { $trainArgs += "--resume" }

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
