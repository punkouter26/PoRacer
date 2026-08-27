# Training run for the four .glb bipeds, split off from the coded-gait fleet.
# Usage: .\scripts\train_humanoids.ps1 [-Hours 0.75] [-ConfigPath Config\Humanoids02_Biomechanical.yaml]
# max_steps in the config is deliberately far out of reach for one session; the
# time box here is the real stop. Continue toward it with -ResumeRunId.
# Starts TensorBoard first (project rule: no training run without it).
# num-envs: this env holds only 8 areas (4 bipeds x 2 track variants), so it is
# much lighter than AllEnv; cores/3 capped at 4 leaves torch its share
# (recorded per MLOps rule).
param(
    [string]$ConfigPath = "Config\Humanoids02_Biomechanical.yaml",
    [string]$EnvExe = "Builds\HumanoidEnv\HumanoidEnv.exe",
    [double]$Hours = 0.75,
    [string]$ResumeRunId = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$RunId = if ($ResumeRunId) { $ResumeRunId } else { "humanoids_" + (Get-Date -Format "yyyyMMdd_HHmm") }
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
$logName = if ($ResumeRunId) { "$RunId-resume-" + (Get-Date -Format "HHmm") + "-console.log" } else { "$RunId-console.log" }
$log = Join-Path $root "results\$logName"

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv. Build via PoRacer/Build Humanoid Training Env." }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF." }
# No demo preflight on purpose: this config carries no BC/GAIL, because a
# biped's coded gait is a fall and cloning it teaches nothing worth having.

# TensorBoard BEFORE the trainer, always (project rule) - a run you cannot watch
# is a run you cannot judge, and these four need watching more than most.
$tbUp = $false
try { $tbUp = (Invoke-WebRequest -Uri "http://localhost:6006" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch {}
if (-not $tbUp) {
    Start-Process -FilePath $tensorboard -ArgumentList "--logdir", (Join-Path $root "results"), "--port", "6006" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}
Write-Host "TensorBoard: http://localhost:6006"

$cores = [Environment]::ProcessorCount
$numEnvs = [Math]::Max(2, [Math]::Min(4, [int][Math]::Floor($cores / 3)))

# base-port 5105 keeps this clear of train_all_8h.ps1's 5005 block, so a fleet
# run and a humanoid run can share the machine without a silent port collision.
$trainArgs = @(
    $config,
    "--run-id=$RunId",
    "--env=$resolvedEnv",
    "--no-graphics",
    "--num-envs=$numEnvs",
    "--base-port=5105",
    "--time-scale=10",
    "--torch-device=cpu"
)
if ($ResumeRunId) { $trainArgs += "--resume" }

Write-Host "Training $RunId for $Hours h: $numEnvs envs ($cores cores), ports 5105-$((5104 + $numEnvs)). Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$deadlineMs = [int]($Hours * 3600 * 1000)
if ($proc.WaitForExit($deadlineMs)) {
    Write-Host "Trainer finished on its own (max_steps). Exit code: $($proc.ExitCode)"
} else {
    Write-Host "Time box reached - stopping trainer (checkpoints preserved; resume with -ResumeRunId $RunId)."
    taskkill /PID $proc.Id | Out-Null
    if (-not $proc.WaitForExit(120000)) { taskkill /PID $proc.Id /F | Out-Null }
}

# Teardown order per project rules: trainer (above) -> envs. TensorBoard stays up.
Get-Process -Name "HumanoidEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Brains: results\$RunId\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
