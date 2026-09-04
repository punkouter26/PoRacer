# Unattended training run on the Acrobat course with a timestamped run-id.
# Usage: .\scripts\train_acrobat.ps1 [-ConfigPath Config\AcrobatLoco01.yaml] [-Hours 8]
# Starts TensorBoard first (project rule) and refuses to run while port 6006 is
# held by something that is not answering, so the run is never blind.
# num-envs: AcrobatEnv holds 4 course areas per instance, each a 184k-triangle
# mountain; cores/3 capped at 4 leaves torch its share (recorded per MLOps rule).
#
# The config must declare exactly the behaviours AcrobatEnv contains:
# mlagents-learn aborts on the first one it reports that the config does not name.
param(
    [string]$ConfigPath = "Config\AcrobatLoco01.yaml",
    [string]$EnvExe = "Builds\AcrobatEnv\AcrobatEnv.exe",
    [double]$Hours = 8,
    # Continue an interrupted run toward max_steps: pass its existing run-id.
    [string]$ResumeRunId = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$RunId = if ($ResumeRunId) { $ResumeRunId } else { "acrobat_" + (Get-Date -Format "yyyyMMdd_HHmm") }
$resolvedEnv = Join-Path $root $EnvExe
$config = Join-Path $root $ConfigPath
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$tensorboard = Join-Path $root ".venv\Scripts\tensorboard.exe"
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force $results | Out-Null
$logName = if ($ResumeRunId) { "$RunId-resume-" + (Get-Date -Format "HHmm") + "-console.log" } else { "$RunId-console.log" }
$log = Join-Path $results $logName

# Preflight
if (-not (Test-Path $resolvedEnv)) { throw "Env build missing: $resolvedEnv (Editor_BuildCourseTrainingScene.BuildEnv)" }
if (-not (Test-Path $config)) { throw "Config missing: $config" }
if (-not (Test-Path $mlagents)) { throw "mlagents-learn missing: $mlagents" }
$settings = Get-Content (Join-Path $root "ProjectSettings\ProjectSettings.asset") -Raw
if ($settings -match "runInBackground: 0") { throw "runInBackground is OFF." }
$demoBehaviors = @("Centipede", "Crab", "Hexapod", "Quad")
$missingDemos = $demoBehaviors | Where-Object { -not (Test-Path (Join-Path $root "training\demos\$_.demo")) }
if ($missingDemos) { throw "Missing demos: $($missingDemos -join ', ') under training\demos." }

# TensorBoard first (project rule). A leaked listener from a crashed run answers
# nothing on 6006; that is a blind run, so stop rather than warn.
$tbUp = $false
try { $tbUp = (Invoke-WebRequest -Uri "http://localhost:6006" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch {}
if (-not $tbUp) {
    $held = Get-NetTCPConnection -LocalPort 6006 -State Listen -ErrorAction SilentlyContinue
    if ($held) { throw "Port 6006 is held by PID $($held.OwningProcess) but not serving TensorBoard - kill it first." }
    Start-Process -FilePath $tensorboard -ArgumentList "--logdir", $results, "--port", "6006" -WindowStyle Hidden
    Start-Sleep -Seconds 4
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
    "--base-port=5105",
    "--time-scale=10",
    "--torch-device=cpu"
)
if ($ResumeRunId) { $trainArgs += "--resume" }
Set-Content -Path (Join-Path $results "ACTIVE_RUN_ID.txt") -Value $RunId
Add-Content -Path (Join-Path $results "RUN_STATUS.txt") -Value "START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  run-id=$RunId  time-box=${Hours}h  envs=$numEnvs"
Write-Host "Training $RunId for $Hours h: $numEnvs envs ($cores cores), ports 5105-$((5104 + $numEnvs)). Console -> $log"
$proc = Start-Process -FilePath $mlagents -ArgumentList $trainArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$deadlineMs = [int]($Hours * 3600 * 1000)
if ($proc.WaitForExit($deadlineMs)) {
    Write-Host "Trainer finished on its own (max_steps). Exit code: $($proc.ExitCode)"
    Add-Content -Path (Join-Path $results "RUN_STATUS.txt") -Value "END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=$($proc.ExitCode)"
} else {
    Write-Host "Time box reached - stopping trainer (checkpoints preserved; resume via -ResumeRunId $RunId)."
    taskkill /PID $proc.Id | Out-Null
    if (-not $proc.WaitForExit(120000)) { taskkill /PID $proc.Id /F | Out-Null }
    Add-Content -Path (Join-Path $results "RUN_STATUS.txt") -Value "END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=124 (time box)"
}
# Teardown order per project rules: trainer (above) -> envs. TensorBoard stays up.
Get-Process -Name "AcrobatEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done. Brains: results\$RunId\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
