# Overnight evolution: sequentially trains the next brain version for each creature
# on the terrain-enabled envs, then stages the results for auto-bake.
# Teardown order per project rules: trainer -> envs -> (TensorBoard is never started here).
# Run manually or via the "PoRacer Evolve" scheduled task (02:00 daily).
#
# Staging: results are copied to Assets/Agents/<Name>_v<NN>/<Name>_v<NN>.onnx with a
# .autobake marker; Editor_AutoBake picks them up on the next editor load and adds
# catalog entries so old and new versions race each other (that is what ELO is for).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$mlagents = Join-Path $root ".venv\Scripts\mlagents-learn.exe"
$log = Join-Path $root "results\evolve.log"

function Train-And-Stage {
    param(
        [string]$Creature,      # "Worm" | "Spider" (behavior name)
        [string]$ConfigPath,    # yaml with the behavior
        [string]$EnvExe,        # terrain-enabled env build
        [string]$InitFrom       # previous run-id to initialize from ("" = from scratch)
    )
    # Next version number = highest existing Assets/Agents/<Creature>_vNN + 1
    $existing = Get-ChildItem (Join-Path $root "Assets\Agents") -Directory -Filter "$($Creature)_v*" -ErrorAction SilentlyContinue |
        ForEach-Object { [int]($_.Name -replace ".*_v", "") } | Sort-Object -Descending | Select-Object -First 1
    $next = if ($existing) { $existing + 1 } else { 1 }
    $versionName = "{0}_v{1:d2}" -f $Creature, $next
    $runId = "evolve_$($versionName.ToLower())_$(Get-Date -Format 'yyyyMMdd')"

    Add-Content $log "$(Get-Date -Format o) START $runId (init-from: $InitFrom)"
    $trainArgs = @($ConfigPath, "--run-id=$runId", "--env=$EnvExe", "--no-graphics",
                   "--num-envs=8", "--base-port=5005", "--time-scale=10", "--torch-device=cpu")
    if ($InitFrom -ne "") { $trainArgs += "--initialize-from=$InitFrom" }
    & $mlagents @trainArgs
    if ($LASTEXITCODE -ne 0) {
        Add-Content $log "$(Get-Date -Format o) FAILED $runId (exit $LASTEXITCODE) - checking for a usable final export anyway"
    }

    $onnx = Join-Path $root "results\$runId\$Creature.onnx"
    if (Test-Path $onnx) {
        $destDir = Join-Path $root "Assets\Agents\$versionName"
        New-Item -ItemType Directory -Force $destDir | Out-Null
        Copy-Item $onnx (Join-Path $destDir "$versionName.onnx") -Force
        New-Item -ItemType File -Force (Join-Path $destDir "$versionName.autobake") | Out-Null
        Add-Content $log "$(Get-Date -Format o) STAGED $versionName"
    } else {
        Add-Content $log "$(Get-Date -Format o) NO EXPORT for $runId - nothing staged"
    }
    # Envs die with the trainer; make sure nothing lingers before the next run.
    Get-Process WormEnv, SpiderEnv -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force (Join-Path $root "results") | Out-Null
Add-Content $log "$(Get-Date -Format o) === EVOLVE NIGHT START ==="

Train-And-Stage -Creature "Worm"   -ConfigPath "Config\WormLoco01.yaml"   -EnvExe "Builds\WormEnv\WormEnv.exe"    -InitFrom "worm_loco02"
Train-And-Stage -Creature "Spider" -ConfigPath "Config\SpiderLoco01.yaml" -EnvExe "Builds\SpiderEnv2\SpiderEnv.exe" -InitFrom "spider_loco01"

Add-Content $log "$(Get-Date -Format o) === EVOLVE NIGHT DONE ==="
