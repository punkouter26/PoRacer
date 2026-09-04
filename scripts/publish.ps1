# Scripts\publish.ps1 - build the signed AAB and upload it to Play internal testing.
#
#   .\Scripts\publish.ps1              # build + upload as DRAFT to internal track
#   .\Scripts\publish.ps1 -DryRun      # full rehearsal, nothing lands in the console
#   .\Scripts\publish.ps1 -SkipBuild   # upload the existing Builds\Android\PoRacer.aab
#
# bundleVersionCode is bumped automatically before each build (Play rejects
# duplicate versionCodes); pass -NoBump to keep the current one.
# The Unity editor must NOT have this project open, or the headless build fails on
# the project lock — the script checks and refuses up front.
param(
    [switch]$SkipBuild,
    [switch]$DryRun,
    [switch]$NoBump,
    [string]$Track = 'internal',
    [string]$Status = 'draft'
)
$ErrorActionPreference = 'Stop'
$App = 'PoRacer'
$BuildMethod = 'PoRacer.EditorTools.Editor_BuildAndroidAAB.Build'
$Proj = Split-Path $PSScriptRoot -Parent
# The publish venv lives in the repo, NOT beside another project's uploads — the
# path this used to carry (…\PlayStoreUploads\publish-venv) does not exist here, so
# the script died on its own Test-Path guard before it could build anything. It is
# deliberately separate from .venv, which carries load-bearing ml-agents/torch pins.
$Python = Join-Path $Proj 'Tools\publish-venv\Scripts\python.exe'
$Creds = 'C:\Users\punko\Downloads\PoRacer-Release\play-service-account.json'
$Aab = Join-Path $Proj "Builds\Android\$App.aab"

if (-not (Test-Path $Python)) { Write-Error "Publish venv python not found at $Python" }
if (-not (Test-Path $Creds)) { Write-Error "Service account key not found at $Creds" }

if (-not $SkipBuild) {
    # Fail fast if the project is open in the editor — the headless build would
    # only discover the project lock after ~30s of Unity startup.
    $open = Get-Process Unity -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -like "$App -*" }
    if ($open) {
        Write-Error "$App is open in the Unity editor (PID $($open[0].Id)). Close it and retry."
    }

    # Play refuses a versionCode it has seen before, so bump it every build.
    $settings = Join-Path $Proj 'ProjectSettings\ProjectSettings.asset'
    $raw = Get-Content $settings -Raw
    if ($raw -match 'AndroidBundleVersionCode: (\d+)') {
        $current = [int]$Matches[1]
        if ($NoBump) {
            Write-Host "bundleVersionCode: $current (kept, -NoBump)"
        } else {
            $next = $current + 1
            ($raw -replace 'AndroidBundleVersionCode: \d+', "AndroidBundleVersionCode: $next") |
                Set-Content $settings -NoNewline
            Write-Host "bundleVersionCode: $current -> $next"
        }
    } else {
        Write-Warning 'AndroidBundleVersionCode not found in ProjectSettings.asset; not bumping.'
    }

    $ver = (Select-String -Path (Join-Path $Proj 'ProjectSettings\ProjectVersion.txt') `
            -Pattern 'm_EditorVersion: (.+)').Matches[0].Groups[1].Value.Trim()
    $unity = "C:\Program Files\Unity\Hub\Editor\$ver\Editor\Unity.exe"
    if (-not (Test-Path $unity)) {
        $latest = Get-ChildItem 'C:\Program Files\Unity\Hub\Editor' -Directory |
            Sort-Object Name | Select-Object -Last 1
        Write-Warning "Unity $ver not installed; falling back to $($latest.Name)"
        $unity = Join-Path $latest.FullName 'Editor\Unity.exe'
    }
    New-Item -ItemType Directory -Force (Join-Path $Proj 'Builds') | Out-Null
    $log = Join-Path $Proj 'Builds\publish-build.log'
    Write-Host "Building $App AAB headlessly (log: $log)..."
    $proc = Start-Process -FilePath $unity -PassThru -Wait -ArgumentList `
        '-batchmode', '-nographics', '-quit', '-projectPath', $Proj, `
        '-executeMethod', $BuildMethod, '-logFile', $log
    $result = Select-String -Path $log -Pattern 'AAB BUILD RESULT:' | Select-Object -Last 1
    if ($result) { Write-Host $result.Line }
    if ($proc.ExitCode -ne 0 -or -not $result -or $result.Line -notmatch 'Succeeded') {
        Select-String -Path $log -Pattern 'error CS|Error building|BuildFailedException|Aborted|could not be found' |
            Select-Object -Last 5 | ForEach-Object { Write-Host "  $($_.Line)" }
        Write-Error "Build failed (Unity exit $($proc.ExitCode)). Full log: $log"
    }
}

if (-not (Test-Path $Aab)) { Write-Error "No AAB at $Aab - run without -SkipBuild first" }
$pyArgs = @((Join-Path $Proj 'Tools\play_publish.py'),
            '--credentials', $Creds, '--track', $Track, '--status', $Status)
if ($DryRun) { $pyArgs += '--dry-run' }
& $Python @pyArgs
exit $LASTEXITCODE
