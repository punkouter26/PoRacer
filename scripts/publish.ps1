# Scripts\publish.ps1 - build the signed AAB and upload it to Play internal testing.
#
#   .\Scripts\publish.ps1              # build + upload as DRAFT to internal track
#   .\Scripts\publish.ps1 -DryRun      # full rehearsal, nothing lands in the console
#   .\Scripts\publish.ps1 -SkipBuild   # upload the existing Builds\Android\PoRacer.aab
#
# Before a new version: bump bundleVersionCode in ProjectSettings (Player > Android),
# or Play will reject the upload as a duplicate versionCode.
# The Unity editor must NOT have this project open, or the headless build fails on
# the project lock.
param(
    [switch]$SkipBuild,
    [switch]$DryRun,
    [string]$Track = 'internal',
    [string]$Status = 'draft'
)
$ErrorActionPreference = 'Stop'
$App = 'PoRacer'
$BuildMethod = 'PoRacer.EditorTools.Editor_BuildAndroidAAB.Build'
$Proj = Split-Path $PSScriptRoot -Parent
$Python = 'C:\Users\punko\Downloads\PlayStoreUploads\publish-venv\Scripts\python.exe'
$Creds = 'C:\Users\punko\Downloads\PoRacer-Release\play-service-account.json'
$Aab = Join-Path $Proj "Builds\Android\$App.aab"

if (-not (Test-Path $Python)) { Write-Error "Publish venv python not found at $Python" }
if (-not (Test-Path $Creds)) { Write-Error "Service account key not found at $Creds" }

if (-not $SkipBuild) {
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
        '-batchmode', '-quit', '-projectPath', $Proj, '-executeMethod', $BuildMethod, '-logFile', $log
    $result = Select-String -Path $log -Pattern 'AAB BUILD RESULT:' | Select-Object -Last 1
    if ($result) { Write-Host $result.Line }
    if ($proc.ExitCode -ne 0 -or -not $result -or $result.Line -notmatch 'Succeeded') {
        Write-Error "Build failed (Unity exit $($proc.ExitCode)). If this project is open in the Unity editor, close it and retry. Details: $log"
    }
}

if (-not (Test-Path $Aab)) { Write-Error "No AAB at $Aab - run without -SkipBuild first" }
$pyArgs = @((Join-Path $Proj 'Tools\play_publish.py'),
            '--credentials', $Creds, '--track', $Track, '--status', $Status)
if ($DryRun) { $pyArgs += '--dry-run' }
& $Python @pyArgs
exit $LASTEXITCODE
