# Reports (and optionally deletes) stale results/ run folders from ML-Agents
# training. A run is "stale" only when nothing still depends on it:
#   - not the run-id a Config/*.yaml pairs to 1:1 (e.g. WormLoco01.yaml -> worm_loco01)
#   - not passed as -InitFrom by scripts/evolve.ps1 (the nightly evolution loop
#     resumes training from these; deleting one breaks the next overnight run)
# Never touches Assets/Agents/ or the CreatureCatalog — old-but-still-racing
# brain versions are kept on purpose so ELO can compare them (see evolve.ps1).
#
# Usage:
#   .\scripts\Clean-TrainingArtifacts.ps1            # dry run: lists candidates only
#   .\scripts\Clean-TrainingArtifacts.ps1 -Delete     # deletes the listed candidates
#
# Stop TensorBoard before using -Delete: Windows file handles silently fail
# the wipe otherwise (per project rules).
param(
    [switch]$Delete
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$resultsDir = Join-Path $root "results"
$configDir = Join-Path $root "Config"
$evolveScript = Join-Path $root "scripts\evolve.ps1"

if (-not (Test-Path $resultsDir)) {
    Write-Host "No results/ directory found; nothing to clean."
    exit 0
}

$protected = New-Object System.Collections.Generic.List[string]

if (Test-Path $evolveScript) {
    $evolveText = Get-Content $evolveScript -Raw
    [regex]::Matches($evolveText, '-InitFrom\s+"([^"]+)"') |
        ForEach-Object { $protected.Add($_.Groups[1].Value) }
}

Get-ChildItem $configDir -Filter "*.yaml" | ForEach-Object {
    # <Name><Phase><NN>.yaml -> <name>_<phase><nn>, per the project's own
    # Config/run-id pairing rule.
    if ($_.BaseName -cmatch '^([A-Z][a-z]*)([A-Z].*)$') {
        $protected.Add(("{0}_{1}" -f $Matches[1], $Matches[2]).ToLowerInvariant())
    }
}
$protected = $protected | Select-Object -Unique

Write-Host "Protected run-ids (evolve init-from / Config baseline):"
$protected | ForEach-Object { Write-Host "  $_" }
Write-Host ""

$candidates = Get-ChildItem $resultsDir -Directory | Where-Object { $protected -notcontains $_.Name }

if (-not $candidates) {
    Write-Host "Nothing to clean — every results/ run is protected or already gone."
    exit 0
}

$rows = $candidates | ForEach-Object {
    $sizeMb = "{0:N1}" -f ((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum / 1MB)
    [pscustomobject]@{ RunId = $_.Name; SizeMB = $sizeMb }
}

Write-Host "Stale run candidates (unreferenced by evolve.ps1 or any Config):"
$rows | Format-Table RunId, SizeMB -AutoSize

if ($Delete) {
    Write-Host "Deleting..." -ForegroundColor Yellow
    foreach ($candidate in $candidates) {
        Remove-Item $candidate.FullName -Recurse -Force
        Write-Host "  deleted $($candidate.Name)"
    }
} else {
    Write-Host "Dry run only — re-run with -Delete to remove these folders."
}
