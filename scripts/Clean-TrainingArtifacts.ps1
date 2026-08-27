# Reports (and optionally deletes) stale results/ run folders from ML-Agents
# training. A run is "stale" only when nothing still depends on it.
#
# Protection is deliberately conservative, because the cost of a false positive
# here is an unrecoverable 8-hour run:
#   - the newest run folder for each run-id prefix (e.g. the latest all8h_*)
#   - anything named *_backup_* (staged brains and demos)
#   - any run-id a Config/*.yaml pairs to under the old <Name><Phase><NN> scheme
#
# The previous version protected ONLY that last category plus runs named by
# scripts/evolve.ps1. Unattended runs are timestamped (all8h_20260826_2244), so
# they matched no config, evolve.ps1 has been deleted, and the protected set came
# out empty - which would have listed every run including the shipped brains as a
# deletion candidate.
#
# Usage:
#   .\scripts\Clean-TrainingArtifacts.ps1            # dry run: lists candidates only
#   .\scripts\Clean-TrainingArtifacts.ps1 -Delete     # deletes the listed candidates
#
# Never touches Assets/Agents/ or the CreatureCatalog - old-but-still-racing
# brain versions are kept on purpose so ELO can compare them.
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
if (-not (Test-Path $resultsDir)) {
    Write-Host "No results/ directory found; nothing to clean."
    exit 0
}

$protected = New-Object System.Collections.Generic.List[string]

$runDirs = Get-ChildItem $resultsDir -Directory

# Newest run per prefix. "all8h_20260826_2244" -> prefix "all8h". A folder with
# no underscore is its own prefix, so it always protects itself.
$runDirs | Group-Object { ($_.Name -split '_')[0] } | ForEach-Object {
    $newest = $_.Group | Sort-Object Name -Descending | Select-Object -First 1
    $protected.Add($newest.Name)
}

# Staged brains and demos.
$runDirs | Where-Object { $_.Name -like "*_backup_*" } | ForEach-Object {
    $protected.Add($_.Name)
}

Get-ChildItem $configDir -Filter "*.yaml" | ForEach-Object {
    # <Name><Phase><NN>.yaml -> <name>_<phase><nn>, per the project's own
    # Config/run-id pairing rule. Kept for any run still using that scheme.
    if ($_.BaseName -cmatch '^([A-Z][a-z]*)([A-Z].*)$') {
        $protected.Add(("{0}_{1}" -f $Matches[1], $Matches[2]).ToLowerInvariant())
    }
}
$protected = $protected | Select-Object -Unique

Write-Host "Protected run-ids (newest per prefix / backups / Config baseline):"
$protected | ForEach-Object { Write-Host "  $_" }
Write-Host ""

$candidates = Get-ChildItem $resultsDir -Directory | Where-Object { $protected -notcontains $_.Name }

if (-not $candidates) {
    Write-Host "Nothing to clean - every results/ run is protected or already gone."
    exit 0
}

$rows = $candidates | ForEach-Object {
    $sizeMb = "{0:N1}" -f ((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum / 1MB)
    [pscustomobject]@{ RunId = $_.Name; SizeMB = $sizeMb }
}

Write-Host "Stale run candidates (superseded, not a backup, unreferenced by any Config):"
$rows | Format-Table RunId, SizeMB -AutoSize

if ($Delete) {
    Write-Host "Deleting..." -ForegroundColor Yellow
    foreach ($candidate in $candidates) {
        Remove-Item $candidate.FullName -Recurse -Force
        Write-Host "  deleted $($candidate.Name)"
    }
} else {
    Write-Host "Dry run only - re-run with -Delete to remove these folders."
}
