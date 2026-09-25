# Reports (and optionally deletes) superseded training checkpoints from the MuJoCo and
# Isaac Lab run folders (AGENTS.md rules C and J).
#
# Scanned:
#   training/<creature>/mujoco/runs/<run>/   MuJoCo Warp + torch PPO
#   ISAAC/logs/<rsl_rl|rsl_rl_v3>/<task>/<run>/   Isaac Lab / RSL-RL (model_<N>.pt)
#
# Kept, always:
#   - every TensorBoard event file, config and summary (the curves stay readable)
#   - the newest model_<N>.pt in each run, so the run can be resumed
#   - any checkpoint a shipped brain was exported from: every path named by a
#     "checkpoint" field in training/*/export/*_report.json
#
# Everything else - intermediate model_<N>.pt snapshots - is a candidate.
# Empty run folders are candidates too.
#
# Usage:
#   .\scripts\Clean-TrainingArtifacts.ps1            # dry run: lists candidates only
#   .\scripts\Clean-TrainingArtifacts.ps1 -Delete    # deletes the listed candidates
#
# Stop TensorBoard before using -Delete: Windows file handles silently fail the wipe.
param(
    [switch]$Delete
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$exported = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::OrdinalIgnoreCase)
Get-ChildItem (Join-Path $root "training") -Recurse -Filter "*_report.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq "export" } | ForEach-Object {
        $report = Get-Content $_.FullName -Raw | ConvertFrom-Json
        if ($report.checkpoint) {
            [void]$exported.Add([IO.Path]::GetFullPath((Join-Path $root $report.checkpoint)))
        }
    }

$runDirs = @()
$runDirs += Get-ChildItem (Join-Path $root "training") -Directory -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName "mujoco\runs" } | Where-Object { Test-Path $_ } |
    ForEach-Object { Get-ChildItem $_ -Directory }
$isaacLogs = Join-Path $root "ISAAC\logs"
if (Test-Path $isaacLogs) {
    $runDirs += Get-ChildItem $isaacLogs -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory } |
        ForEach-Object { $_; Get-ChildItem $_.FullName -Directory }
}

$candidates = New-Object System.Collections.Generic.List[System.IO.FileSystemInfo]
foreach ($run in $runDirs) {
    $contents = @(Get-ChildItem $run.FullName -Force)
    if ($contents.Count -eq 0) {
        $candidates.Add($run)
        continue
    }
    $snapshots = @($contents | Where-Object { $_.Name -match '^model_(\d+)\.pt$' } |
        Sort-Object { [int]($_.Name -replace '\D', '') })
    if ($snapshots.Count -le 1) {
        continue
    }
    $newest = $snapshots[-1]
    foreach ($snapshot in $snapshots) {
        if ($snapshot -ne $newest -and -not $exported.Contains($snapshot.FullName)) {
            $candidates.Add($snapshot)
        }
    }
}

if ($candidates.Count -eq 0) {
    Write-Host "Nothing to clean - every run holds only its newest and exported checkpoints."
    exit 0
}

$totalMb = ($candidates | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Superseded checkpoints / empty runs ({0} items, {1:N1} MB):" -f $candidates.Count, $totalMb)
$candidates | ForEach-Object { Write-Host ("  " + $_.FullName.Substring($root.Length + 1)) }

if ($Delete) {
    Write-Host "Deleting..." -ForegroundColor Yellow
    foreach ($candidate in $candidates) {
        Remove-Item $candidate.FullName -Recurse -Force
    }
    Write-Host "Done."
} else {
    Write-Host "Dry run only - re-run with -Delete to remove these."
}
