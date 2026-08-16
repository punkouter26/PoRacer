# One-shot watchdog: stops the running mlagents trainer 6 hours after ITS OWN
# start time, then tears down envs (project teardown order; TensorBoard stays).
# Created 2026-08-15 to extend the live all5h run from a 5 h to a 6 h box.
$t = Get-Process mlagents-learn -ErrorAction Stop
$deadline = $t.StartTime.AddHours(6)
Write-Host "Watchdog armed: trainer PID $($t.Id), hard stop at $deadline"
while ((Get-Date) -lt $deadline) {
    if ($t.HasExited) { break }
    Start-Sleep -Seconds 60
}
if (-not $t.HasExited) {
    Write-Host "Time box reached - stopping trainer (checkpoints preserved)."
    taskkill /PID $t.Id | Out-Null
    if (-not $t.WaitForExit(120000)) { taskkill /PID $t.Id /F | Out-Null }
} else {
    Write-Host "Trainer finished on its own (max_steps)."
}
Get-Process -Name "AllEnv" -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Done. Brains in results\<run-id>\<Behavior>\*.onnx  |  TensorBoard: http://localhost:6006"
