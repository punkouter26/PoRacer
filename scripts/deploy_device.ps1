<#
.SYNOPSIS
  Installs Builds/Android/PoRacer.apk on the connected device, drives the app
  through menu -> race -> debug sheet, and collects the evidence. Build the APK
  first (Editor_BuildAsync.Start("apk") from the running editor).

.DESCRIPTION
  Everything this script does is the half of a deploy that only a physically
  connected device can answer: does it install, does it start, does it draw,
  does it log anything, and what did it write down.

  It leaves four kinds of evidence under -OutDir:
    screens\*.png        menu, roster, race, debug sheet
    logcat.txt           the full log from launch
    logcat-errors.txt    just the lines worth reading
    telemetry\*.json     elo.json and race-history.json pulled off the device

  SIGNATURE CHANGE. The APK is signed with Gradle's debug key while the release
  upload key is missing (see Editor_BuildAndroid). Android refuses to install a
  package whose signing key differs from the installed one, so -Reinstall
  uninstalls first. That DELETES app data, including the telemetry this script
  pulls, so the pull of the PREVIOUS run has to happen before the uninstall —
  which is why -Reinstall does them in that order rather than uninstalling up
  front.

.PARAMETER Reinstall
  Uninstall before installing. Required when the signing key has changed;
  harmless otherwise beyond losing app data.
#>
[CmdletBinding()]
param(
    [string]$Apk = "Builds/Android/PoRacer.apk",
    [string]$OutDir = "Logs/device",
    [string]$PackageName = "com.punkoutersoftware.poracer",
    [switch]$Reinstall,
    # Seconds to let a race run before the race screenshot. The countdown alone
    # is 2.4 s, so anything under ~6 photographs the grid, not a race.
    [int]$RaceSeconds = 20
)

$ErrorActionPreference = "Stop"

$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
    $candidate = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\platform-tools\adb.exe"
    if (-not (Test-Path $candidate)) { throw "adb not found on PATH or at $candidate" }
    $adb = $candidate
} else {
    $adb = $adb.Source
}

function Adb { & $adb @args }

# --- Device ---
$devices = (Adb devices) | Select-Object -Skip 1 | Where-Object { $_ -match "\sdevice$" }
if (-not $devices) {
    throw "No device. Plug the phone in, unlock it, and accept the USB-debugging prompt."
}
$model = (Adb shell getprop ro.product.model).Trim()
$sdk = (Adb shell getprop ro.build.version.sdk).Trim()
Write-Host "Device: $model (SDK $sdk)" -ForegroundColor Cyan

# A LOCKED DEVICE SILENTLY RUINS THE RUN. The install works, the app launches and
# is even the focused activity, but the keyguard is drawn over it — so every
# screenshot is of the lock screen and every tap goes to the keyguard instead of
# the app. Nothing errors; the evidence is just quietly wrong. `wm dismiss-keyguard`
# only works on an INSECURE lock, and this device has a fingerprint on it, which
# no amount of adb can satisfy. So check, and say so plainly, up front.
$locked = (Adb shell dumpsys trust 2>&1 | Select-String -Pattern "deviceLocked=1")
if ($locked) {
    $insecure = (Adb shell locksettings get-disabled 2>&1).Trim()
    Adb shell input keyevent KEYCODE_WAKEUP | Out-Null
    Start-Sleep -Seconds 1
    Adb shell wm dismiss-keyguard | Out-Null
    Start-Sleep -Seconds 2
    $stillLocked = (Adb shell dumpsys trust 2>&1 | Select-String -Pattern "deviceLocked=1")
    if ($stillLocked) {
        throw @"
Device is LOCKED and adb cannot unlock it (secure lock disabled = $insecure).
Screenshots would capture the keyguard and taps would never reach the app, so
this script stops rather than collecting evidence that looks real and is not.

Unlock the phone by hand (fingerprint/PIN), keep the screen on, and re-run.
Optionally: Settings > Developer options > "Stay awake while charging".
"@
    }
    Write-Host "  keyguard dismissed"
}

if (-not (Test-Path $Apk)) { throw "APK not found at $Apk — build it first." }

$screens = Join-Path $OutDir "screens"
$telemetry = Join-Path $OutDir "telemetry"
New-Item -ItemType Directory -Force $screens | Out-Null
New-Item -ItemType Directory -Force $telemetry | Out-Null

function Shot([string]$name) {
    $remote = "/sdcard/poracer-shot.png"
    Adb shell screencap -p $remote | Out-Null
    Adb pull $remote (Join-Path $screens "$name.png") | Out-Null
    Adb shell rm $remote | Out-Null
    Write-Host "  captured $name.png"
}

# Taps are in DEVICE PIXELS, derived from the live screen size rather than
# hardcoded, so the script survives a different handset. The fractions are read
# off the portrait layout: the primary action sits on the bottom control row,
# and DBG is the bottom-left furniture anchor.
$size = (Adb shell wm size) -replace '.*:\s*',''
$w, $h = $size.Trim().Split('x') | ForEach-Object { [int]$_ }
Write-Host "Screen: ${w}x${h}"
function Tap([double]$fx, [double]$fy) {
    Adb shell input tap ([int]($w * $fx)) ([int]($h * $fy)) | Out-Null
}

# --- Telemetry from the PREVIOUS install, before an uninstall destroys it ---
$dataDir = "/sdcard/Android/data/$PackageName/files"
foreach ($file in @("elo.json", "race-history.json")) {
    $probe = Adb shell "ls $dataDir/$file 2>/dev/null"
    if ($probe -and $probe -notmatch "No such") {
        Adb pull "$dataDir/$file" (Join-Path $telemetry "before-$file") | Out-Null
        Write-Host "  pulled before-$file"
    }
}

# --- Install ---
if ($Reinstall) {
    Write-Host "Uninstalling (signing key changed; app data is lost)..." -ForegroundColor Yellow
    Adb uninstall $PackageName | Out-Null
}
Write-Host "Installing $Apk..." -ForegroundColor Cyan
$install = Adb install -r $Apk 2>&1 | Out-String
Write-Host $install
if ($install -notmatch "Success") {
    if ($install -match "INSTALL_FAILED_UPDATE_INCOMPATIBLE|signatures do not match") {
        throw "Signature mismatch. Re-run with -Reinstall."
    }
    throw "Install failed: $install"
}

# --- Launch, with a clean log ---
# An open notification shade sits over the app: every tap below then lands on the
# shade (and from there in the Play Store) and every screenshot is of the shade.
Adb shell cmd statusbar collapse | Out-Null
Adb shell input keyevent KEYCODE_HOME | Out-Null
Adb logcat -c | Out-Null
Adb shell monkey -p $PackageName -c android.intent.category.LAUNCHER 1 | Out-Null
Start-Sleep -Seconds 12
Shot "01-menu"

# The menu is one screen: maps, roster and RACE together. RACE is the bottom-row
# primary action, and the roster starts with one of each creature selected.
#
# 0.896, NOT 0.93. Measured off a real 960x2142 capture: the button spans roughly
# 0.862-0.930 of screen height, so 0.93 lands on its bottom border and the tap is
# swallowed. Nothing reports a miss — the next screenshot is simply of the screen
# you were already on, which reads as "the app is stuck" rather than "the tap
# missed". Keep these fractions on the CENTRE of each control.
Tap 0.5 0.896
Start-Sleep -Seconds $RaceSeconds
Shot "02-race"

# DBG is the bottom-left furniture anchor; the sheet needs a beat to populate,
# because its first refresh is also what starts the ProfilerRecorders.
Tap 0.085 0.954
Start-Sleep -Seconds 5
Shot "03-debug-sheet"

# --- Logs ---
$logPath = Join-Path $OutDir "logcat.txt"
Adb logcat -d | Out-File -FilePath $logPath -Encoding utf8
Write-Host "Wrote $logPath"

# Only OUR app's errors count. A naive "-match 'Error'" sweep over a whole
# logcat reports hundreds of lines and every one of them belongs to Finsky,
# CrashRecovery or some other system service — noise that makes a clean run look
# broken. Attribute by tag first, then subtract the two known-harmless entries:
#
#   TensorProxy   upstream ml-agents finalizer NRE (see CLAUDE.md MLOps). Harmless
#                 here because SentisModelInfo builds its Worker on the CPU
#                 backend, so the guarded body is a no-op. Log noise only.
#   AssetPackManager  Unity probing for Play Asset Delivery, which a sideloaded
#                 APK does not use. Absent by design, not a failure.
$log = Get-Content $logPath
$appErrorIndexes = @()
for ($i = 0; $i -lt $log.Count; $i++) {
    if ($log[$i] -match "\sE\s+(Unity|AndroidRuntime|libc|DEBUG)\s*:" -or $log[$i] -match "FATAL EXCEPTION") {
        $appErrorIndexes += $i
    }
}

# A managed exception spans SEVERAL log lines: the message, then its stack frames,
# then a blank. Only the FRAME lines name the type, so matching "TensorProxy"
# line-by-line suppressed two lines of every four and reported the other two as
# real errors. One clean run scored 1360 "real" errors that way, all of which were
# the message and blank halves of 680 known-harmless TensorProxy blocks.
# So classify a line by the BLOCK it belongs to: look ahead a few lines for the
# signature before deciding.
$LOOKAHEAD = 3
$known = @()
$real = @()
foreach ($i in $appErrorIndexes) {
    # The MESSAGE, not the whole line: "09-11 12:00:00.000 1 2 E Unity   : " is a
    # content-free line (the blank that terminates a stack trace) but it is not an
    # empty string, so trimming the raw line never catches it and five of them got
    # reported as real errors with nothing in them to read.
    $message = ($log[$i] -replace "^.*?\sE\s+\w+\s*:\s?", "").Trim()
    if ($message -eq "") { continue }
    $window = $log[$i..([Math]::Min($i + $LOOKAHEAD, $log.Count - 1))] -join "`n"
    if ($window -match "TensorProxy|AssetPackManager") { $known += $log[$i] }
    else { $real += $log[$i] }
}

$errPath = Join-Path $OutDir "logcat-errors.txt"
$real | Out-File -FilePath $errPath -Encoding utf8
$errors = $real

Write-Host "App-tagged error lines: $($appErrors.Count)  (known-harmless: $($known.Count), real: $($real.Count))"
Write-Host "Wrote $errPath"

# A crash is the one thing that must never pass silently.
$fatal = $log | Where-Object { $_ -match "FATAL EXCEPTION|signal 11|SIGSEGV|Force finishing activity" }
if ($fatal) {
    Write-Host "CRASH SIGNATURES FOUND:" -ForegroundColor Red
    $fatal | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
}

# --- Telemetry written by THIS run ---
foreach ($file in @("elo.json", "race-history.json")) {
    $probe = Adb shell "ls $dataDir/$file 2>/dev/null"
    if ($probe -and $probe -notmatch "No such") {
        Adb pull "$dataDir/$file" (Join-Path $telemetry "after-$file") | Out-Null
        Write-Host "  pulled after-$file"
    } else {
        Write-Host "  $file absent — the app wrote no telemetry this run" -ForegroundColor Yellow
    }
}

Write-Host "`nEvidence under $OutDir" -ForegroundColor Green
if ($errors.Count -gt 0) {
    Write-Host "$($errors.Count) error lines — read $errPath" -ForegroundColor Yellow
} else {
    Write-Host "No error lines in logcat." -ForegroundColor Green
}
