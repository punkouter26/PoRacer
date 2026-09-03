<#
.SYNOPSIS
  Sets up Isaac Lab inside PoRacer\ISAAC\ and installs the Boy task package into it.

.DESCRIPTION
  1. Clones Isaac Lab into ISAAC\isaaclab (skipped if present).
  2. Creates ISAAC\isaaclab\.venv (Python 3.11) and pip-installs Isaac Sim into it, the way
     the Isaac Lab "pip installation" docs describe. Isaac Sim is ~10 GB of wheels.
  3. Runs isaaclab.bat --install rsl_rl, then pip install -e ISAAC\boy_tasks plus onnx,
     onnxruntime and tensorboard.

  Nothing here touches PoRacer\.venv - that one carries the ML-Agents/torch pins and must
  stay in exact parity with the C# package.

  HARDWARE NOTE (re-checked 2026-09-02): this machine is an RTX 5070 Ti Laptop, 12 GB,
  compute capability 12.0 (Blackwell / sm_120). That clears Isaac Sim's stated 8 GB
  minimum, so the old RTX 2060 "use --num_envs 512 or train elsewhere" advice is gone;
  2048 envs is the expected working point.

  Blackwell needs Isaac Sim 5.0 or newer. sm_120 has no kernels in the torch 2.5/cu118
  build Isaac Sim 4.5 ships, and the first CUDA op dies with "no kernel image is available
  for execution on the device". Isaac Sim 5.0 ships torch 2.7/cu128, which does carry
  sm_120. Step 4 asserts this before any training is attempted.

.PARAMETER IsaacLabRef
  Git ref of Isaac Lab to check out. Default: main.

.PARAMETER Python
  Python 3.11 interpreter to build the venv from. Either a path to python.exe or a
  launcher invocation like "py -3.11". Default: the repo-local PoRacer\Python311, which
  exists because this machine has no system 3.11, and PoRacer\.venv's 3.10 carries the
  load-bearing ml-agents/torch pins that must not be resolved over.
#>
param(
    [string]$IsaacLabRef = "main",
    [string]$Python = "$PSScriptRoot\..\Python311\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Lab = Join-Path $Root "isaaclab"

Write-Host "== Boy / Isaac Lab setup in $Root"

# ---- 1. Isaac Lab checkout -------------------------------------------------------------
if (-not (Test-Path (Join-Path $Lab "isaaclab.bat"))) {
    Write-Host "-- cloning Isaac Lab ($IsaacLabRef) into $Lab"
    git clone --depth 1 --branch $IsaacLabRef https://github.com/isaac-sim/IsaacLab.git $Lab
} else {
    Write-Host "-- Isaac Lab already present at $Lab"
}

# ---- 2. venv + Isaac Sim --------------------------------------------------------------
$Venv = Join-Path $Lab ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "-- creating $Venv"
    # $Python is either a single path or a launcher plus switches ("py -3.11"). Split
    # handles both, but [1..99] over a one-element array yields $nulls that venv chokes on.
    $pyParts = @($Python.Split(" ") | Where-Object { $_ -ne "" })
    $pyArgs = if ($pyParts.Count -gt 1) { $pyParts[1..($pyParts.Count - 1)] } else { @() }
    & $pyParts[0] @pyArgs -m venv $Venv
    if (-not (Test-Path $Py)) { throw "venv creation failed: no interpreter at $Py" }
}
& $Py -m pip install --upgrade pip
$hasSim = & $Py -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('isaacsim') else 1)"; $simOk = $LASTEXITCODE -eq 0
if (-not $simOk) {
    Write-Host "-- installing Isaac Sim wheels (large download)"
    & $Py -m pip install "isaacsim[all,extscache]" --extra-index-url https://pypi.nvidia.com
}

# ---- 3a. torch with CUDA -----------------------------------------------------------------
# Isaac Sim's wheels declare a bare torch==2.7.0, and on Windows the PyPI default for that
# is the CPU-ONLY build - it imports fine and then dies with "Torch not compiled with CUDA
# enabled". Install the cu128 build explicitly. 2.7.0+cu128 still satisfies isaacsim-core's
# torch==2.7.0 pin, because PEP 440 ignores the local version segment when matching.
# The +cu128 local version is REQUIRED in these specifiers. Asking for a bare "torch==2.7.0"
# is a no-op when the CPU build is already there: PEP 440 ignores the local segment when
# matching, so pip reads the installed 2.7.0+cpu as already satisfying it and skips silently.
Write-Host "-- installing torch 2.7.0+cu128 into the venv"
& $Py -m pip install --index-url https://download.pytorch.org/whl/cu128 `
    "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"
if ($LASTEXITCODE -ne 0) { throw "cu128 torch install failed" }

# ---- 3b. Isaac Lab + rsl_rl + the Boy task ----------------------------------------------
# isaaclab.bat picks its interpreter in :extract_python_exe as conda -> _isaac_sim\python.bat
# -> the FIRST "where python" on PATH. It NEVER reads ISAACLAB_PYTHON; we used to set that
# and it did precisely nothing. On 2026-09-02 that silently sent 3.3 GB of cu128 torch and
# an editable isaaclab into the SYSTEM Python 3.10 and left this venv untouched. So put the
# venv at the head of PATH, clear CONDA_PREFIX, and assert what it actually resolved before
# handing it a single install.
$VenvScripts = Split-Path $Py
$env:PATH = "$VenvScripts;$env:PATH"
$env:CONDA_PREFIX = ""
$resolved = (& where.exe python 2>$null | Select-Object -First 1)
if ($resolved -ne $Py) {
    throw "isaaclab.bat would install into '$resolved', not the venv '$Py'. Refusing: this is exactly the bug that polluted system Python 3.10."
}
Write-Host "-- isaaclab.bat will use $resolved"

Push-Location $Lab
try {
    Write-Host "-- isaaclab.bat --install rsl_rl"
    & (Join-Path $Lab "isaaclab.bat") --install rsl_rl
} finally {
    Pop-Location
}

# ---- 3c. undo what isaaclab.bat breaks ---------------------------------------------------
# isaaclab.bat's torch step (its :install_torch) probes the current version with
#   pip show torch | findstr /B /C:"Version:"
# which comes back EMPTY under pip 26, so it always believes torch is wrong. It then runs
#   pip uninstall -y torch torchvision torchaudio
#   pip install torch==X torchvision==Y --index-url <cu128>
# - note torchaudio is uninstalled and never put back, while isaacsim-core requires it. It
# also drags psutil off isaacsim-kernel's exact 5.9.8 pin. Both are silent until something
# imports them, so repair them here and print the result rather than trust the exit code.
Write-Host "-- repairing what isaaclab.bat's torch step removed"
& $Py -m pip install --index-url https://download.pytorch.org/whl/cu128 "torchaudio==2.7.0+cu128"
& $Py -m pip install "psutil==5.9.8"

# tensordict ships a COMPILED _C extension built against one exact torch ABI, but rsl-rl-lib
# asks for a bare "tensordict>=0.7.0", so pip takes the newest (0.14.0, built for torch 2.9+).
# Against torch 2.7 that does not raise ImportError - it hard-crashes the process inside
# PyInit__C, taking Isaac Sim's crash reporter with it and leaving a minidump instead of a
# traceback. The 0.8.x line is the one built for torch 2.7.
& $Py -m pip install "tensordict==0.8.3"

Write-Host "-- installing boy_tasks (editable) + export deps"
& $Py -m pip install -e (Join-Path $Root "boy_tasks")

# These four were installed unpinned once and pip happily dragged numpy 1.26.0 -> 2.4.6 and
# typing_extensions 4.12.2 -> 4.16.0, both of which isaacsim-kernel pins exactly. Constrain
# them so a transitive resolve cannot walk over the simulator's environment again.
$constraints = Join-Path $env:TEMP "boy_isaac_constraints.txt"
@("numpy==1.26.0", "typing_extensions==4.12.2") | Set-Content -Path $constraints -Encoding ascii
# usd-core is deliberately ABSENT here. Isaac Sim ships its own USD (pxr) inside its kit
# extensions; a PyPI usd-core in site-packages shadows it, and once kit has loaded its USD
# DLLs the duplicate blows up with "DLL load failed while importing _tf" - which surfaces as
# an unrelated-looking traceback deep inside isaaclab.utils.mesh. Install usd-core into the
# STANDALONE Python311 instead, where build_boy_rig.py's optional validation uses it.
& $Py -m pip install -c $constraints onnx onnxruntime tensorboard
Remove-Item $constraints -ErrorAction SilentlyContinue

$Standalone = Join-Path $Root "..\Python311\python.exe"
if (Test-Path $Standalone) {
    Write-Host "-- usd-core into the standalone Python311 (NOT the Isaac venv)"
    & $Standalone -m pip install usd-core
}

# ---- 4. GPU / torch sanity: Blackwell needs cu128 ---------------------------------------
# This box is sm_120. A torch built against cu118/cu121 imports and reports is_available()
# True, then dies on the first real kernel with "no kernel image is available for execution
# on the device" - several GB of download and a simulator start later. Catch it here.
Write-Host "-- checking torch carries kernels for this GPU"
$gpuCheck = @'
import sys, torch
cap = torch.cuda.get_device_capability()
arch = torch.cuda.get_arch_list()
print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("device     :", torch.cuda.get_device_name(0), cap)
print("arch list  :", arch)
tag = "sm_%d%d" % cap
if tag not in arch and not any(a.startswith("sm_90") and cap[0] >= 12 for a in arch):
    sys.exit("FAIL: torch has no kernels for %s (arch list %s). Blackwell needs a cu128 "
             "build, i.e. Isaac Sim 5.0+. Reinstall with a newer isaacsim." % (tag, arch))
x = torch.randn(2048, 2048, device="cuda")
(x @ x).sum().item()   # a real kernel, not just is_available()
print("matmul OK  : kernels execute on", tag)
'@
$gpuCheck | & $Py -
if ($LASTEXITCODE -ne 0) { throw "GPU/torch check failed - see the message above." }

# ---- 5. rig outputs ---------------------------------------------------------------------
# Run the rig build under the STANDALONE python: it is pure Python, and only that
# interpreter has usd-core, so the USD validation actually runs instead of silently skipping.
$RigPy = if (Test-Path $Standalone) { $Standalone } else { $Py }
Write-Host "-- regenerating the rig (USD + json) with $RigPy"
& $RigPy (Join-Path $Root "boy_rig\build_boy_rig.py")

# Report the remaining resolver state. One conflict is expected and upstream: isaacsim-kernel
# pins fastapi 0.115.7 (which wants starlette<0.46) while isaaclab pins starlette==0.49.1.
# Headless training does not exercise Isaac Sim's web UI, so it is noted, not "fixed".
Write-Host ""
Write-Host "-- pip check (a starlette/fastapi conflict here is known and upstream):"
& $Py -m pip check

Write-Host ""
Write-Host "Done. Next - call the venv python DIRECTLY, not isaaclab.bat:"
Write-Host "  $Py $Root\scripts\train.py --num_envs 2048 --headless"
Write-Host "  $Py $Root\scripts\export_bundle.py"
Write-Host ""
Write-Host "isaaclab.bat resolves its interpreter from PATH and picks the system Python,"
Write-Host "which has no isaacsim - and it still exits 0 while printing the traceback, so"
Write-Host "the failure is easy to miss. Check logs for tracebacks, not exit codes."
Write-Host "TensorBoard: http://localhost:6006 (train.py starts it)."
