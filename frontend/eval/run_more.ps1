# Additional runs: distractor multi-seed, Feynman 80-task scale-up, gplearn multi-seed.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
$env:PYTHONUNBUFFERED = "1"

# ---- (1) Distractor sweep at additional seeds (seed=42 d={5,10,30} already done) ----
$env:N_TASKS = "30"; $env:TIMEOUT = "10"; $env:NITERS = "40"; $env:Q = "100"
$pairs = @(
  @(43, 5), @(43, 10), @(43, 30),
  @(44, 5), @(44, 10), @(44, 30)
)
$i = 0
foreach ($p in $pairs) {
  $i++
  $s = $p[0]; $d = $p[1]
  $out = "data/results/sweep/3way_seed${s}_d${d}.json"
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) {
    Write-Host "[A$i] SKIP $out"; continue
  }
  Write-Host "[A$i/6] PySR distractor seed=$s d=$d  $(Get-Date -Format HH:mm:ss)"
  $env:SEED = "$s"; $env:N_DIST = "$d"; $env:OUT_PATH = $out
  python -u frontend/eval/eval_pysr_frontend.py 2>&1 |
    Tee-Object -FilePath "data/results/sweep/3way_seed${s}_d${d}.log" |
    Out-Null
  Write-Host "  done $(Get-Date -Format HH:mm:ss)"
}

# ---- (2) Scale to 80 Feynman tasks at headline config ----
$env:N_TASKS = "80"; $env:TIMEOUT = "10"; $env:NITERS = "40"; $env:Q = "100"
$env:SEED = "42"; $env:N_DIST = "20"
$env:OUT_PATH = "data/results/pysr_frontend_3way_n80.json"
if ((Test-Path $env:OUT_PATH) -and ((Get-Item $env:OUT_PATH).Length -gt 1000)) {
  Write-Host "[B] SKIP n80 exists"
} else {
  Write-Host "[B] PySR 80-Feynman scale-up  $(Get-Date -Format HH:mm:ss)"
  python -u frontend/eval/eval_pysr_frontend.py 2>&1 |
    Tee-Object -FilePath "data/results/pysr_frontend_3way_n80.log" |
    Out-Null
  Write-Host "  done $(Get-Date -Format HH:mm:ss)"
}

# ---- (3) gplearn multi-seed (seed 42 already done) ----
$env:N_TASKS = "30"; $env:N_DIST = "20"; $env:Q = "100"
$env:GP_GENS = "20"; $env:POP = "500"
foreach ($s in @(43, 44)) {
  $out = "data/results/gplearn_backend_3way_seed${s}.json"
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) {
    Write-Host "[C seed=$s] SKIP"; continue
  }
  Write-Host "[C] gplearn seed=$s  $(Get-Date -Format HH:mm:ss)"
  $env:SEED = "$s"; $env:OUT_PATH = $out
  python -u frontend/eval/eval_gplearn_backend.py 2>&1 |
    Tee-Object -FilePath "data/results/gplearn_backend_3way_seed${s}.log" |
    Out-Null
  Write-Host "  done $(Get-Date -Format HH:mm:ss)"
}

Write-Host "ALL DONE $(Get-Date -Format HH:mm:ss)"
