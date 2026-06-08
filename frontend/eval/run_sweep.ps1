# Minimal sweep: variance at headline (seed=43,44 at d=20) + distractor sweep
# at seed=42 (d=5,10,30). seed=42/d=20 already exists as headline file.
# python -u for live progress logging.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
$env:PYTHONUNBUFFERED = "1"
$env:N_TASKS = "30"
$env:TIMEOUT = "10"
$env:NITERS  = "40"
$env:Q       = "100"

# Jobs: (seed, n_dist)
$jobs = @(
  @(43, 20),  # seed variance at headline
  @(44, 20),
  @(42,  5),  # distractor sweep at seed 42
  @(42, 10),
  @(42, 30)
)
$total = $jobs.Count
$i = 0
foreach ($pair in $jobs) {
  $i++
  $s = $pair[0]; $d = $pair[1]
  $out = "data/results/sweep/3way_seed${s}_d${d}.json"
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) {
    Write-Host "[$i/$total] SKIP exists: $out"
    continue
  }
  Write-Host "[$i/$total] seed=$s n_dist=$d -> $out  $(Get-Date -Format HH:mm:ss)"
  $env:SEED   = "$s"
  $env:N_DIST = "$d"
  $env:OUT_PATH = $out
  $log = "data/results/sweep/3way_seed${s}_d${d}.log"
  python -u frontend/eval/eval_pysr_frontend.py 2>&1 | Tee-Object -FilePath $log | Out-Null
  if (Test-Path $out) {
    $sz = (Get-Item $out).Length
    Write-Host "  OK ($sz bytes)  $(Get-Date -Format HH:mm:ss)"
  } else {
    Write-Host "  FAILED — see $log"
  }
}
Write-Host "DONE all $total runs."
