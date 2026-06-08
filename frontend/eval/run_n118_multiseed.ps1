# Full Feynman 118 x 3 seeds. Replaces the 30-equation headline.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
$env:PYTHONUNBUFFERED = "1"
$env:N_TASKS = "118"
$env:TIMEOUT = "10"
$env:NITERS  = "40"
$env:Q       = "100"
$env:N_DIST  = "20"

foreach ($s in @(42, 43, 44)) {
  $out = "data/results/pysr_frontend_3way_n118_seed${s}.json"
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) {
    Write-Host "SKIP $out ($((Get-Item $out).Length) bytes)"; continue
  }
  Write-Host "n118 seed=$s start $(Get-Date -Format HH:mm:ss)"
  $env:SEED = "$s"
  $env:OUT_PATH = $out
  python -u frontend/eval/eval_pysr_frontend.py 2>&1 |
    Tee-Object -FilePath "data/results/pysr_frontend_3way_n118_seed${s}.log" |
    Out-Null
  Write-Host "  done $(Get-Date -Format HH:mm:ss) -> $out"
}
Write-Host "ALL DONE $(Get-Date -Format HH:mm:ss)"
