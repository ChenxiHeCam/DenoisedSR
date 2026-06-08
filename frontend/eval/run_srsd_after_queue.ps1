# Wait for the q-scaling driver to finish (PID 75632), then launch SRSD-PySR
# end-to-end with the bug-fixed ensemble. Sequential, no CPU contention.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
$env:PYTHONUNBUFFERED = "1"

$QUEUE_PID = 75632
Write-Host "waiting for queue driver PID $QUEUE_PID (q-scaling) ..."
while ($true) {
    $p = Get-Process -Id $QUEUE_PID -ErrorAction SilentlyContinue
    if (-not $p) { Write-Host "queue done $(Get-Date -Format HH:mm:ss)"; break }
    Start-Sleep -Seconds 60
}

Write-Host "`n=== SRSD-PySR end-to-end (120 tasks, bug-fixed ensemble) $(Get-Date -Format HH:mm:ss) ==="
$env:N_TASKS = "120"; $env:TIMEOUT = "10"; $env:NITERS = "40"
$env:Q = "100"; $env:SEED = "42"
$env:OUT_PATH = "data/results/pysr_srsd_3way.json"
python -u frontend/eval/eval_pysr_srsd.py 2>&1 |
    Tee-Object -FilePath "data/results/pysr_srsd_3way.log" | Out-Null
Write-Host "SRSD-PySR done $(Get-Date -Format HH:mm:ss)"
"SRSD_PYSR_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File data/results/_srsd_pysr_done.flag
