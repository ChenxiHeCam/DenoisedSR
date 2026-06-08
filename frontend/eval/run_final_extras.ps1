# Wait for the seed=43 retry to finish, then run SRSD-PySR end-to-end + q-scaling rerun at tau=0.10.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
$env:PYTHONUNBUFFERED = "1"
$SEED43_RETRY_PID = 53016
Write-Host "waiting for seed=43 retry (PID $SEED43_RETRY_PID) ..."
while ($true) {
    $p = Get-Process -Id $SEED43_RETRY_PID -ErrorAction SilentlyContinue
    if (-not $p) { Write-Host "seed=43 retry done."; break }
    Start-Sleep -Seconds 60
}

# 1) SRSD-PySR end-to-end on all 120 SRSD-Feynman dummy tasks (single seed first)
Write-Host "`n=== SRSD-PySR end-to-end (120 tasks) $(Get-Date -Format HH:mm:ss) ==="
$env:N_TASKS = "120"; $env:TIMEOUT = "10"; $env:NITERS = "40"
$env:Q = "100"; $env:SEED = "42"
$env:OUT_PATH = "data/results/pysr_srsd_3way.json"
python -u frontend/eval/eval_pysr_srsd.py 2>&1 |
    Tee-Object -FilePath "data/results/pysr_srsd_3way.log" | Out-Null
Write-Host "SRSD-PySR done $(Get-Date -Format HH:mm:ss)"

# 2) q-scaling rerun at deployed tau=0.10 on 50 Feynman tasks, q in {50, 100, 200}, 3 seeds
Write-Host "`n=== q-scaling rerun at tau=0.10 $(Get-Date -Format HH:mm:ss) ==="
$env:N_TASKS = "50"; $env:N_DIST = "10"
foreach ($q in @(50, 100, 200)) {
    foreach ($s in @(42, 43, 44)) {
        $out = "data/results/pysr_qscaling_tau010_s${s}_q${q}.json"
        if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) { continue }
        $env:Q = "$q"; $env:SEED = "$s"; $env:OUT_PATH = $out
        Write-Host "  q=$q seed=$s start $(Get-Date -Format HH:mm:ss)"
        python -u frontend/eval/eval_pysr_frontend.py 2>&1 |
            Tee-Object -FilePath "data/results/pysr_qscaling_tau010_s${s}_q${q}.log" | Out-Null
        Write-Host "    done $(Get-Date -Format HH:mm:ss)"
    }
}

# Final flag file
"FINAL_EXTRAS_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Out-File data/results/_final_extras.flag
Write-Host "`nALL DONE $(Get-Date -Format HH:mm:ss)"
