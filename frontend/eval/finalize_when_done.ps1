# Watch for the background sweep driver (PID 86340) to finish, then re-aggregate
# results, regenerate figures, and recompile the PDF. Does NOT auto-commit.
$ErrorActionPreference = "Continue"
$watchPid = 86340
Write-Host "Watching PID $watchPid ..."
while ($true) {
  $p = Get-Process -Id $watchPid -ErrorAction SilentlyContinue
  if (-not $p) { Write-Host "PID $watchPid gone -- proceeding."; break }
  Start-Sleep -Seconds 120
}
$env:PYTHONPATH = "src;frontend/train;D:/Physics Fundation model/src;D:/Physics Fundation model/scripts"
Write-Host "Re-aggregating ..."
python frontend/eval/aggregate_sweep.py 2>&1 | Tee-Object -FilePath data/results/_aggregate_final.log | Out-Null
Write-Host "Regenerating figures ..."
Push-Location ../comms_physics/code
python make_figures.py 2>&1 | Tee-Object -FilePath ../../sr_frontend/data/results/_figures_final.log | Out-Null
Pop-Location
Write-Host "Recompiling PDF ..."
Push-Location ../comms_physics
pdflatex -interaction=nonstopmode -jobname=main_v_final main.tex 2>&1 | Out-Null
bibtex main_v_final 2>&1 | Out-Null
pdflatex -interaction=nonstopmode -jobname=main_v_final main.tex 2>&1 | Out-Null
pdflatex -interaction=nonstopmode -jobname=main_v_final main.tex 2>&1 | Out-Null
$sz = (Get-Item main_v_final.pdf -ErrorAction SilentlyContinue).Length
Write-Host "PDF: main_v_final.pdf  ($sz bytes)"
Pop-Location
Write-Host "Syncing release ..."
Copy-Item ../comms_physics/main_v_final.pdf "D:/Physics Fundation model/papers/sr_frontend_release/paper/main.pdf" -Force
Copy-Item ../comms_physics/figures/*.png "D:/Physics Fundation model/papers/sr_frontend_release/paper/figures/" -Force
Copy-Item ../comms_physics/figures/*.pdf "D:/Physics Fundation model/papers/sr_frontend_release/paper/figures/" -Force
Copy-Item data/results/multiseed_distractor_aggregate.json "D:/Physics Fundation model/papers/sr_frontend_release/data/results/" -Force
Copy-Item data/results/sweep/*.json "D:/Physics Fundation model/papers/sr_frontend_release/data/results/sweep/" -Force
if (Test-Path data/results/pysr_frontend_3way_n80.json) {
  Copy-Item data/results/pysr_frontend_3way_n80.json "D:/Physics Fundation model/papers/sr_frontend_release/data/results/" -Force
}
"FINALIZED $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File data/results/_finalized.flag
Write-Host "DONE."
