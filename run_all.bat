@echo off
REM Windows runner. Safe to stop (Ctrl+C) and restart: finished runs are skipped.
setlocal EnableDelayedExpansion
if "%MODEL%"=="" set MODEL=qwen3:8b
if "%DEFENCES%"=="" set DEFENCES=none repeat_prompt pi_detector tool_filter melon melon_firststep desc_scan
if "%CHANNELS%"=="" set CHANNELS=benign A B1 B2 B3
if "%GOALS%"=="" set GOALS=goal_1 goal_2
if "%FRAMING%"=="" set FRAMING=yes
if "%OUT%"=="" set OUT=results

set SAMPLING=--temperature 0
echo %MODEL% | findstr /b "qwen3" >nul && set SAMPLING=--temperature 0.6 --top-p 0.95

for %%D in (%DEFENCES%) do (
  for %%C in (%CHANNELS%) do (
    echo === %MODEL% ^| %%C ^| %%D ===
    python -m mcpsec.run --model %MODEL% %SAMPLING% --channel %%C --defence %%D --goals %GOALS% --out %OUT% --save-messages
  )
)
if /i "%FRAMING%"=="yes" (
  for %%C in (A_provider B1_user) do (
    echo === %MODEL% ^| %%C ^| none ===
    python -m mcpsec.run --model %MODEL% %SAMPLING% --channel %%C --defence none --goals %GOALS% --out %OUT% --save-messages
  )
)
echo Done.
endlocal