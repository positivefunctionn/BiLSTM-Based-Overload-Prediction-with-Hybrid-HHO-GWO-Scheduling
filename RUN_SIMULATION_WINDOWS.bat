@echo off
cd /d "%~dp0"
py -m src.simulation
if errorlevel 1 (
  echo Simulation failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Simulation finished. Results are in results\simulation\
pause
