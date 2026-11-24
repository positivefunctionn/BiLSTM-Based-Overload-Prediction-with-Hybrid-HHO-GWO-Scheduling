@echo off
cd /d "%~dp0"
py experiments\run_event_simulation.py
if errorlevel 1 goto fail
py experiments\run_event_simulation.py --horizon-spacing 1.0 --output results\event_simulation_fixed_spacing_1.0
if errorlevel 1 goto fail
echo.
echo Event simulation finished. Results are in results\event_simulation\ and results\event_simulation_fixed_spacing_1.0\
pause
exit /b 0
:fail
echo.
echo Event simulation failed. Read the error above.
pause
exit /b 1
