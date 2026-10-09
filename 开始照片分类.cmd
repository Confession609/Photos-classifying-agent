@echo off
setlocal
chcp 65001 >nul
title Photo Classification Agent
set "taskPython=%~dp0.venv\Scripts\python.exe"
set "taskEntry=%~dp0scripts\classify_daily_photos.py"
if not exist "%taskPython%" goto missing_python
if not exist "%taskEntry%" goto missing_entry
echo Starting the local photo workflow. Please keep this window open.
echo.
"%taskPython%" -B -X utf8 "%taskEntry%" %*
set "taskExitCode=%ERRORLEVEL%"
echo.
if "%taskExitCode%"=="0" goto completed
echo The workflow exited with code %taskExitCode%. See the error above.
goto finished

:completed
echo Command finished successfully. Review the details printed above.
goto finished

:missing_python
echo ERROR: Project Python is missing: "%taskPython%"
set "taskExitCode=1"
goto finished

:missing_entry
echo ERROR: Workflow entry is missing: "%taskEntry%"
set "taskExitCode=1"

:finished
echo.
echo The window will remain open until you press a key.
pause
exit /b %taskExitCode%
