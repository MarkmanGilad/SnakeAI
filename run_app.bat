@echo off
rem run_app.bat - Run Environment.py using the project's virtual environment Python
rem Usage: run_app.bat [--pause] [args...]

setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_DIR%Trainer.py" %*
) else (
  echo Virtual environment not found at "%VENV_PY%".& echo Running system Python instead.
  python "%SCRIPT_DIR%Trainer.py" %*
)

rem add `--pause` as the first arg if you want the window to stay open after completion
if "%1"=="--pause" pause
endlocal
