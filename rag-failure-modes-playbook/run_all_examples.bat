@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo 🚀 RAG Failure Modes Playbook - Dynamic Auto-Runner
echo ================================================================================

:: Check for Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3.11 not found.
    pause
    exit /b
)

:: Loop through every .py file in the current folder
for /f "tokens=*" %%f in ('dir /b /on *.py') do (
    :: Use findstr to only match files starting with two digits (01, 02, etc.)
    echo %%f | findstr /R "^[0-9][0-9]_" >nul
    if !errorlevel! equ 0 (
        echo.
        echo 🏃 Running: %%f
        echo --------------------------------------------------------------------------------
        
        :: CALL is required to return to the loop after Python finishes
        call py -3.11 "%%f"
        
        echo.
        echo ✅ Finished: %%f
        set "choice="
        set /p "choice=Press Enter for next example, or 'q' to quit: "
        if /i "!choice!"=="q" goto :end
    )
)

:end
echo.
echo ✅ All selected examples completed.
pause