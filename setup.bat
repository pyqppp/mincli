@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   mincli Setup Script (Windows)
echo ========================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists, skipping.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [Error] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install tiktoken typer "python-dotenv>=1.0.0" "openai>=1.0.0" rich prompt-toolkit pdfminer.six python-docx trafilatura

if errorlevel 1 (
    echo [Error] Dependency installation failed. Please check network connection.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup complete!
echo   Create a .env file and fill in DEEPSEEK_API_KEY
echo   Activate virtual environment before use:
echo   venv\Scripts\activate
echo   Run:
echo   python main.py chat
echo ========================================
pause