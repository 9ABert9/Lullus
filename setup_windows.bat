@echo off
REM UniMentor Setup Script for Windows
REM This script sets up everything needed to run UniMentor locally.

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   UniMentor Setup for Windows
echo ========================================
echo.
echo Setting up your local AI course assistant...
echo Estimated total time: 5-15 minutes (depends on model download speed)
echo.

set TOTAL_STEPS=7
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM --- Step 1: Check Python ---
echo [STEP 1/%TOTAL_STEPS%] Checking Python installation...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python is not installed.
    echo.
    echo   Please install Python 3.10 or higher from:
    echo     https://www.python.org/downloads/
    echo.
    echo   IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo   [OK] Python %PYTHON_VERSION% found

REM Parse version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo   [ERROR] Python 3.10+ is required. Found %PYTHON_VERSION%.
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 10 (
    echo   [ERROR] Python 3.10+ is required. Found %PYTHON_VERSION%.
    pause
    exit /b 1
)

REM --- Step 2: Check Ollama ---
echo.
echo [STEP 2/%TOTAL_STEPS%] Checking Ollama installation...

where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo   [WARNING] Ollama is not installed.
    echo.
    echo   Please install Ollama from: https://ollama.com/download
    echo   Download and run the Windows installer.
    echo.
    echo   Press any key after installing Ollama to continue...
    pause >nul
    where ollama >nul 2>&1
    if %errorlevel% neq 0 (
        echo   [ERROR] Ollama still not found. Please install it and re-run this script.
        pause
        exit /b 1
    )
)

echo   [OK] Ollama is installed

REM Check if Ollama is running
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo   [INFO] Starting Ollama...
    start "" ollama serve
    timeout /t 5 /nobreak >nul
)

echo   [OK] Ollama is running

REM --- Step 3: Pull LLM model ---
echo.
echo [STEP 3/%TOTAL_STEPS%] Pulling language model (~4.1 GB)...
echo   Model: mistral:7b-instruct-v0.3-q4_K_M

ollama list 2>nul | findstr /i "mistral:7b-instruct-v0.3-q4_K_M" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Model already downloaded
) else (
    echo   [INFO] Downloading... this may take 2-10 minutes...
    ollama pull mistral:7b-instruct-v0.3-q4_K_M
    if %errorlevel% neq 0 (
        echo   [WARNING] Could not pull model. Run manually: ollama pull mistral:7b-instruct-v0.3-q4_K_M
    ) else (
        echo   [OK] Model downloaded successfully
    )
)

REM --- Step 4: Pull embedding model ---
echo.
echo [STEP 4/%TOTAL_STEPS%] Pulling embedding model (~274 MB)...
echo   Model: nomic-embed-text

ollama list 2>nul | findstr /i "nomic-embed-text" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Embedding model already downloaded
) else (
    echo   [INFO] Downloading...
    ollama pull nomic-embed-text
    if %errorlevel% neq 0 (
        echo   [WARNING] Could not pull model. Run manually: ollama pull nomic-embed-text
    ) else (
        echo   [OK] Embedding model downloaded successfully
    )
)

REM --- Step 5: Create virtual environment ---
echo.
echo [STEP 5/%TOTAL_STEPS%] Setting up Python virtual environment...

if exist "venv" (
    echo   [OK] Virtual environment already exists
) else (
    echo   [INFO] Creating virtual environment...
    python -m venv venv
    echo   [OK] Virtual environment created
)

REM Activate venv
call venv\Scripts\activate.bat
echo   [OK] Virtual environment activated

REM --- Step 6: Install Python dependencies ---
echo.
echo [STEP 6/%TOTAL_STEPS%] Installing Python dependencies...
echo   This may take 1-3 minutes...

pip install --upgrade pip --quiet 2>nul
pip install -r requirements.txt --quiet
if %errorlevel% equ 0 (
    echo   [OK] All dependencies installed
) else (
    echo   [WARNING] Some dependencies may have failed. Try: pip install -r requirements.txt
)

REM --- Step 7: Initialize data directories and config ---
echo.
echo [STEP 7/%TOTAL_STEPS%] Initializing UniMentor...

if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\profiles" mkdir data\profiles
if not exist "knowledge_base" mkdir knowledge_base

if not exist "config\user_config.yaml" (
    copy config\default_config.yaml config\user_config.yaml >nul
    echo   [OK] Default configuration created
) else (
    echo   [OK] User configuration already exists
)

REM Check if profiles directory is empty
dir /b "data\profiles\*" >nul 2>&1
if %errorlevel% neq 0 (
    copy config\sample_profile.yaml data\profiles\default_profile.yaml >nul
    echo   [OK] Sample student profile created
) else (
    echo   [OK] Student profiles already exist
)

echo   [OK] Data directories initialized

REM --- Done ---
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo UniMentor is ready to use!
echo.
echo To start UniMentor, run:
echo.
echo   venv\Scripts\activate
echo   streamlit run app\main.py
echo.
echo Tips:
echo   - Drop your course PDFs, slides, and notes into the knowledge_base\ folder
echo   - Or use the drag-and-drop uploader in the web UI
echo   - Edit config\user_config.yaml to customize settings
echo   - Your data stays 100%% local on your machine
echo.
echo Happy studying!
echo.
pause
