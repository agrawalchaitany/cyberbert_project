@echo off
:: consolidated_runner.bat - Combined script for environment setup and CyberBERT training
echo ===============================================
echo CyberBERT Environment Setup and Training Script
echo ===============================================

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

:: Setup mode vs Training mode selection
set SETUP_MODE=1
set /p CHOICE=Do you want to (1) Setup environment and train or (2) Just train? [1/2]: 
if "%CHOICE%"=="2" (
    set SETUP_MODE=0
    echo Skipping environment setup, proceeding to training...
) else (
    echo Full setup and training selected...
)

:: Environment setup section
if %SETUP_MODE%==1 (
    :: Check if virtual environment exists
    if not exist .venv (
        echo Creating virtual environment...
        python -m venv .venv
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to create virtual environment. 
            echo Please install venv module with: python -m pip install virtualenv
            pause
            exit /b 1
        )
    )

    :: Activate virtual environment
    echo Activating environment...
    call .venv\Scripts\activate.bat

    :: Check if key dependencies are installed
    python -c "import torch" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Installing PyTorch and dependencies...
        python -c "import platform; is_cuda = False; print('Installing CPU version' if not is_cuda else 'Installing CUDA version')"
        python -m pip install -r requirements_base.txt
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to install dependencies.
            call .venv\Scripts\deactivate.bat
            pause
            exit /b 1
        )
    )
    echo Environment setup completed successfully!
) else (
    :: If not doing setup, check if we should activate an existing environment
    if exist .venv (
        echo Activating existing virtual environment...
        call .venv\Scripts\activate.bat
    ) else (
        echo No virtual environment detected. Continuing without environment activation...
    )
)

:: Training section - same for both modes
echo Detecting hardware configuration...

:: Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())" > gpu_check.tmp
set /p GPU_AVAILABLE=<gpu_check.tmp
del gpu_check.tmp

if "%GPU_AVAILABLE%"=="True" (
    echo GPU detected! Using GPU-optimized settings...
    
    :: Run with GPU optimized settings
    python train.py --data "data/processed/clean_data.csv" ^
                    --epochs 10 ^
                    --batch-size 32 ^
                    --mixed-precision ^
                    --cache-tokenization ^
                    --feature-count 40 ^
                    --max-length 256
) else (
    echo No GPU detected. Using CPU-optimized settings...
    
    :: Run with CPU optimized settings
    python train.py --data "data/processed/clean_data.csv" ^
                    --epochs 3 ^
                    --batch-size 8 ^
                    --max-length 128 ^
                    --sample-frac 0.8 ^
                    --feature-count 20
)

echo Training completed!

:: Deactivate the virtual environment if it was activated
if defined VIRTUAL_ENV (
    call .venv\Scripts\deactivate.bat
    echo Environment deactivated.
)

echo All operations completed successfully.
pause