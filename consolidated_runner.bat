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

:: Load environment variables from .env file
echo Loading configuration from .env file...
for /F "tokens=*" %%A in (.env) do (
    set %%A
)

:: Setup mode vs Training mode vs Download model mode selection
set MODE=1
echo Choose an operation:
echo 1: Setup environment and train
echo 2: Just train
echo 3: Download model only
echo 4: Download dataset only
set /p MODE=Enter your choice [1-4]: 

if "%MODE%"=="2" (
    echo Skipping environment setup, proceeding to training...
) else if "%MODE%"=="3" (
    echo Proceeding to model download only...
) else if "%MODE%"=="4" (
    echo Proceeding to dataset download only...
) else (
    echo Full setup and training selected...
)

:: Environment setup section for all modes except "Just train"
if "%MODE%"=="1" || "%MODE%"=="3" || "%MODE%"=="4" (
    :: Check if virtual environment exists
    if not exist venv (
        echo Creating virtual environment...
        python -m venv venv
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to create virtual environment. 
            echo Please install venv module with: python -m pip install virtualenv
            pause
            exit /b 1
        )
    )

    :: Activate virtual environment
    echo Activating environment...
    call venv\Scripts\activate.bat

    :: Install and upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Install psutil
    echo Installing psutil...
    python -m pip install psutil
    
    :: Remove any existing PyTorch installations
    echo Removing existing PyTorch installations if any...
    python -m pip uninstall -y torch torchvision torchaudio
    
    :: Check CUDA availability
    echo Checking for CUDA availability...
    python -c "import ctypes; has_cuda = False; try: ctypes.CDLL('nvcuda.dll'); has_cuda = True; except: pass; print('CUDA' if has_cuda else 'CPU')" > cuda_check.tmp
    set /p CUDA_STATUS=<cuda_check.tmp
    del cuda_check.tmp
    
    :: Install PyTorch based on CUDA availability
    if "%CUDA_STATUS%"=="CUDA" (
        echo CUDA detected, installing GPU version of PyTorch...
        python -m pip install torch torchvision torchaudio
    ) else (
        echo No CUDA detected, installing CPU-only PyTorch...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
    
    :: Install remaining requirements
    echo Installing remaining requirements...
    python -m pip install -r requirements_base.txt
    
    echo Environment setup completed successfully!
) else if "%MODE%"=="2" (
    :: If just training, check if we should activate an existing environment
    if exist venv (
        echo Activating existing virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo No virtual environment detected. Continuing without environment activation...
    )
)

:: Model download section
if "%MODE%"=="1" || "%MODE%"=="3" (
    echo Checking models directory...
    if not exist models mkdir models
    if not exist models\cyberbert_model mkdir models\cyberbert_model
    
    echo Downloading model: %MODEL_NAME%
    python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('%MODEL_NAME%'); model = AutoModel.from_pretrained('%MODEL_NAME%'); model.save_pretrained('./models/cyberbert_model'); tokenizer.save_pretrained('./models/cyberbert_model'); print('Model downloaded and saved in ./models/cyberbert_model/')"
    
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to download model.
        if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
        pause
        exit /b 1
    )
    
    echo Model download completed successfully!
    
    if "%MODE%"=="3" (
        if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
        echo All operations completed successfully.
        pause
        exit /b 0
    )
)

:: Dataset download section
if "%MODE%"=="1" || "%MODE%"=="4" (
    if not "%DATASET_URL%"=="" (
        echo Checking data directories...
        if not exist data mkdir data
        if not exist data\processed mkdir data\processed
        
        echo Downloading dataset from: %DATASET_URL%
        python -c "import urllib.request; import os; print('Downloading dataset...'); urllib.request.urlretrieve('%DATASET_URL%', './data/processed/clean_data.csv'); print('Dataset downloaded to ./data/processed/clean_data.csv')"
        
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to download dataset.
            if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
            pause
            exit /b 1
        )
        
        echo Dataset download completed successfully!
    ) else (
        echo No dataset URL provided in .env file. Skipping dataset download.
    )
    
    if "%MODE%"=="4" (
        if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
        echo All operations completed successfully.
        pause
        exit /b 0
    )
)

:: Training section - only for modes 1 and 2
if "%MODE%"=="1" || "%MODE%"=="2" (
    echo Detecting hardware configuration...

    :: Check if GPU is available
    python -c "import torch; print(torch.cuda.is_available())" > gpu_check.tmp
    set /p GPU_AVAILABLE=<gpu_check.tmp
    del gpu_check.tmp

    if "%GPU_AVAILABLE%"=="True" (
        echo GPU detected! Using GPU-optimized settings...
        
        :: Run with GPU optimized settings from .env
        python train.py --data "data/processed/clean_data.csv" ^
                        --epochs %EPOCHS% ^
                        --batch-size %BATCH_SIZE% ^
                        --mixed-precision ^
                        --cache-tokenization ^
                        --feature-count %FEATURE_COUNT% ^
                        --max-length %MAX_LENGTH%
    ) else (
        echo No GPU detected. Using CPU-optimized settings...
        
        :: Run with CPU optimized settings from .env
        python train.py --data "data/processed/clean_data.csv" ^
                        --epochs %CPU_EPOCHS% ^
                        --batch-size %CPU_BATCH_SIZE% ^
                        --max-length %CPU_MAX_LENGTH% ^
                        --sample-frac 0.8 ^
                        --feature-count %CPU_FEATURE_COUNT%
    )

    echo Training completed!
)

:: Deactivate the virtual environment if it was activated
if defined VIRTUAL_ENV (
    call venv\Scripts\deactivate.bat
    echo Environment deactivated.
)

echo All operations completed successfully.
pause