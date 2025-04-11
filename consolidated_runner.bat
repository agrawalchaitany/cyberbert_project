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

:: Check if .env file exists, create it if not
if not exist .env (
    echo Creating default .env file...
    (
        echo MODEL_NAME=bert-base-uncased
        echo DATASET_URL=
        echo EPOCHS=5
        echo BATCH_SIZE=32
        echo FEATURE_COUNT=20
        echo MAX_LENGTH=128
        echo CPU_EPOCHS=3
        echo CPU_BATCH_SIZE=16
        echo CPU_MAX_LENGTH=128
        echo CPU_FEATURE_COUNT=20
    ) > .env
    echo .env file created
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
if "%MODE%"=="1" (
    goto SETUP
) else if "%MODE%"=="3" (
    goto SETUP
) else if "%MODE%"=="4" (
    goto SETUP
) else if "%MODE%"=="2" (
    goto SKIP_SETUP
)

:SETUP
:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    
    :: First try with virtualenv as it's more reliable
    python -m pip install virtualenv >nul 2>&1
    python -m virtualenv venv
    
    if %ERRORLEVEL% neq 0 (
        :: Try with venv only if virtualenv fails
        echo Virtualenv failed, trying with venv module...
        python -m venv venv
        
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to create virtual environment.
            echo Please make sure you have virtualenv or venv installed.
            echo Try: python -m pip install virtualenv
            pause
            exit /b 1
        )
    )
) else (
    echo Virtual environment already exists, skipping creation.
)

:: Activate virtual environment
echo Activating environment...
call venv\Scripts\activate.bat

:: Check if activation was successful
if not defined VIRTUAL_ENV (
    echo ERROR: Failed to activate virtual environment.
    echo Try running the script again or manually activate the environment:
    echo call venv\Scripts\activate.bat
    pause
    exit /b 1
)

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

:: Check if requirements_base.txt exists
if exist requirements_base.txt (
    echo Installing requirements from requirements_base.txt...
    python -m pip install -r requirements_base.txt
) else (
    echo WARNING: requirements_base.txt not found. Installing essential packages...
    python -m pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm python-dotenv
)

echo Environment setup completed successfully!
goto ENV_SETUP_DONE

:SKIP_SETUP
:: If just training, check if we should activate an existing environment
if exist venv (
    echo Activating existing virtual environment...
    call venv\Scripts\activate.bat
    
    :: Check if activation was successful
    if not defined VIRTUAL_ENV (
        echo WARNING: Failed to activate virtual environment.
        echo Continuing without virtual environment, but this may cause issues...
    )
) else (
    echo No virtual environment detected. Continuing without environment activation...
)

:ENV_SETUP_DONE

:: Model download section
if "%MODE%"=="1" (
    goto MODEL_DOWNLOAD
) else if "%MODE%"=="3" (
    goto MODEL_DOWNLOAD
) else (
    goto SKIP_MODEL_DOWNLOAD
)

:MODEL_DOWNLOAD
echo Checking models directory...
if not exist models mkdir models
if not exist models\cyberbert_model mkdir models\cyberbert_model

if "%MODEL_NAME%"=="" (
    echo ERROR: MODEL_NAME not set in .env file.
    echo Setting default model to bert-base-uncased
    set MODEL_NAME=bert-base-uncased
)

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

:SKIP_MODEL_DOWNLOAD

:: Dataset download section
if "%MODE%"=="1" (
    goto DATASET_DOWNLOAD
) else if "%MODE%"=="4" (
    goto DATASET_DOWNLOAD
) else (
    goto SKIP_DATASET_DOWNLOAD
)

:DATASET_DOWNLOAD
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
    echo You'll need to manually place dataset files in the data/processed/ directory.
    
    :: Create data directories anyway
    if not exist data mkdir data
    if not exist data\processed mkdir data\processed
    
    :: Create a small dummy dataset if none exists
    if not exist data\processed\clean_data.csv (
        echo Creating a small dummy dataset for testing...
        echo "feature1,feature2,feature3,label" > data\processed\clean_data.csv
        echo "1.2,3.4,5.6,normal" >> data\processed\clean_data.csv
        echo "7.8,9.0,1.2,attack" >> data\processed\clean_data.csv
        echo "3.3,4.4,5.5,normal" >> data\processed\clean_data.csv
        echo Dummy dataset created at data\processed\clean_data.csv
    )
)

if "%MODE%"=="4" (
    if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
    echo All operations completed successfully.
    pause
    exit /b 0
)

:SKIP_DATASET_DOWNLOAD

:: Check if train.py exists before attempting training
if not exist train.py (
    echo ERROR: train.py not found. Cannot proceed with training.
    if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
    pause
    exit /b 1
)

:: Check if src directory exists
if not exist src (
    echo ERROR: src directory not found. Cannot proceed with training.
    echo The project appears to be missing essential source files.
    if defined VIRTUAL_ENV call venv\Scripts\deactivate.bat
    pause
    exit /b 1
)

:: Training section - only for modes 1 and 2
if "%MODE%"=="1" (
    goto TRAINING
) else if "%MODE%"=="2" (
    goto TRAINING
) else (
    goto SKIP_TRAINING
)

:TRAINING
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

:SKIP_TRAINING

:: Deactivate the virtual environment if it was activated
if defined VIRTUAL_ENV (
    call venv\Scripts\deactivate.bat
    echo Environment deactivated.
)

echo All operations completed successfully.
pause