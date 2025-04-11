@echo off
:: train_cyberbert.bat - Helper script to run CyberBERT training with optimal parameters on Windows

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
                    --epochs 5 ^
                    --batch-size 8 ^
                    --max-length 128 ^
                    --sample-frac 0.8 ^
                    --feature-count 20
)

echo Training completed!
pause