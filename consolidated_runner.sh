#!/bin/bash
# consolidated_runner.sh - Combined script for environment setup and CyberBERT training
echo "==============================================="
echo "CyberBERT Environment Setup and Training Script"
echo "==============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Load environment variables from .env file
echo "Loading configuration from .env file..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Setup mode vs Training mode vs Download model mode selection
echo "Choose an operation:"
echo "1: Setup environment and train"
echo "2: Just train"
echo "3: Download model only"
echo "4: Download dataset only"
read -p "Enter your choice [1-4]: " MODE

if [ "$MODE" = "2" ]; then
    echo "Skipping environment setup, proceeding to training..."
elif [ "$MODE" = "3" ]; then
    echo "Proceeding to model download only..."
elif [ "$MODE" = "4" ]; then
    echo "Proceeding to dataset download only..."
else
    echo "Full setup and training selected..."
fi

# Environment setup section for all modes except "Just train"
if [ "$MODE" = "1" ] || [ "$MODE" = "3" ] || [ "$MODE" = "4" ]; then
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create virtual environment."
            echo "Please install venv module with: pip install virtualenv"
            exit 1
        fi
    fi

    # Activate virtual environment
    echo "Activating environment..."
    source venv/bin/activate

    # Install and upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install psutil
    echo "Installing psutil..."
    pip install psutil
    
    # Remove any existing PyTorch installations
    echo "Removing existing PyTorch installations if any..."
    pip uninstall -y torch torchvision torchaudio
    
    # Check CUDA availability
    echo "Checking for CUDA availability..."
    if [ "$(uname)" = "Darwin" ]; then
        # macOS - no CUDA support
        CUDA_STATUS="CPU"
    elif [ "$(uname)" = "Linux" ]; then
        # Linux - check for libcuda.so
        if ldconfig -p | grep -q libcuda.so; then
            CUDA_STATUS="CUDA"
        else
            CUDA_STATUS="CPU"
        fi
    else
        # Fallback - assume no CUDA
        CUDA_STATUS="CPU"
    fi
    
    # Install PyTorch based on CUDA availability
    if [ "$CUDA_STATUS" = "CUDA" ]; then
        echo "CUDA detected, installing GPU version of PyTorch..."
        pip install torch torchvision torchaudio
    else
        echo "No CUDA detected, installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install remaining requirements
    echo "Installing remaining requirements..."
    pip install -r requirements_base.txt
    
    echo "Environment setup completed successfully!"
elif [ "$MODE" = "2" ]; then
    # If just training, check if we should activate an existing environment
    if [ -d "venv" ]; then
        echo "Activating existing virtual environment..."
        source venv/bin/activate
    else
        echo "No virtual environment detected. Continuing without environment activation..."
    fi
fi

# Model download section
if [ "$MODE" = "1" ] || [ "$MODE" = "3" ]; then
    echo "Checking models directory..."
    mkdir -p models/cyberbert_model
    
    echo "Downloading model: $MODEL_NAME"
    python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME'); model = AutoModel.from_pretrained('$MODEL_NAME'); model.save_pretrained('./models/cyberbert_model'); tokenizer.save_pretrained('./models/cyberbert_model'); print('Model downloaded and saved in ./models/cyberbert_model/')"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download model."
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
        exit 1
    fi
    
    echo "Model download completed successfully!"
    
    if [ "$MODE" = "3" ]; then
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
        echo "All operations completed successfully."
        exit 0
    fi
fi

# Dataset download section
if [ "$MODE" = "1" ] || [ "$MODE" = "4" ]; then
    if [ ! -z "$DATASET_URL" ]; then
        echo "Checking data directories..."
        mkdir -p data/processed
        
        echo "Downloading dataset from: $DATASET_URL"
        python -c "import urllib.request; import os; print('Downloading dataset...'); urllib.request.urlretrieve('$DATASET_URL', './data/processed/clean_data.csv'); print('Dataset downloaded to ./data/processed/clean_data.csv')"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to download dataset."
            if [ -n "$VIRTUAL_ENV" ]; then
                deactivate
            fi
            exit 1
        fi
        
        echo "Dataset download completed successfully!"
    else
        echo "No dataset URL provided in .env file. Skipping dataset download."
    fi
    
    if [ "$MODE" = "4" ]; then
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
        echo "All operations completed successfully."
        exit 0
    fi
fi

# Training section - only for modes 1 and 2
if [ "$MODE" = "1" ] || [ "$MODE" = "2" ]; then
    echo "Checking for GPU availability..."

    # Check if GPU is available with PyTorch
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo "GPU detected! Using GPU-optimized settings..."
        
        # Run with GPU optimized settings from .env
        python train.py --data "data/processed/clean_data.csv" \
                        --epochs "$EPOCHS" \
                        --batch-size "$BATCH_SIZE" \
                        --mixed-precision \
                        --cache-tokenization \
                        --feature-count "$FEATURE_COUNT" \
                        --max-length "$MAX_LENGTH"
    else
        echo "No GPU detected. Using CPU-optimized settings..."
        
        # Run with CPU optimized settings from .env
        python train.py --data "data/processed/clean_data.csv" \
                        --epochs "$CPU_EPOCHS" \
                        --batch-size "$CPU_BATCH_SIZE" \
                        --max-length "$CPU_MAX_LENGTH" \
                        --sample-frac 0.8 \
                        --feature-count "$CPU_FEATURE_COUNT"
    fi

    echo "Training completed!"
fi

# Deactivate the virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Environment deactivated."
fi

echo "All operations completed successfully."