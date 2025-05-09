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

# Check if .env file exists, create it if not
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cat > .env << EOF
# CyberBERT Environment Configuration

# Model to download
MODEL_NAME=distilbert-base-uncased

# Dataset URL (leave empty if no dataset to download)
DATASET_URL=

# Training parameters
EPOCHS=2
BATCH_SIZE=8
FEATURE_COUNT=10
MAX_LENGTH=96
SAMPLE_FRACTION=0.2

# CPU-specific parameters (used when no GPU is available)
CPU_EPOCHS=1
CPU_BATCH_SIZE=4
CPU_MAX_LENGTH=64
CPU_FEATURE_COUNT=5
CPU_SAMPLE_FRACTION=0.05
EOF
    echo ".env file created"
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
    
    # Check if requirements_base.txt exists
    if [ -f requirements_base.txt ]; then
        echo "Installing requirements from requirements_base.txt..."
        pip install -r requirements_base.txt
    else
        echo "WARNING: requirements_base.txt not found. Installing essential packages..."
        pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm python-dotenv
    fi
    
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
    
    # Check if MODEL_NAME is set
    if [ -z "$MODEL_NAME" ]; then
        echo "ERROR: MODEL_NAME not set in .env file."
        echo "Setting default model to distilbert-base-uncased"
        MODEL_NAME="distilbert-base-uncased"
    fi
    
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
        echo "You'll need to manually place dataset files in the data/processed/ directory."
        
        # Create data directories anyway
        mkdir -p data/processed
        
        # Create a small dummy dataset if none exists
        if [ ! -f data/processed/clean_data.csv ]; then
            echo "Creating a small dummy dataset for testing..."
            echo "feature1,feature2,feature3,label" > data/processed/clean_data.csv
            echo "1.2,3.4,5.6,normal" >> data/processed/clean_data.csv
            echo "7.8,9.0,1.2,attack" >> data/processed/clean_data.csv
            echo "3.3,4.4,5.5,normal" >> data/processed/clean_data.csv
            echo "Dummy dataset created at data/processed/clean_data.csv"
        fi
    fi
    
    if [ "$MODE" = "4" ]; then
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
        echo "All operations completed successfully."
        exit 0
    fi
fi

# Check if train.py exists before attempting training
if [ ! -f train.py ]; then
    echo "ERROR: train.py not found. Cannot proceed with training."
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    exit 1
fi

# Training section - only for modes 1 and 2
if [ "$MODE" = "1" ] || [ "$MODE" = "2" ]; then
    echo "Detecting hardware configuration..."

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
                        --max-length "$MAX_LENGTH" \
                        --sample-frac "${SAMPLE_FRACTION:-0.5}"
    else
        echo "No GPU detected. Using CPU-optimized settings..."
        
        # Run with CPU optimized settings from .env
        python train.py --data "data/processed/clean_data.csv" \
                        --epochs "$CPU_EPOCHS" \
                        --batch-size "$CPU_BATCH_SIZE" \
                        --max-length "$CPU_MAX_LENGTH" \
                        --sample-frac "${CPU_SAMPLE_FRACTION:-0.05}" \
                        --feature-count "$CPU_FEATURE_COUNT" \
                        --cache-tokenization
    fi

    echo "Training completed!"
fi

# Deactivate the virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Environment deactivated."
fi

echo "All operations completed successfully."