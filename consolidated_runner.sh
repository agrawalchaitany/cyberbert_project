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

# Setup mode vs Training mode selection
SETUP_MODE=1
read -p "Do you want to (1) Setup environment and train or (2) Just train? [1/2]: " CHOICE
if [ "$CHOICE" = "2" ]; then
    SETUP_MODE=0
    echo "Skipping environment setup, proceeding to training..."
else
    echo "Full setup and training selected..."
fi

# Environment setup section
if [ $SETUP_MODE -eq 1 ]; then
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create virtual environment."
            echo "Please install venv module with: pip install virtualenv"
            exit 1
        fi
    fi

    # Activate virtual environment
    echo "Activating environment..."
    source .venv/bin/activate

    # Check if key dependencies are installed
    python -c "import torch" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "Installing PyTorch and dependencies..."
        pip install -r requirements_base.txt
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install dependencies."
            deactivate
            exit 1
        fi
    fi
    echo "Environment setup completed successfully!"
else
    # If not doing setup, check if we should activate an existing environment
    if [ -d ".venv" ]; then
        echo "Activating existing virtual environment..."
        source .venv/bin/activate
    else
        echo "No virtual environment detected. Continuing without environment activation..."
    fi
fi

# Training section - same for both modes
echo "Checking for GPU availability..."

# Check if GPU is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "GPU detected! Using GPU-optimized settings..."
    
    # Run with GPU optimized settings
    python train.py --data "data/processed/clean_data.csv" \
                    --epochs 10 \
                    --batch-size 32 \
                    --mixed-precision \
                    --cache-tokenization \
                    --feature-count 40 \
                    --max-length 256
else
    echo "No GPU detected. Using CPU-optimized settings..."
    
    # Run with CPU optimized settings
    python train.py --data "data/processed/clean_data.csv" \
                    --epochs 5 \
                    --batch-size 8 \
                    --max-length 128 \
                    --sample-frac 0.8 \
                    --feature-count 20
fi

echo "Training completed!"

# Deactivate the virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Environment deactivated."
fi

echo "All operations completed successfully."