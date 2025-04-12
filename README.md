# CyberBERT: Network Traffic Classification Using BERT

A deep learning model for network traffic classification that leverages BERT architecture. CyberBERT processes network flow data to accurately classify various types of network traffic in real-time, including benign and malicious patterns.

## Features

- Deep learning-based network traffic classification
- Real-time traffic analysis and threat detection
- Processing of 84 network flow features
- Conversion of numerical flow features to BERT-compatible text format
- Command-line interface for flexible configuration
- Support for both CPU and GPU training/inference
- Automated data preprocessing and cleaning
- Training progress monitoring and model checkpointing
- Multi-class traffic classification (6 classes)
- Automatic hardware detection and optimization
- Feature selection for improved performance
- Mixed precision training for faster GPU performance
- Early stopping to prevent overfitting
- Class weight balancing for imbalanced datasets
- Visualizations of training metrics and confusion matrices
- Advanced system monitoring for resource tracking during training
- Consolidated runner scripts for Windows and Linux/Mac
- Environment variable configuration via .env file

## Project Structure

```
cyberbert_project/
├── CICFlowMeter/            # Network flow feature extraction
│   ├── CICFlowMeter/        # Core implementation
│   │   ├── flow_meter.py    # Flow metering engine
│   │   ├── flow_session.py  # Flow tracking
│   │   ├── features.py      # Feature calculations
│   │   ├── main.py          # Entry point for flow meter
│   │   └── utils.py         # Utility functions
│   ├── docs/                # Feature documentation
│   ├── requirements.txt     # CICFlowMeter dependencies
│   ├── setup.py             # Setup script for CICFlowMeter
│   └── README.md            # CICFlowMeter usage guide
├── data/
│   └── processed/           # Cleaned and preprocessed data
│       └── clean_data.csv   # Processed flow data
├── logs/                    # Training logs
│   └── cyberbert_*.log      # Log files with timestamps
├── models/
│   ├── cyberbert_model/     # Pre-trained BERT model files
│   │   ├── config.json      # Model configuration
│   │   ├── model.safetensors # Model weights
│   │   ├── tokenizer.json   # Tokenizer data
│   │   ├── vocab.txt        # Vocabulary file
│   │   └── ...              # Other model files
│   └── trained_cyberbert/   # Fine-tuned model
│       ├── best_model/      # Best checkpoint during training
│       ├── interrupted_checkpoint/ # Checkpoint from interruption
│       ├── metrics/         # Training metrics
│       ├── label_mapping.json # Class label mappings
│       ├── selected_features.txt # Selected features
│       └── training_*.json  # Training configurations
├── src/
│   ├── data/
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── dataset.py       # PyTorch dataset classes
│   ├── data_preprocessing/
│   │   └── data_cleaner.ipynb # Data cleaning notebook
│   ├── download_model/
│   │   └── p_c_d_check.py   # Model download utility
│   ├── services/
│   │   └── flow_labeler.py  # Flow labeling service
│   ├── training/
│   │   └── trainer.py       # Model training logic
│   └── utils/
│       ├── config.py        # Configuration management
│       ├── logger.py        # Logging configuration
│       ├── metrics.py       # Metrics tracking
│       ├── model_registry.py # Model registration
│       └── system_monitor.py # System resource monitoring
├── consolidated_runner.bat  # Windows batch script
├── consolidated_runner.sh   # Linux/Mac shell script
├── flows.csv                # Sample flow data
├── flows.db                 # Flow database
├── requirements_base.txt    # Project dependencies
├── train.py                 # Main training script
└── README.md                # This file
```

## Requirements

```
transformers>=4.49.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
pandas>=2.2.3
numpy>=2.2.3
scikit-learn>=1.6.1
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.65.0
psutil>=5.9.0
scapy>=2.4.5
python-dotenv>=1.0.0
huggingface-hub>=0.29.1
```

Additional dependencies are listed in the requirements_base.txt file.

## Installation and Usage

The project uses consolidated runner scripts that handle environment setup, model download, dataset download, and training in a single unified interface.

### Using the Consolidated Runner Scripts

#### Windows:
```batch
.\consolidated_runner.bat
```

#### Linux/Mac:
```bash
chmod +x consolidated_runner.sh
./consolidated_runner.sh
```

### Options in Consolidated Runner

The consolidated runner provides four operation modes:

1. **Setup environment and train**: Creates a virtual environment, installs dependencies, downloads model, downloads dataset (if URL provided), and runs training
2. **Just train**: Skips environment setup and proceeds directly to training
3. **Download model only**: Only downloads the pre-trained model specified in the .env file
4. **Download dataset only**: Only downloads the dataset from the URL specified in the .env file

### Configuration via .env File

The project uses a .env file for configuration. If the file doesn't exist, the runner scripts will automatically create it with default values:

```
# Model to download
MODEL_NAME=distilbert-base-uncased

# Dataset URL (leave empty if no dataset to download)
DATASET_URL=

# Training parameters
EPOCHS=3
BATCH_SIZE=8
FEATURE_COUNT=78
MAX_LENGTH=64
SAMPLE_FRACTION=0.5

# CPU-specific parameters (used when no GPU is available)
CPU_EPOCHS=3
CPU_BATCH_SIZE=1
CPU_MAX_LENGTH=64
CPU_FEATURE_COUNT=78
CPU_SAMPLE_FRACTION=0.5
```

Edit this file to customize the model, dataset URL, and training parameters before running the consolidated scripts.

### Manual Usage (Alternative)

#### For GPU Users:
```bash
# Full GPU training with mixed precision
python train.py --data "data/processed/clean_data.csv" \
                --epochs 5 \
                --batch-size 32 \
                --mixed-precision \
                --cache-tokenization \
                --feature-count 78 \
                --max-length 128
```

#### For CPU Users:
```bash
# Optimized CPU training
python train.py --data "data/processed/clean_data.csv" \
                --epochs 3 \
                --batch-size 1 \
                --max-length 64 \
                --sample-frac 0.5 \
                --feature-count 78 \
                --no-cache-tokenization
```

### Command Line Arguments

- `--data`: Path to the input CSV data file (default: data/processed/clean_data.csv)
- `--model`: Path to pre-trained model (default: models/cyberbert_model)
- `--output`: Directory to save trained model (default: models/trained_cyberbert)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8, auto-adjusted based on hardware)
- `--max-length`: Maximum sequence length (default: 64)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--sample-frac`: Fraction of data to use for faster development (default: 0.5)
- `--feature-count`: Number of top features to select (default: 78)
- `--no-feature-selection`: Disable feature selection (by default, feature selection is enabled)
- `--mixed-precision`: Enable mixed precision training for compatible GPUs
- `--cache-tokenization`: Cache tokenized data for faster training (uses more memory)
- `--early-stopping`: Early stopping patience in epochs (default: 3)
- `--eval-steps`: Evaluate on validation set every N steps (default: 100, 0 to disable)
- `--monitor-system`: Enable detailed system resource monitoring (default: True)
- `--monitor-interval`: Interval in seconds for system monitoring (default: 5.0)

## System Monitoring

CyberBERT includes a comprehensive system monitoring tool that tracks:

- CPU usage (overall and per-core)
- Memory usage (RAM and swap)
- GPU utilization and memory (when available)
- Disk usage and I/O statistics
- Process-specific resource utilization

To view system metrics during training:
```bash
python train.py --monitor-system --monitor-interval 2.0
```

To disable system monitoring (for minimal overhead):
```bash
python train.py --no-monitor-system
```

System metrics are logged to the console and saved to the output directory for later analysis.

## Supported Traffic Classes

The model can detect and classify 6 different types of network traffic:

1. BENIGN (Normal Traffic)
2. DoS GoldenEye
3. DoS Slowhttptest
4. Portscan
5. DDoS
6. FTP-Patator/SSH-Patator

## Real-time Classification Performance

| Metric              | GPU Mode | CPU Mode |
|--------------------|----------|----------|
| Classification Latency | ~5ms     | ~40ms    |
| Max Flows/Second   | ~1500    | ~250     |
| Memory Usage       | ~1.5GB   | ~1GB     |
| Accuracy          | 96.0%    | 96.0%    |
| F1 Score          | 0.96     | 0.96     |

The real-time classification can be performed using:

```bash
python -m CICFlowMeter.CICFlowMeter.main -i eth0 -r -m models/trained_cyberbert/best_model
```

## Data Format

The input data should be a CSV file containing:
- 84 numerical features extracted from network flows using CICFlowMeter
- Features include packet lengths, flow durations, inter-arrival times, etc.
- Each flow must have a 'Label' column for traffic classification
- Features are automatically normalized and converted to model-compatible tokens

## Data Preprocessing

1. Feature extraction using CICFlowMeter:
```bash
python -m CICFlowMeter.CICFlowMeter.main -f pcap_file.pcap -o features.csv
```

2. Data cleaning and preprocessing:
```bash
jupyter notebook src/data_preprocessing/data_cleaner.ipynb
```

## Model Architecture

- Base: DistilBERT (Distilled version of BERT)
- Pretrained: distilbert-base-uncased with custom token embeddings
- Hidden size: 768
- Attention heads: 12
- Layers: 6
- Parameters: ~66M (40% smaller than BERT-base)
- Task head: Sequence classification with dropout
- Output classes: 6 traffic types

## Hardware Requirements

Minimum:
- CPU: 2 cores
- RAM: 2GB
- Storage: 5GB free space

Recommended:
- CPU: 4+ cores
- RAM: 8GB+
- GPU: NVIDIA with 4GB+ VRAM
- Storage: 10GB+ SSD

## Training Details

1. **Data Split**:
   - Training: 80% (12,758 samples)
   - Validation: 20% (3,190 samples)

2. **Hyperparameters**:
   - Batch size: 1 (CPU), 8+ (GPU)
   - Learning rate: 2e-5
   - Maximum sequence length: 64
   - Weight decay: 0.01
   - Warmup steps: 500
   - Early stopping: 3 epochs patience
   - Feature selection: True (78 features)
   - Sample fraction: 0.5
   - Evaluation steps: 100

3. **Training Progression**:
   - Initial accuracy: ~24%
   - Final accuracy: ~97%
   - Final F1 score: 0.96

4. **Training Duration**:
   - CPU: ~6-8 hours for 3 epochs
   - GPU: ~1 hour for 3 epochs

## Benchmarks

| Model         | Accuracy | F1-Score | Training Time (CPU) | Training Time (GPU) |
|--------------|----------|----------|---------------------|---------------------|
| CyberBERT    | 96.0%    | 0.96     | ~6-8 hours          | ~1 hour             |
| DistilBERT   | 94.5%    | 0.94     | ~4-6 hours          | ~45 mins            |
| Random Forest| 92.3%    | 0.92     | ~30 mins            | N/A                 |
| KNN          | 89.0%    | 0.88     | ~15 mins            | N/A                 |

## Inference

1. Real-time classification:
```bash
python -m CICFlowMeter.CICFlowMeter.main -i eth0 -r -m models/trained_cyberbert/best_model
```

2. Batch processing:
```bash
python -m CICFlowMeter.CICFlowMeter.main -f capture.pcap -m models/trained_cyberbert/best_model -o predictions.csv
```

## Troubleshooting

### Memory Issues
- Decrease batch size in the .env file by reducing BATCH_SIZE and CPU_BATCH_SIZE values
- Reduce maximum sequence length by adjusting MAX_LENGTH and CPU_MAX_LENGTH in the .env file
- Reduce sample fraction by setting SAMPLE_FRACTION to a lower value (e.g., 0.1)
- Monitor memory usage with the system monitor to identify bottlenecks

### Slow Training
- Enable mixed precision through the consolidated runner (automatic on GPU systems)
- Consider using a GPU for training (10-20x faster than CPU)
- Reduce feature count by adjusting FEATURE_COUNT in the .env file
- Use system monitoring to identify performance bottlenecks
- For CPU training, reduce the sample fraction significantly (5-10% is often sufficient for testing)
- Switch to DistilBERT models which are about 40% faster than full BERT models
- Reduce the max sequence length to 32-64 for CPU training to speed up processing

### Model Architecture Issues
- If you encounter warnings about "parameter shape mismatch" or "weights not being initialized", make sure the model type in your .env file matches the actual model architecture (BERT vs DistilBERT)
- The system now automatically detects model architecture to prevent mismatch warnings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `python -m pytest tests/`
4. Submit a pull request

## Citation

If you use this project in your research, please cite:
```
@software{cyberbert2025,
  author = {Chaitany Agrawal},
  title = {CyberBERT: Network Traffic Classification Using BERT},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/agrawalchaitany/cyberbert_project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CICIDS2017 dataset providers
- Hugging Face Transformers team
- CICFlowMeter developers
- Network security community
