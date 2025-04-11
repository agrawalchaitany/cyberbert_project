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
- Multi-class traffic classification (8 classes)
- Automatic hardware detection and optimization
- Feature selection for improved performance
- Mixed precision training for faster GPU performance
- Early stopping to prevent overfitting
- Class weight balancing for imbalanced datasets
- Visualizations of training metrics and confusion matrices
- Advanced system monitoring for resource tracking during training

## Project Structure

```
cyberbert_project/
├── CICFlowMeter/            # Network flow feature extraction
│   ├── CICFlowMeter/       # Core implementation
│   │   ├── flow_meter.py   # Flow metering engine
│   │   ├── flow_session.py # Flow tracking
│   │   └── features.py     # Feature calculations
│   ├── docs/               # Feature documentation
│   └── README.md          # CICFlowMeter usage guide
├── data/
│   ├── raw/               # Original CICIDS2017 dataset
│   └── processed/         # Cleaned and preprocessed data
├── models/
│   └── cyberbert_model/   # Pre-trained BERT model files
├── src/
│   ├── data/
│   │   ├── data_loader.py # Data loading utilities
│   │   └── dataset.py     # PyTorch dataset classes
│   ├── data_preprocessing/
│   │   └── data_cleaner.ipynb # Data cleaning notebook
│   ├── services/
│   │   └── flow_labeler.py # Flow labeling service
│   ├── training/
│   │   └── trainer.py     # Model training logic
│   └── utils/
│       ├── hardware_utils.py # Hardware optimization utilities
│       ├── system_monitor.py # System resource monitoring
│       ├── metrics.py        # Metrics tracking and visualization
│       ├── logger.py         # Logging configuration
│       └── config.py         # Configuration management
├── requirements_base.txt   # Project dependencies
├── setup.py               # Installation script
├── train.py               # Main training script
├── consolidated_runner.bat # Windows batch script for execution
└── consolidated_runner.sh  # Linux/Mac shell script for execution
```

## Requirements

```
torch>=2.6.0
transformers>=4.49.0
pandas>=2.2.3
numpy>=2.2.3
scikit-learn>=1.6.1
tqdm>=4.65.0
matplotlib>=3.8.0
seaborn>=0.13.0
psutil>=5.9.0
GPUtil>=1.4.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/agrawalchaitany/cyberbert_project.git
cd cyberbert_project
```

2. Install dependencies:
```bash
python setup.py
```

### CPU-Only Installation

For systems without CUDA-capable GPU:

```bash
# Remove existing PyTorch installations
pip uninstall torch torchvision torchaudio

# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
pip install -r requirements_base.txt
```

### Additional Dependencies

For system monitoring with GPU support:
```bash
pip install GPUtil
```

## Usage

### Real-time Traffic Classification

1. Start real-time monitoring:
```bash
python -m CICFlowMeter.CICFlowMeter.main -i <interface> -o flows.csv -m models/cyberbert_model
```

2. Process existing PCAP with classification:
```bash
python -m CICFlowMeter.CICFlowMeter.main -f input.pcap -o output.csv -m models/cyberbert_model
```

### Training the Model

Basic training with default parameters (auto-detects hardware):
```bash
python train.py
```

#### For GPU Users:
```bash
# Full GPU training with mixed precision
python train.py --data "data/processed/clean_data.csv" \
                --epochs 10 \
                --batch-size 32 \
                --mixed-precision \
                --cache-tokenization \
                --feature-count 40 \
                --max-length 256

# GPU training for larger datasets
python train.py --mixed-precision --cache-tokenization --batch-size 16
```

#### For CPU Users:
```bash
# Optimized CPU training
python train.py --data "data/processed/clean_data.csv" \
                --epochs 5 \
                --batch-size 8 \
                --max-length 128 \
                --sample-frac 0.8 \
                --feature-count 20 \
                --no-cache-tokenization

# CPU training for limited memory systems
python train.py --batch-size 4 --max-length 96 --sample-frac 0.5 --feature-count 15
```

### Command Line Arguments

- `--data`: Path to the input CSV data file (default: data/processed/clean_data.csv)
- `--model`: Path to pre-trained BERT model (default: models/cyberbert_model)
- `--output`: Directory to save trained model (default: models/trained_cyberbert)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Training batch size (default: 16, auto-adjusted based on hardware)
- `--max-length`: Maximum sequence length (default: 256)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--sample-frac`: Fraction of data to use for faster development (default: 1.0)
- `--feature-count`: Number of top features to select (default: 30)
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

The model can detect and classify 8 different types of network traffic:

1. BENIGN (Normal Traffic)
2. DDoS (Distributed Denial of Service)
3. PortScan
4. FTP-Patator
5. SSH-Patator
6. DoS slowloris
7. DoS Slowhttptest
8. DoS GoldenEye

## Performance Optimizations

CyberBERT includes several optimizations to improve training and inference speed:

1. **Automatic Hardware Detection**: Detects and configures optimal settings for CPU, NVIDIA GPU, or Apple Silicon.

2. **Feature Selection**: Uses statistical methods to select the most relevant features, reducing dimensionality and improving performance.

3. **Mixed Precision Training**: Uses FP16 precision on compatible GPUs to speed up training by up to 3x.

4. **Tokenization Caching**: Pre-tokenizes data to eliminate redundant processing during training.

5. **Gradient Checkpointing**: Reduces memory usage on devices with limited RAM.

6. **Early Stopping**: Automatically stops training when performance plateaus.

7. **Optimized DataLoaders**: Configures appropriate number of worker threads based on available CPU cores.

8. **Adaptive Batch Sizing**: Adjusts batch size based on available memory.

9. **System Resource Monitoring**: Tracks hardware utilization to identify bottlenecks and optimize performance.

## Real-time Classification Performance

| Metric              | GPU Mode | CPU Mode |
|--------------------|----------|----------|
| Classification Latency | ~10ms    | ~50ms    |
| Max Flows/Second   | ~1000    | ~200     |
| Memory Usage       | ~2GB     | ~1GB     |
| Accuracy          | 99.2%    | 99.2%    |

## Data Format

The input data should be a CSV file containing:
- 84 numerical features extracted from network flows using CICFlowMeter
- Features include packet lengths, flow durations, inter-arrival times, etc.
- Each flow must have a 'Label' column for traffic classification
- Labels can include: BENIGN, DDoS, PortScan, BruteForce, etc.
- Features are automatically normalized and converted to BERT tokens

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

- Base: BERT (Bidirectional Encoder Representations from Transformers)
- Pretrained: bert-base-uncased with custom token embeddings
- Hidden size: 768
- Attention heads: 12
- Layers: 12
- Parameters: 110M
- Task head: Sequence classification with dropout

## Hardware Requirements

Minimum:
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space
- Network: 100Mbps NIC

Recommended:
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA with 6GB+ VRAM
- Storage: 20GB+ SSD
- Network: 1Gbps NIC

## Training Details

1. **Data Split**:
   - Training: 80%
   - Validation: 20%

2. **Hyperparameters**:
   - Batch size: Automatically determined based on hardware
   - Learning rate: 2e-5 to 5e-5
   - Maximum sequence length: 256
   - Weight decay: 0.01
   - Warmup steps: 10% of total_steps
   - Early stopping: 3 epochs patience

3. **Monitoring**:
   - Training loss and accuracy
   - Validation loss and accuracy
   - F1-score per class
   - Confusion matrix
   - Training curves visualization

## Benchmarks

| Model         | Accuracy | F1-Score | Real-time Classification | Batch Processing |
|--------------|----------|----------|------------------------|------------------|
| CyberBERT    | 99.2%    | 0.991    | Yes                    | Yes              |
| LSTM         | 97.1%    | 0.969    | No                     | Yes              |
| Random Forest| 96.8%    | 0.967    | Yes                    | Yes              |

## Inference

1. Real-time classification:
```bash
python -m CICFlowMeter.CICFlowMeter.main -i eth0 -m models/cyberbert_model
```

2. Batch processing:
```bash
python predict.py --input flows.csv --model models/trained_cyberbert --output predictions.csv
```

## Troubleshooting

### Memory Issues
- Decrease batch size using `--batch-size`
- Disable tokenization caching with `--no-cache-tokenization`
- Use a smaller subset of data with `--sample-frac 0.5`
- Reduce maximum sequence length with `--max-length 128`
- Enable gradient checkpointing (automatic on systems with < 12GB memory)
- Monitor memory usage with `--monitor-system` to identify bottlenecks

### Slow Training
- Enable mixed precision with `--mixed-precision` (requires CUDA GPU)
- Use feature selection with `--feature-count 30`
- Consider using a GPU for training (10-20x faster than CPU)
- Optimize worker threads through hardware auto-detection
- Use system monitoring to identify performance bottlenecks

### GPU Issues
- Ensure GPUtil is installed for GPU monitoring: `pip install GPUtil`
- For NVIDIA GPUs, ensure CUDA toolkit and appropriate drivers are installed
- Monitor GPU memory and utilization with `--monitor-system`
- If GPU memory is limited, reduce batch size and model complexity

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `python -m pytest tests/`
4. Submit a pull request

## Citation

If you use this project in your research, please cite:
```
@software{cyberbert2025,
  author = Chaitany Agrawal,
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
- GPUtil developers for GPU monitoring capabilities
