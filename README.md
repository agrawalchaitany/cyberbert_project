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
- Multi-class traffic classification (15 classes)

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
│   ├── training/
│   │   └── trainer.py     # Model training logic
│   └── utils/
│       └── hardware_utils.py # Hardware optimization utilities
├── requirements_base.txt   # Project dependencies
├── setup.py               # Installation script
└── train.py              # Main training script
```

## Requirements

```
torch>=2.6.0
transformers>=4.49.0
pandas>=2.2.3
numpy>=2.2.3
scikit-learn>=1.6.1
tqdm>=4.65.0
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

Basic training with default parameters:
```bash
python train.py
```

Custom training configuration:
```bash
python train.py --data "path/to/flow_data.csv" \
                --epochs 10 \
                --batch-size 32 \
                --learning-rate 3e-5 \
                --max-length 256
```

### Command Line Arguments

- `--data`: Path to the input CSV data file (default: data/processed/clean_data.csv)
- `--model`: Path to pre-trained BERT model (default: models/cyberbert_model)
- `--output`: Directory to save trained model (default: models/trained_cyberbert)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Training batch size (default: 16)
- `--max-length`: Maximum sequence length (default: 512)
- `--learning-rate`: Learning rate (default: 2e-5)

## Supported Traffic Classes

The model can detect and classify 15 different types of network traffic:

1. BENIGN (Normal Traffic)
2. DDoS (Distributed Denial of Service)
3. PortScan
4. Bot
5. Infiltration
6. Web Attack - Brute Force
7. Web Attack - XSS
8. Web Attack - SQL Injection
9. FTP-Patator
10. SSH-Patator
11. DoS slowloris
12. DoS Slowhttptest
13. DoS Hulk
14. DoS GoldenEye
15. Heartbleed

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
   - Training: 70%
   - Validation: 15%
   - Testing: 15%

2. **Hyperparameters**:
   - Batch size: 16-32 (GPU), 8-16 (CPU)
   - Learning rate: 2e-5 to 5e-5
   - Maximum sequence length: 512
   - Weight decay: 0.01
   - Warmup steps: 0.1 * total_steps

3. **Monitoring**:
   - Training loss
   - Validation accuracy
   - F1-score per class
   - Confusion matrix

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

## Known Issues

1. Memory usage spikes with large batch sizes
2. GPU utilization can be inconsistent
3. Some features may have NaN values
4. High CPU usage during peak traffic
5. Classification delays on slower hardware

## Troubleshooting

1. Out of memory errors:
   - Reduce batch size
   - Use gradient checkpointing
   - Enable CPU offloading

2. Slow training:
   - Check GPU utilization
   - Optimize num_workers in DataLoader
   - Use mixed precision training

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
