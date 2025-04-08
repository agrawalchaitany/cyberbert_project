# CICFlowMeter-Python

A Python implementation of CICFlowMeter for network traffic analysis and flow-based feature extraction, capable of accurately extracting 84 network traffic flow features.

## Core Components

1. **Flow Management**
   - Flow session tracking
   - Bidirectional traffic analysis 
   - Flow expiration handling
   - Memory optimization

2. **Feature Extraction**
   - 84 statistical flow features
   - Forward/backward direction analysis
   - Inter-arrival time statistics
   - Bulk transfer analysis
   - Activity/idle periods tracking

3. **Data Processing**
   - Real-time packet processing
   - PCAP file analysis
   - Data preprocessing
   - High-precision calculations
   - CSV output generation

## Requirements

- Python 3.7+
- Administrator/root privileges for packet capture
- Required Python packages:
  - scapy>=2.4.5
  - numpy>=1.19.5
  - pandas>=1.3.0

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd CICFlowMeter-Python
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

From the project root directory (`C:\Users\DELL\Desktop\collage\projects\CyberBERT\CICFlowMeter`):

### Basic Commands

1. List available network interfaces:
```bash
python -m CICFlowMeter.CICFlowMeter.main -l
```

2. Capture live traffic:
```bash
python -m CICFlowMeter.CICFlowMeter.main -i <interface> -o flows.csv
```

3. Process PCAP file:
```bash
python -m CICFlowMeter.CICFlowMeter.main -f input.pcap -o output.csv
```

### Advanced Options

1. Capture with custom timeout and packet count:
```bash
python -m CICFlowMeter.CICFlowMeter.main -i <interface> -t 120 -c 1000 -o flows.csv
```

2. Process PCAP with specific packet limit:
```bash
python -m CICFlowMeter.CICFlowMeter.main -f input.pcap -c 5000 -o flows.csv
```

### Command Line Options

- `-i, --interface`: Network interface to capture
- `-f, --file`: PCAP file to process
- `-o, --output`: Output CSV file path (default: flows.csv)
- `-t, --timeout`: Flow timeout in seconds (default: 120)
- `-c, --count`: Number of packets to capture (0 = infinite)
- `-l, --list`: List available interfaces

### Output

The tool generates a CSV file containing:
- 84 flow features for each network flow
- Features organized in 15 groups
- High-precision numerical values
- ISO format timestamps

### Runtime Controls

- Press Ctrl+C to stop capture
- Progress updates every 100 packets
- Final statistics shown after completion:
  - Total packets processed
  - Valid flow packets
  - Active flows
  - Completed flows

## Project Structure

```
CICFlowMeter/
├── CICFlowMeter/              # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Entry point and CLI interface
│   ├── flow_meter.py         # Flow metering engine and state management
│   ├── flow_session.py       # Flow session tracking and management
│   ├── features.py           # Feature extraction and calculations (84 features)
│   └── utils.py              # Helper utilities and packet processing
├── docs/                      # Documentation directory
│   └── feature_documentation.md  # Detailed feature descriptions
├── requirements.txt          # Package dependencies
└── README.md                 # Project documentation

### Features Extracted

The tool extracts the following flow features:
- Basic flow information (IPs, ports, protocol)
- Flow duration
- Packet counts (forward/backward)
- Packet length statistics
- Flow bytes/packets per second
- Inter-arrival time statistics
- Active/Idle time statistics

## Common Issues

- **Permission Denied**: Make sure to run with administrator/root privileges when capturing packets
- **Interface Not Found**: Use the `-l` flag to list available interfaces
- **Module Not Found**: Ensure you're in the virtual environment and all dependencies are installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
