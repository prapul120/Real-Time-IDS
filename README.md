# Real-Time-IDS
# Real-Time Network Intrusion Detection System (IDS)

A Python-based Intrusion Detection System that uses Machine Learning to detect malicious network traffic in real-time, with a Streamlit dashboard for visualization and monitoring.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)

## Features

- **Machine Learning Models**: Decision Tree and Random Forest classifiers
- **Real-Time Packet Capture**: Live network monitoring using Scapy
- **Feature Extraction**: NSL-KDD style feature extraction from network packets
- **Attack Detection**: Binary classification (Normal vs Attack)
- **Streamlit Dashboard**: Real-time visualization and monitoring
- **Alert Logging**: Automatic logging of detected intrusions
- **Simulation Mode**: Test without actual network access

## Project Structure

```
ids_project/
│
├── dataset/
│   └── dataset.csv              # Training dataset (NSL-KDD format)
│
├── models/
│   └── ids_model.pkl            # Trained ML model
│
├── logs/
│   ├── alerts.log               # Detection alerts
│   └── ids.log                  # System logs
│
├── plots/
│   └── confusion_matrix_*.png   # Model evaluation plots
│
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── train_model.py           # Model training and evaluation
│   ├── packet_capture.py        # Packet capture (Scapy)
│   ├── feature_extraction.py    # Feature extraction from packets
│   └── realtime_detection.py    # Real-time detection engine
│
├── app.py                       # Streamlit dashboard
├── main.py                      # Main CLI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Root/administrator privileges (for live packet capture)

### Step 1: Clone or Download the Project

```bash
cd ids_project/
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning
- `scapy` - Network packet capture
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `joblib` - Model serialization
- `streamlit` - Web dashboard
- `plotly` - Interactive charts

## Usage

### Quick Start

#### 1. Train the Model

```bash
python main.py train
```

This will:
- Load and preprocess the dataset
- Train Decision Tree and Random Forest models
- Evaluate models using accuracy, confusion matrix, and classification report
- Save the best performing model to `models/ids_model.pkl`

#### 2. Launch the Dashboard

```bash
streamlit run app.py
```

Or use the CLI:

```bash
python main.py dashboard
```

The dashboard will open in your browser at `http://localhost:8501`

#### 3. Start Real-Time Detection

From the dashboard:
- Click the "▶️ Start" button to begin detection
- Monitor traffic statistics and alerts in real-time
- Click "⏹️ Stop" to stop detection

### Command Line Interface

The project includes a comprehensive CLI for all operations:

```bash
# Train the model
python main.py train

# Start detection with simulated packets (default)
python main.py detect

# Start detection with live packets (requires root/admin)
python main.py detect --live

# Run detection for a specific duration
python main.py detect --duration 60

# Test packet capture
python main.py test-capture --count 20

# Preprocess dataset only
python main.py preprocess

# Launch dashboard
python main.py dashboard
```

### Advanced Usage

#### Live Packet Capture (requires root/admin)

```bash
# Capture on specific interface
sudo python main.py detect --live --interface eth0

# Apply BPF filter
sudo python main.py detect --live --filter "tcp port 80"

# Run for 5 minutes
sudo python main.py detect --live --duration 300
```

#### Programmatic API

```python
from src.realtime_detection import RealtimeDetector

# Initialize detector
detector = RealtimeDetector('models/ids_model.pkl', use_simulation=True)

# Define callback
def on_detection(record):
    print(f"Detection: {record['result']} from {record['src_ip']}")

detector.register_callback(on_detection)

# Start detection
detector.start_detection()

# ... run for some time ...

# Stop detection
detector.stop_detection()

# Get statistics
stats = detector.get_statistics()
print(f"Total packets: {stats['total_packets']}")
print(f"Attacks detected: {stats['attack_packets']}")
```

## Dashboard Features

The Streamlit dashboard provides:

1. **Control Panel**
   - Start/Stop detection
   - Reset statistics
   - Clear alerts

2. **Real-Time Metrics**
   - Total packets analyzed
   - Normal traffic percentage
   - Attack traffic percentage
   - Detection duration

3. **Visualizations**
   - Traffic distribution pie chart
   - Detection timeline
   - Attack vs Normal traffic bar chart

4. **Alert Table**
   - Recent attack alerts with timestamps
   - Source and destination IP addresses
   - Protocol information

5. **Packet Log**
   - Last 20 captured packets
   - Real-time updates

## Model Training

### Dataset

The system uses the NSL-KDD dataset format with 41 features:

**Basic Features:**
- duration, protocol_type, service, flag
- src_bytes, dst_bytes, land

**Content Features:**
- wrong_fragment, urgent, hot
- num_failed_logins, logged_in
- num_compromised, root_shell, su_attempted

**Traffic Features:**
- count, srv_count, serror_rate, rerror_rate
- same_srv_rate, diff_srv_rate

**Host Features:**
- dst_host_count, dst_host_srv_count
- dst_host_same_srv_rate, dst_host_diff_srv_rate

### Models

Two models are trained and compared:

1. **Decision Tree**
   - Criterion: Gini
   - Max depth: 20
   - Fast training and inference

2. **Random Forest**
   - 100 estimators
   - Criterion: Gini
   - Max depth: 20
   - Better generalization

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Logging

### Alert Log (`logs/alerts.log`)

```
[2024-01-01 12:05:23] 192.168.1.10 -> 192.168.1.1 | Protocol: TCP | Result: Attack | Confidence: 0.95
[2024-01-01 12:06:45] 10.0.0.5 -> 8.8.8.8 | Protocol: UDP | Result: Normal
```

### System Log (`logs/ids.log`)

Contains detailed information about:
- Model training progress
- Detection events
- Errors and warnings

## Troubleshooting

### Issue: Permission Denied for Packet Capture

**Solution:** Run with sudo/administrator privileges:
```bash
sudo python main.py detect --live
```

### Issue: Model Not Found

**Solution:** Train the model first:
```bash
python main.py train
```

### Issue: Scapy Not Found

**Solution:** Install scapy with proper dependencies:
```bash
pip install scapy
```

On Linux, you may also need:
```bash
sudo apt-get install libpcap-dev
```

### Issue: Dashboard Not Loading

**Solution:** Check if streamlit is installed:
```bash
pip install streamlit
streamlit hello  # Test installation
```

## Performance Considerations

- **Simulation Mode**: Generates ~2 packets/second for testing
- **Live Capture**: Depends on network traffic volume
- **Model Inference**: ~1-2ms per packet on modern CPU
- **Dashboard Update**: Refreshes every 1 second

## Security Notes

- This IDS is for educational and demonstration purposes
- For production use, consider:
  - More sophisticated feature engineering
  - Additional ML models (XGBoost, Neural Networks)
  - Integration with SIEM systems
  - Encrypted alert transmission
  - Regular model retraining

## Contributing

Contributions are welcome! Areas for improvement:

- Additional ML models
- More sophisticated feature extraction
- Protocol-specific detection
- Integration with threat intelligence feeds
- Performance optimizations

## License

This project is licensed under the MIT License.

## Acknowledgments

- NSL-KDD dataset for intrusion detection research
- Scapy for network packet manipulation
- Scikit-learn for machine learning tools
- Streamlit for the dashboard framework

## Contact

For questions or issues, please open an issue on the project repository.

---

**Disclaimer**: This tool is for educational and authorized testing purposes only. Always ensure you have permission before monitoring network traffic.
