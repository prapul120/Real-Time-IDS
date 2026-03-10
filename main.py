"""
Main Execution Script for Intrusion Detection System

This script provides a command-line interface to run the IDS system
with various options for training, detection, and testing.
"""

import argparse
import os
import sys
import logging
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from train_model import ModelTrainer
from packet_capture import PacketCapture, SimulatedPacketCapture
from feature_extraction import FeatureExtractor
from realtime_detection import RealtimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'ids.log'))
    ]
)
logger = logging.getLogger(__name__)


def train_model_command():
    """Command to train the machine learning model."""
    logger.info("="*60)
    logger.info("TRAINING INTRUSION DETECTION MODEL")
    logger.info("="*60)
    
    # Paths
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset.csv')
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ids_model.pkl')
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please ensure the dataset is in the dataset/ directory.")
        return False
    
    try:
        # Preprocess data
        logger.info("Step 1: Preprocessing data...")
        preprocessor = DataPreprocessor(dataset_path)
        X, y = preprocessor.preprocess_pipeline(binary_classification=True)
        feature_names = preprocessor.get_feature_names()
        
        # Train models
        logger.info("Step 2: Training models...")
        trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
        trainer.split_data()
        trainer.train_all_models()
        
        # Plot confusion matrices
        logger.info("Step 3: Generating evaluation plots...")
        os.makedirs(plots_dir, exist_ok=True)
        for model_name in trainer.models.keys():
            plot_path = os.path.join(plots_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
            trainer.plot_confusion_matrix(model_name, save_path=plot_path)
        
        # Get feature importance
        logger.info("Step 4: Analyzing feature importance...")
        trainer.get_feature_importance('Decision Tree', feature_names)
        trainer.get_feature_importance('Random Forest', feature_names)
        
        # Compare and save best model
        logger.info("Step 5: Selecting and saving best model...")
        best_name, best_model = trainer.compare_models()
        trainer.save_best_model(model_path)
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Best Model: {best_name}")
        logger.info(f"Model Accuracy: {trainer.results[best_name]['accuracy']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False


def detect_command(use_simulation=True, duration=None, interface=None, filter_str=None):
    """Command to start real-time intrusion detection."""
    logger.info("="*60)
    logger.info("STARTING REAL-TIME INTRUSION DETECTION")
    logger.info("="*60)
    
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ids_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Please train the model first using: python main.py train")
        return False
    
    try:
        # Create detector
        detector = RealtimeDetector(model_path, use_simulation=use_simulation)
        
        # Define console callback
        def console_callback(record):
            status = "✅ NORMAL" if record['prediction'] == 0 else "🚨 ATTACK DETECTED"
            confidence_str = f" (Confidence: {record['confidence']:.2f})" if record['confidence'] else ""
            print(f"[{record['timestamp']}] {record['src_ip']:15} -> {record['dst_ip']:15} | "
                  f"{record['protocol']:5} | {status}{confidence_str}")
        
        detector.register_callback(console_callback)
        
        # Start detection
        mode = "SIMULATED" if use_simulation else "LIVE"
        logger.info(f"Starting {mode} packet capture...")
        if duration:
            logger.info(f"Detection will run for {duration} seconds")
        
        detector.start_detection(interface=interface, filter_str=filter_str, timeout=duration)
        
        # Wait for detection to complete
        try:
            while detector.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping detection...")
            detector.stop_detection()
        
        # Print final statistics
        stats = detector.get_statistics()
        logger.info("="*60)
        logger.info("DETECTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Packets: {stats['total_packets']}")
        logger.info(f"Normal Packets: {stats['normal_packets']} ({stats['normal_percentage']:.1f}%)")
        logger.info(f"Attack Packets: {stats['attack_packets']} ({stats['attack_percentage']:.1f}%)")
        logger.info(f"Duration: {stats.get('duration', 0):.2f} seconds")
        
        # Print recent alerts
        alerts = detector.get_recent_alerts(count=5)
        if alerts:
            logger.info("="*60)
            logger.info("RECENT ATTACK ALERTS")
            logger.info("="*60)
            for alert in alerts:
                logger.info(f"[{alert['timestamp']}] {alert['src_ip']} -> {alert['dst_ip']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return False


def test_capture_command(count=10, interval=1.0):
    """Command to test packet capture functionality."""
    logger.info("="*60)
    logger.info("TESTING PACKET CAPTURE")
    logger.info("="*60)
    
    # Use simulated capture for testing
    capture = SimulatedPacketCapture(max_packets=100, simulation_interval=interval)
    
    def print_callback(packet_info):
        print(f"[{packet_info['timestamp']}] {packet_info['protocol']:5} | "
              f"{packet_info['src_ip']:15} -> {packet_info['dst_ip']:15} | "
              f"Len: {packet_info['packet_length']:4}")
    
    capture.register_callback(print_callback)
    
    logger.info(f"Capturing {count} packets...")
    capture.start_capture(timeout=count * interval)
    
    try:
        while capture.is_capturing:
            time.sleep(0.5)
    except KeyboardInterrupt:
        capture.stop_capture()
    
    # Print statistics
    stats = capture.get_statistics()
    logger.info("="*60)
    logger.info("CAPTURE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Packets: {stats['total_packets']}")
    logger.info(f"TCP Packets: {stats['tcp_packets']}")
    logger.info(f"UDP Packets: {stats['udp_packets']}")
    logger.info(f"ICMP Packets: {stats['icmp_packets']}")


def dashboard_command():
    """Command to launch the Streamlit dashboard."""
    logger.info("="*60)
    logger.info("LAUNCHING STREAMLIT DASHBOARD")
    logger.info("="*60)
    
    import subprocess
    
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    if not os.path.exists(app_path):
        logger.error(f"Dashboard app not found at {app_path}")
        return False
    
    try:
        logger.info("Starting Streamlit server...")
        logger.info("The dashboard will open in your web browser.")
        logger.info("Press Ctrl+C to stop the server.")
        
        subprocess.run(['streamlit', 'run', app_path], check=True)
        return True
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped.")
        return True
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        return False


def preprocess_command():
    """Command to preprocess the dataset."""
    logger.info("="*60)
    logger.info("PREPROCESSING DATASET")
    logger.info("="*60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset.csv')
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return False
    
    try:
        preprocessor = DataPreprocessor(dataset_path)
        X, y = preprocessor.preprocess_pipeline(binary_classification=True)
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Feature names: {preprocessor.get_feature_names()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description='Intrusion Detection System (IDS) - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train the ML model
  python main.py detect                   # Start detection with simulated packets
  python main.py detect --live            # Start detection with live packets
  python main.py detect --duration 60     # Run detection for 60 seconds
  python main.py dashboard                # Launch the Streamlit dashboard
  python main.py test-capture             # Test packet capture
  python main.py preprocess               # Preprocess the dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the machine learning model')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Start real-time intrusion detection')
    detect_parser.add_argument('--live', action='store_true', help='Use live packet capture (requires root)')
    detect_parser.add_argument('--duration', type=int, help='Detection duration in seconds')
    detect_parser.add_argument('--interface', type=str, help='Network interface for capture')
    detect_parser.add_argument('--filter', type=str, help='BPF filter string')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the Streamlit dashboard')
    
    # Test capture command
    test_parser = subparsers.add_parser('test-capture', help='Test packet capture functionality')
    test_parser.add_argument('--count', type=int, default=10, help='Number of packets to capture')
    test_parser.add_argument('--interval', type=float, default=1.0, help='Interval between packets')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess the dataset')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        success = train_model_command()
    elif args.command == 'detect':
        success = detect_command(
            use_simulation=not args.live,
            duration=args.duration,
            interface=args.interface,
            filter_str=args.filter
        )
    elif args.command == 'dashboard':
        success = dashboard_command()
    elif args.command == 'test-capture':
        success = test_capture_command(count=args.count, interval=args.interval)
    elif args.command == 'preprocess':
        success = preprocess_command()
    else:
        parser.print_help()
        return
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
