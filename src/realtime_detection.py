"""
Real-Time Detection Module for Intrusion Detection System

This module integrates packet capture, feature extraction, and prediction
to perform real-time intrusion detection.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import logging
import threading
import time

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from packet_capture import PacketCapture, SimulatedPacketCapture
from feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Real-time intrusion detection system that captures packets,
    extracts features, and makes predictions.
    """
    
    def __init__(self, model_path, log_path=None, use_simulation=False):
        """
        Initialize the RealtimeDetector.
        
        Args:
            model_path (str): Path to the trained model file
            log_path (str): Path to the log file (optional)
            use_simulation (bool): If True, use simulated packet capture
        """
        self.model_path = model_path
        self.log_path = log_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'logs', 'alerts.log'
        )
        self.use_simulation = use_simulation
        
        # Load model
        self.model = self._load_model()
        
        # Initialize components
        if use_simulation:
            self.packet_capture = SimulatedPacketCapture(max_packets=1000, simulation_interval=0.5)
        else:
            self.packet_capture = PacketCapture(max_packets=1000)
        
        self.feature_extractor = FeatureExtractor(window_size=100)
        
        # Detection statistics
        self.stats = {
            'total_packets': 0,
            'normal_packets': 0,
            'attack_packets': 0,
            'start_time': None,
            'detection_history': []
        }
        
        # Callbacks for dashboard updates
        self.detection_callbacks = []
        
        # Running state
        self.is_running = False
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
    
    def _load_model(self):
        """
        Load the trained machine learning model.
        
        Returns:
            object: Loaded model
        """
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _log_alert(self, packet_info, prediction, confidence=None):
        """
        Log detection results to file.
        
        Args:
            packet_info (dict): Packet information
            prediction (int): Prediction result (0=normal, 1=attack)
            confidence (float): Prediction confidence (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        src_ip = packet_info.get('src_ip', 'unknown')
        dst_ip = packet_info.get('dst_ip', 'unknown')
        protocol = packet_info.get('protocol', 'unknown')
        result = 'Attack' if prediction == 1 else 'Normal'
        
        log_entry = f"[{timestamp}] {src_ip} -> {dst_ip} | Protocol: {protocol} | Result: {result}"
        if confidence is not None:
            log_entry += f" | Confidence: {confidence:.4f}"
        
        # Write to log file
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')
        
        logger.info(f"Alert logged: {log_entry}")
    
    def _process_packet(self, packet_info):
        """
        Process a single packet: extract features and make prediction.
        
        Args:
            packet_info (dict): Packet information from capture
        
        Returns:
            tuple: (prediction, confidence, packet_info)
        """
        try:
            # Extract features
            features_df = self.feature_extractor.extract_single_packet_features(packet_info)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Get prediction probability if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_df)[0]
                confidence = max(proba)
            
            # Update statistics
            self.stats['total_packets'] += 1
            if prediction == 0:
                self.stats['normal_packets'] += 1
            else:
                self.stats['attack_packets'] += 1
            
            # Create detection record
            detection_record = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'src_ip': packet_info.get('src_ip', 'unknown'),
                'dst_ip': packet_info.get('dst_ip', 'unknown'),
                'protocol': packet_info.get('protocol', 'unknown'),
                'packet_length': packet_info.get('packet_length', 0),
                'prediction': prediction,
                'result': 'Normal' if prediction == 0 else 'Attack',
                'confidence': confidence
            }
            
            self.stats['detection_history'].append(detection_record)
            
            # Keep only last 1000 records
            if len(self.stats['detection_history']) > 1000:
                self.stats['detection_history'] = self.stats['detection_history'][-1000:]
            
            # Log alert if attack detected
            if prediction == 1:
                self._log_alert(packet_info, prediction, confidence)
            
            # Call registered callbacks
            for callback in self.detection_callbacks:
                try:
                    callback(detection_record)
                except Exception as e:
                    logger.error(f"Error in detection callback: {e}")
            
            return prediction, confidence, detection_record
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            return None, None, None
    
    def start_detection(self, interface=None, filter_str=None, timeout=None):
        """
        Start real-time intrusion detection.
        
        Args:
            interface (str): Network interface to capture on
            filter_str (str): BPF filter string
            timeout (int): Detection timeout in seconds
        """
        if self.is_running:
            logger.warning("Detection is already running!")
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        logger.info("="*50)
        logger.info("STARTING REAL-TIME INTRUSION DETECTION")
        logger.info("="*50)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Log file: {self.log_path}")
        logger.info(f"Simulation mode: {self.use_simulation}")
        
        # Register packet callback
        self.packet_capture.register_callback(self._process_packet)
        
        # Start packet capture
        self.packet_capture.start_capture(
            interface=interface,
            filter_str=filter_str,
            timeout=timeout
        )
    
    def stop_detection(self):
        """Stop real-time intrusion detection."""
        if not self.is_running:
            logger.warning("Detection is not running!")
            return
        
        logger.info("Stopping real-time detection...")
        
        # Stop packet capture
        self.packet_capture.stop_capture()
        
        # Unregister callback
        self.packet_capture.unregister_callback(self._process_packet)
        
        self.is_running = False
        
        logger.info("Real-time detection stopped.")
    
    def get_statistics(self):
        """
        Get detection statistics.
        
        Returns:
            dict: Detection statistics
        """
        stats = self.stats.copy()
        
        # Calculate percentages
        if stats['total_packets'] > 0:
            stats['normal_percentage'] = (stats['normal_packets'] / stats['total_packets']) * 100
            stats['attack_percentage'] = (stats['attack_packets'] / stats['total_packets']) * 100
        else:
            stats['normal_percentage'] = 0
            stats['attack_percentage'] = 0
        
        # Calculate duration
        if stats['start_time']:
            stats['duration'] = time.time() - stats['start_time']
        
        return stats
    
    def get_recent_alerts(self, count=10):
        """
        Get recent attack alerts.
        
        Args:
            count (int): Number of alerts to return
        
        Returns:
            list: List of recent attack detection records
        """
        alerts = [r for r in self.stats['detection_history'] if r['prediction'] == 1]
        return alerts[-count:]
    
    def get_recent_detections(self, count=10):
        """
        Get recent detection records.
        
        Args:
            count (int): Number of records to return
        
        Returns:
            list: List of recent detection records
        """
        return self.stats['detection_history'][-count:]
    
    def register_callback(self, callback):
        """
        Register a callback function for detection updates.
        
        Args:
            callback (function): Callback function that takes detection_record as argument
        """
        self.detection_callbacks.append(callback)
        logger.info(f"Detection callback registered. Total callbacks: {len(self.detection_callbacks)}")
    
    def unregister_callback(self, callback):
        """
        Unregister a callback function.
        
        Args:
            callback (function): Callback function to remove
        """
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
            logger.info(f"Detection callback unregistered. Total callbacks: {len(self.detection_callbacks)}")
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.stats = {
            'total_packets': 0,
            'normal_packets': 0,
            'attack_packets': 0,
            'start_time': time.time() if self.is_running else None,
            'detection_history': []
        }
        logger.info("Detection statistics reset.")
    
    def clear_log(self):
        """Clear the alert log file."""
        try:
            with open(self.log_path, 'w') as f:
                f.write('')
            logger.info(f"Log file cleared: {self.log_path}")
        except Exception as e:
            logger.error(f"Error clearing log file: {e}")


def demo_realtime_detection():
    """
    Demonstrate real-time detection functionality.
    """
    # Paths
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ids_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first by running: python src/train_model.py")
        return
    
    # Create detector with simulation
    detector = RealtimeDetector(model_path, use_simulation=True)
    
    # Define a callback to print detections
    def print_detection(record):
        status = "NORMAL" if record['prediction'] == 0 else "ATTACK DETECTED"
        print(f"[{record['timestamp']}] {record['src_ip']:15} -> {record['dst_ip']:15} | "
              f"{record['protocol']:5} | {status}")
    
    # Register callback
    detector.register_callback(print_detection)
    
    # Start detection for 15 seconds
    print("\n" + "="*50)
    print("REAL-TIME DETECTION DEMO")
    print("="*50)
    print("Starting detection for 15 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    detector.start_detection(timeout=15)
    
    try:
        while detector.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping detection...")
        detector.stop_detection()
    
    # Print statistics
    stats = detector.get_statistics()
    print("\n" + "="*50)
    print("DETECTION STATISTICS")
    print("="*50)
    print(f"Total Packets: {stats['total_packets']}")
    print(f"Normal Packets: {stats['normal_packets']} ({stats['normal_percentage']:.1f}%)")
    print(f"Attack Packets: {stats['attack_packets']} ({stats['attack_percentage']:.1f}%)")
    print(f"Duration: {stats.get('duration', 0):.2f} seconds")
    
    # Print recent alerts
    alerts = detector.get_recent_alerts(count=5)
    if alerts:
        print("\n" + "="*50)
        print("RECENT ATTACK ALERTS")
        print("="*50)
        for alert in alerts:
            print(f"[{alert['timestamp']}] {alert['src_ip']} -> {alert['dst_ip']}")


if __name__ == "__main__":
    demo_realtime_detection()
