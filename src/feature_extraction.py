"""
Feature Extraction Module for Intrusion Detection System

This module converts captured packet data into feature format suitable
for machine learning model prediction.
"""

import pandas as pd
import numpy as np
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Class for extracting features from captured network packets.
    Transforms packet data into the format expected by the ML model.
    """
    
    def __init__(self, window_size=100):
        """
        Initialize the FeatureExtractor.
        
        Args:
            window_size (int): Number of packets to consider for feature aggregation
        """
        self.window_size = window_size
        self.packet_history = deque(maxlen=window_size)
        
        # Protocol mapping (must match training data encoding)
        self.protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2, 'other': 3}
        
        # Service mapping (simplified)
        self.service_map = {
            'http': 0, 'ftp': 1, 'smtp': 2, 'telnet': 3, 'other': 4
        }
        
        # Flag mapping (simplified)
        self.flag_map = {
            'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'RSTR': 4, 'other': 5
        }
    
    def extract_features_from_packet(self, packet_info):
        """
        Extract features from a single packet.
        
        Args:
            packet_info (dict): Packet information from packet capture
        
        Returns:
            dict: Extracted features
        """
        features = {}
        
        # Basic packet features
        features['duration'] = 0  # Single packet has no duration
        features['protocol_type'] = self._encode_protocol(packet_info.get('protocol', 'other'))
        features['service'] = self._infer_service(packet_info.get('dst_port', 0))
        features['flag'] = self._encode_flag(packet_info.get('flags', 'other'))
        features['src_bytes'] = packet_info.get('packet_length', 0)
        features['dst_bytes'] = packet_info.get('payload_size', 0)
        features['land'] = 1 if packet_info.get('src_ip') == packet_info.get('dst_ip') else 0
        features['wrong_fragment'] = 0  # Would need deeper packet inspection
        features['urgent'] = 1 if packet_info.get('flags') and 'U' in str(packet_info.get('flags', '')) else 0
        
        # Content features (simplified)
        features['hot'] = self._calculate_hot_features(packet_info)
        features['num_failed_logins'] = 0  # Would need application layer inspection
        features['logged_in'] = 1 if packet_info.get('flags') == 'PA' else 0
        features['num_compromised'] = 0
        features['root_shell'] = 0
        features['su_attempted'] = 0
        features['num_root'] = 0
        features['num_file_creations'] = 0
        features['num_shells'] = 0
        features['num_access_files'] = 0
        features['num_outbound_cmds'] = 0
        features['is_host_login'] = 1 if packet_info.get('dst_port') == 513 else 0
        features['is_guest_login'] = 0
        
        # Traffic features based on packet history
        traffic_features = self._calculate_traffic_features(packet_info)
        features.update(traffic_features)
        
        # Add packet to history
        self.packet_history.append(packet_info)
        
        return features
    
    def _encode_protocol(self, protocol):
        """Encode protocol type to numeric value."""
        return self.protocol_map.get(protocol.lower(), 3)
    
    def _infer_service(self, port):
        """Infer service type from port number."""
        if port is None:
            return 4  # other
        
        service_ports = {
            80: 0,    # http
            443: 0,   # http (https)
            8080: 0,  # http
            21: 1,    # ftp
            20: 1,    # ftp
            25: 2,    # smtp
            23: 3,    # telnet
        }
        
        return service_ports.get(port, 4)  # default to other
    
    def _encode_flag(self, flag):
        """Encode TCP flag to numeric value."""
        if flag is None:
            return 5  # other
        return self.flag_map.get(flag, 5)
    
    def _calculate_hot_features(self, packet_info):
        """Calculate 'hot' indicators (suspicious patterns)."""
        hot = 0
        
        # Check for suspicious port patterns
        dst_port = packet_info.get('dst_port')
        if dst_port:
            suspicious_ports = [23, 513, 514]  # telnet, rlogin, rsh
            if dst_port in suspicious_ports:
                hot += 1
        
        # Check for large packets
        if packet_info.get('packet_length', 0) > 1000:
            hot += 1
        
        return hot
    
    def _calculate_traffic_features(self, current_packet):
        """Calculate traffic-based features using packet history."""
        features = {}
        
        # Default values
        features['count'] = len(self.packet_history)
        features['srv_count'] = 0
        features['serror_rate'] = 0.0
        features['srv_serror_rate'] = 0.0
        features['rerror_rate'] = 0.0
        features['srv_rerror_rate'] = 0.0
        features['same_srv_rate'] = 0.0
        features['diff_srv_rate'] = 0.0
        features['srv_diff_host_rate'] = 0.0
        
        # Destination host features (simplified)
        features['dst_host_count'] = len(self.packet_history)
        features['dst_host_srv_count'] = 0
        features['dst_host_same_srv_rate'] = 0.0
        features['dst_host_diff_srv_rate'] = 0.0
        features['dst_host_same_src_port_rate'] = 0.0
        features['dst_host_srv_diff_host_rate'] = 0.0
        features['dst_host_serror_rate'] = 0.0
        features['dst_host_srv_serror_rate'] = 0.0
        features['dst_host_rerror_rate'] = 0.0
        features['dst_host_srv_rerror_rate'] = 0.0
        
        if len(self.packet_history) == 0:
            return features
        
        # Calculate rates based on packet history
        current_dst_ip = current_packet.get('dst_ip')
        current_protocol = current_packet.get('protocol')
        current_dst_port = current_packet.get('dst_port')
        
        same_dst_count = 0
        same_srv_count = 0
        syn_errors = 0
        rej_errors = 0
        
        for pkt in self.packet_history:
            # Same destination
            if pkt.get('dst_ip') == current_dst_ip:
                same_dst_count += 1
                
                # Same service (port)
                if pkt.get('dst_port') == current_dst_port:
                    same_srv_count += 1
            
            # SYN errors (S0 flag)
            if pkt.get('flags') == 'S0':
                syn_errors += 1
            
            # REJ errors
            if pkt.get('flags') == 'REJ':
                rej_errors += 1
        
        # Update features
        features['count'] = same_dst_count
        features['srv_count'] = same_srv_count
        
        if len(self.packet_history) > 0:
            features['serror_rate'] = syn_errors / len(self.packet_history)
            features['srv_serror_rate'] = syn_errors / len(self.packet_history)
            features['rerror_rate'] = rej_errors / len(self.packet_history)
            features['srv_rerror_rate'] = rej_errors / len(self.packet_history)
        
        if same_dst_count > 0:
            features['same_srv_rate'] = same_srv_count / same_dst_count
            features['diff_srv_rate'] = 1 - features['same_srv_rate']
        
        # Destination host features
        features['dst_host_count'] = same_dst_count
        features['dst_host_srv_count'] = same_srv_count
        
        if same_dst_count > 0:
            features['dst_host_same_srv_rate'] = same_srv_count / same_dst_count
            features['dst_host_diff_srv_rate'] = 1 - features['dst_host_same_srv_rate']
            features['dst_host_serror_rate'] = features['serror_rate']
            features['dst_host_srv_serror_rate'] = features['srv_serror_rate']
            features['dst_host_rerror_rate'] = features['rerror_rate']
            features['dst_host_srv_rerror_rate'] = features['srv_rerror_rate']
        
        return features
    
    def transform_to_dataframe(self, features_list):
        """
        Transform a list of feature dictionaries to a DataFrame.
        
        Args:
            features_list (list): List of feature dictionaries
        
        Returns:
            pd.DataFrame: Features in DataFrame format
        """
        if not features_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(features_list)
        
        # Ensure column order matches training data
        expected_columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Reorder columns
        df = df[expected_columns]
        
        return df
    
    def extract_single_packet_features(self, packet_info):
        """
        Extract features from a single packet and return as DataFrame.
        
        Args:
            packet_info (dict): Packet information
        
        Returns:
            pd.DataFrame: Single row DataFrame with features
        """
        features = self.extract_features_from_packet(packet_info)
        return self.transform_to_dataframe([features])
    
    def clear_history(self):
        """Clear the packet history."""
        self.packet_history.clear()
        logger.info("Packet history cleared.")


class SimpleFeatureExtractor:
    """
    Simplified feature extractor for basic packet features.
    Useful when full NSL-KDD feature set is not needed.
    """
    
    def __init__(self):
        """Initialize the SimpleFeatureExtractor."""
        self.feature_names = [
            'packet_length', 'protocol_type', 'src_port', 'dst_port',
            'has_payload', 'is_tcp', 'is_udp', 'is_icmp'
        ]
    
    def extract(self, packet_info):
        """
        Extract simple features from packet info.
        
        Args:
            packet_info (dict): Packet information
        
        Returns:
            np.array: Feature array
        """
        protocol = packet_info.get('protocol', 'Other')
        
        features = [
            packet_info.get('packet_length', 0),
            0 if protocol == 'TCP' else (1 if protocol == 'UDP' else 2),
            packet_info.get('src_port', 0) if packet_info.get('src_port') else 0,
            packet_info.get('dst_port', 0) if packet_info.get('dst_port') else 0,
            1 if packet_info.get('payload_size', 0) > 0 else 0,
            1 if protocol == 'TCP' else 0,
            1 if protocol == 'UDP' else 0,
            1 if protocol == 'ICMP' else 0
        ]
        
        return np.array(features).reshape(1, -1)
    
    def get_feature_names(self):
        """Get list of feature names."""
        return self.feature_names


def demo_feature_extraction():
    """
    Demonstrate feature extraction functionality.
    """
    extractor = FeatureExtractor(window_size=10)
    
    # Simulate some packets
    sample_packets = [
        {
            'timestamp': '2024-01-01 12:00:00',
            'packet_length': 150,
            'protocol': 'TCP',
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.2',
            'src_port': 12345,
            'dst_port': 80,
            'flags': 'S',
            'payload_size': 0
        },
        {
            'timestamp': '2024-01-01 12:00:01',
            'packet_length': 200,
            'protocol': 'TCP',
            'src_ip': '192.168.1.2',
            'dst_ip': '192.168.1.1',
            'src_port': 80,
            'dst_port': 12345,
            'flags': 'SA',
            'payload_size': 0
        },
        {
            'timestamp': '2024-01-01 12:00:02',
            'packet_length': 500,
            'protocol': 'UDP',
            'src_ip': '192.168.1.1',
            'dst_ip': '8.8.8.8',
            'src_port': 12346,
            'dst_port': 53,
            'flags': None,
            'payload_size': 100
        }
    ]
    
    print("="*50)
    print("FEATURE EXTRACTION DEMO")
    print("="*50)
    
    all_features = []
    for i, packet in enumerate(sample_packets):
        features = extractor.extract_features_from_packet(packet)
        all_features.append(features)
        print(f"\nPacket {i+1}:")
        print(f"  Protocol: {packet['protocol']}")
        print(f"  Src: {packet['src_ip']}:{packet['src_port']}")
        print(f"  Dst: {packet['dst_ip']}:{packet['dst_port']}")
        print(f"  Features extracted: {len(features)}")
        print(f"  Key features: duration={features['duration']}, "
              f"protocol_type={features['protocol_type']}, "
              f"service={features['service']}, "
              f"src_bytes={features['src_bytes']}")
    
    # Transform to DataFrame
    df = extractor.transform_to_dataframe(all_features)
    print(f"\n{'='*50}")
    print("FEATURE DATAFRAME")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    demo_feature_extraction()
