"""
Packet Capture Module for Intrusion Detection System

This module handles real-time network packet capture using Scapy.
Features:
- Start/stop packet sniffing
- Extract packet features
- Store captured packets for analysis
"""

from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
from collections import deque
import threading
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PacketCapture:
    """
    Class for capturing and processing network packets in real-time.
    """
    
    def __init__(self, max_packets=1000):
        """
        Initialize the PacketCapture.
        
        Args:
            max_packets (int): Maximum number of packets to store in memory
        """
        self.max_packets = max_packets
        self.captured_packets = deque(maxlen=max_packets)
        self.packet_count = 0
        self.is_capturing = False
        self.sniff_thread = None
        self.stop_event = threading.Event()
        self.packet_callbacks = []
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'other_packets': 0,
            'start_time': None
        }
    
    def _packet_handler(self, packet):
        """
        Internal packet handler called for each captured packet.
        
        Args:
            packet: Scapy packet object
        """
        if self.stop_event.is_set():
            return
        
        self.packet_count += 1
        self.stats['total_packets'] += 1
        
        # Extract packet info
        packet_info = self.extract_packet_info(packet)
        
        # Store packet
        self.captured_packets.append(packet_info)
        
        # Update statistics
        if packet_info['protocol'] == 'TCP':
            self.stats['tcp_packets'] += 1
        elif packet_info['protocol'] == 'UDP':
            self.stats['udp_packets'] += 1
        elif packet_info['protocol'] == 'ICMP':
            self.stats['icmp_packets'] += 1
        else:
            self.stats['other_packets'] += 1
        
        # Call registered callbacks
        for callback in self.packet_callbacks:
            try:
                callback(packet_info)
            except Exception as e:
                logger.error(f"Error in packet callback: {e}")
    
    def extract_packet_info(self, packet):
        """
        Extract relevant information from a packet.
        
        Args:
            packet: Scapy packet object
        
        Returns:
            dict: Extracted packet information
        """
        packet_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'packet_length': len(packet),
            'protocol': 'Other',
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'flags': None,
            'payload_size': 0,
            'raw_packet': packet
        }
        
        # Extract IP layer information
        if IP in packet:
            packet_info['src_ip'] = packet[IP].src
            packet_info['dst_ip'] = packet[IP].dst
            
            # Determine protocol
            if TCP in packet:
                packet_info['protocol'] = 'TCP'
                packet_info['src_port'] = packet[TCP].sport
                packet_info['dst_port'] = packet[TCP].dport
                packet_info['flags'] = str(packet[TCP].flags)
            elif UDP in packet:
                packet_info['protocol'] = 'UDP'
                packet_info['src_port'] = packet[UDP].sport
                packet_info['dst_port'] = packet[UDP].dport
            elif ICMP in packet:
                packet_info['protocol'] = 'ICMP'
            
            # Get payload size
            if Raw in packet:
                packet_info['payload_size'] = len(packet[Raw].load)
        
        return packet_info
    
    def start_capture(self, interface=None, filter_str=None, timeout=None):
        """
        Start capturing packets in a separate thread.
        
        Args:
            interface (str): Network interface to capture on (None for default)
            filter_str (str): BPF filter string (e.g., 'tcp', 'udp', 'port 80')
            timeout (int): Capture timeout in seconds (None for indefinite)
        """
        if self.is_capturing:
            logger.warning("Packet capture is already running!")
            return
        
        self.is_capturing = True
        self.stop_event.clear()
        self.stats['start_time'] = time.time()
        
        logger.info("="*50)
        logger.info("Starting Packet Capture...")
        logger.info("="*50)
        logger.info(f"Interface: {interface or 'default'}")
        logger.info(f"Filter: {filter_str or 'none'}")
        logger.info(f"Timeout: {timeout or 'indefinite'}")
        
        def capture_thread():
            try:
                sniff(
                    iface=interface,
                    filter=filter_str,
                    prn=self._packet_handler,
                    stop_filter=lambda x: self.stop_event.is_set(),
                    timeout=timeout,
                    store=0
                )
            except Exception as e:
                logger.error(f"Error in packet capture: {e}")
            finally:
                self.is_capturing = False
                logger.info("Packet capture stopped.")
        
        self.sniff_thread = threading.Thread(target=capture_thread)
        self.sniff_thread.daemon = True
        self.sniff_thread.start()
    
    def stop_capture(self):
        """
        Stop the packet capture.
        """
        if not self.is_capturing:
            logger.warning("Packet capture is not running!")
            return
        
        logger.info("Stopping packet capture...")
        self.stop_event.set()
        self.is_capturing = False
        
        if self.sniff_thread and self.sniff_thread.is_alive():
            self.sniff_thread.join(timeout=2)
        
        logger.info("Packet capture stopped.")
    
    def get_captured_packets(self, count=None):
        """
        Get captured packets.
        
        Args:
            count (int): Number of packets to return (None for all)
        
        Returns:
            list: List of captured packet information
        """
        packets = list(self.captured_packets)
        if count:
            return packets[-count:]
        return packets
    
    def get_statistics(self):
        """
        Get capture statistics.
        
        Returns:
            dict: Capture statistics
        """
        stats = self.stats.copy()
        
        # Calculate duration
        if stats['start_time']:
            stats['duration'] = time.time() - stats['start_time']
            if stats['duration'] > 0:
                stats['packets_per_second'] = stats['total_packets'] / stats['duration']
        
        return stats
    
    def register_callback(self, callback):
        """
        Register a callback function to be called for each packet.
        
        Args:
            callback (function): Callback function that takes packet_info as argument
        """
        self.packet_callbacks.append(callback)
        logger.info(f"Callback registered. Total callbacks: {len(self.packet_callbacks)}")
    
    def unregister_callback(self, callback):
        """
        Unregister a callback function.
        
        Args:
            callback (function): Callback function to remove
        """
        if callback in self.packet_callbacks:
            self.packet_callbacks.remove(callback)
            logger.info(f"Callback unregistered. Total callbacks: {len(self.packet_callbacks)}")
    
    def clear_packets(self):
        """
        Clear all captured packets.
        """
        self.captured_packets.clear()
        self.packet_count = 0
        logger.info("Captured packets cleared.")
    
    def reset_statistics(self):
        """
        Reset capture statistics.
        """
        self.stats = {
            'total_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'other_packets': 0,
            'start_time': None if not self.is_capturing else time.time()
        }
        logger.info("Statistics reset.")


class SimulatedPacketCapture(PacketCapture):
    """
    Simulated packet capture for testing without actual network access.
    Generates synthetic packets for demonstration purposes.
    """
    
    def __init__(self, max_packets=1000, simulation_interval=1.0):
        """
        Initialize the SimulatedPacketCapture.
        
        Args:
            max_packets (int): Maximum number of packets to store
            simulation_interval (float): Interval between simulated packets in seconds
        """
        super().__init__(max_packets)
        self.simulation_interval = simulation_interval
        self.simulation_thread = None
        
        # Sample data for simulation
        self.sample_ips = [
            '192.168.1.1', '192.168.1.2', '192.168.1.10', '192.168.1.50',
            '10.0.0.1', '10.0.0.5', '172.16.0.1', '8.8.8.8',
            '1.1.1.1', '192.168.0.1', '192.168.1.100', '192.168.1.200'
        ]
        self.sample_ports = [80, 443, 22, 21, 25, 53, 110, 143, 3306, 8080, 23, 445]
        self.protocols = ['TCP', 'UDP', 'ICMP']
        self.flags = ['S', 'A', 'PA', 'FA', 'R', 'SA']
    
    def _generate_simulated_packet(self):
        """
        Generate a simulated packet.
        
        Returns:
            dict: Simulated packet information
        """
        import random
        
        protocol = random.choice(self.protocols)
        
        packet_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'packet_length': random.randint(40, 1500),
            'protocol': protocol,
            'src_ip': random.choice(self.sample_ips),
            'dst_ip': random.choice(self.sample_ips),
            'src_port': None,
            'dst_port': None,
            'flags': None,
            'payload_size': random.randint(0, 1400),
            'raw_packet': None
        }
        
        if protocol in ['TCP', 'UDP']:
            packet_info['src_port'] = random.choice(self.sample_ports)
            packet_info['dst_port'] = random.choice(self.sample_ports)
        
        if protocol == 'TCP':
            packet_info['flags'] = random.choice(self.flags)
        
        return packet_info
    
    def start_capture(self, interface=None, filter_str=None, timeout=None):
        """
        Start simulated packet capture.
        
        Args:
            interface (str): Ignored for simulation
            filter_str (str): Ignored for simulation
            timeout (int): Capture timeout in seconds
        """
        if self.is_capturing:
            logger.warning("Simulated packet capture is already running!")
            return
        
        self.is_capturing = True
        self.stop_event.clear()
        self.stats['start_time'] = time.time()
        
        logger.info("="*50)
        logger.info("Starting Simulated Packet Capture...")
        logger.info("="*50)
        logger.info(f"Simulation interval: {self.simulation_interval} seconds")
        logger.info(f"Timeout: {timeout or 'indefinite'}")
        
        def simulation_thread():
            import random
            
            start_time = time.time()
            
            while not self.stop_event.is_set():
                # Generate simulated packet
                packet_info = self._generate_simulated_packet()
                
                # Process packet
                self.packet_count += 1
                self.stats['total_packets'] += 1
                self.captured_packets.append(packet_info)
                
                # Update statistics
                if packet_info['protocol'] == 'TCP':
                    self.stats['tcp_packets'] += 1
                elif packet_info['protocol'] == 'UDP':
                    self.stats['udp_packets'] += 1
                elif packet_info['protocol'] == 'ICMP':
                    self.stats['icmp_packets'] += 1
                else:
                    self.stats['other_packets'] += 1
                
                # Call registered callbacks
                for callback in self.packet_callbacks:
                    try:
                        callback(packet_info)
                    except Exception as e:
                        logger.error(f"Error in packet callback: {e}")
                
                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    logger.info(f"Simulation timeout reached ({timeout}s)")
                    break
                
                # Wait for next packet
                time.sleep(self.simulation_interval)
            
            self.is_capturing = False
            logger.info("Simulated packet capture stopped.")
        
        self.simulation_thread = threading.Thread(target=simulation_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()


def demo_packet_capture():
    """
    Demonstrate packet capture functionality.
    """
    # Use simulated capture for demo
    capture = SimulatedPacketCapture(max_packets=100, simulation_interval=0.5)
    
    # Define a callback to print packet info
    def print_packet(packet_info):
        print(f"[{packet_info['timestamp']}] {packet_info['protocol']:5} | "
              f"{packet_info['src_ip']:15} -> {packet_info['dst_ip']:15} | "
              f"Len: {packet_info['packet_length']:4}")
    
    # Register callback
    capture.register_callback(print_packet)
    
    # Start capture for 10 seconds
    capture.start_capture(timeout=10)
    
    # Wait for capture to complete
    try:
        while capture.is_capturing:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture...")
        capture.stop_capture()
    
    # Print statistics
    stats = capture.get_statistics()
    print("\n" + "="*50)
    print("CAPTURE STATISTICS")
    print("="*50)
    print(f"Total Packets: {stats['total_packets']}")
    print(f"TCP Packets: {stats['tcp_packets']}")
    print(f"UDP Packets: {stats['udp_packets']}")
    print(f"ICMP Packets: {stats['icmp_packets']}")
    print(f"Duration: {stats.get('duration', 0):.2f} seconds")
    print(f"Packets/sec: {stats.get('packets_per_second', 0):.2f}")


if __name__ == "__main__":
    demo_packet_capture()
