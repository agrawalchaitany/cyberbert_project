from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet
import pandas as pd

class PacketInfo:
    """Extracts and normalizes packet information from Scapy packets"""
    
    def __init__(self, packet: Packet):
        # Initialize basic attributes
        self.time = getattr(packet, 'time', 0.0)
        self.length = len(packet)
        
        # Initialize network attributes
        self.src_ip = None
        self.dst_ip = None
        self.src_port = None
        self.dst_port = None
        self.protocol = None
        self.flags = None
        self.window = 0
        self.ip_len = 0
        self.forward_flow = True

        # Extract IP layer information
        if IP in packet:
            ip_layer = packet[IP]
            self.src_ip = ip_layer.src
            self.dst_ip = ip_layer.dst
            self.protocol = ip_layer.proto
            self.ip_len = ip_layer.len

            # Handle TCP packets
            if TCP in packet:
                tcp_layer = packet[TCP]
                self.src_port = tcp_layer.sport
                self.dst_port = tcp_layer.dport
                self.flags = tcp_layer.flags
                self.window = tcp_layer.window
            
            # Handle UDP packets
            elif UDP in packet:
                udp_layer = packet[UDP]
                self.src_port = udp_layer.sport
                self.dst_port = udp_layer.dport

            # Normalize flow direction (smaller IP:port becomes source)
            self._normalize_direction()

    def _normalize_direction(self):
        """Normalize flow direction to ensure consistent flow tracking"""
        if self.src_ip > self.dst_ip:
            self.forward_flow = False
            self.src_ip, self.dst_ip = self.dst_ip, self.src_ip
            self.src_port, self.dst_port = self.dst_port, self.src_port
        elif self.src_ip == self.dst_ip and self.src_port > self.dst_port:
            self.forward_flow = False
            self.src_port, self.dst_port = self.dst_port, self.src_port

    def is_valid_flow(self) -> bool:
        """Verify if packet has all required flow information"""
        return all([
            self.src_ip is not None,
            self.dst_ip is not None,
            self.src_port is not None,
            self.dst_port is not None,
            self.protocol is not None
        ])

    def get_flow_id(self) -> str:
        """Generate unique flow identifier"""
        if not self.is_valid_flow():
            return None
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}-{self.protocol}"

def preprocess_value(value, feature_type="numerical", default=0):
    """
    Preprocess feature values for consistency and error handling
    
    Args:
        value: Raw feature value
        feature_type: Type of feature ("numerical" or "categorical")
        default: Default value if processing fails
    
    Returns:
        Processed value
    """
    if feature_type == "numerical":
        try:
            val = float(value)
            # Handle invalid numerical values
            if val in (float('inf'), float('-inf')) or pd.isna(val):
                return default
            return val
        except (ValueError, TypeError):
            return default
    
    return value
