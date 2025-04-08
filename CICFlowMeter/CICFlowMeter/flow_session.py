from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
from .features import FlowFeatures, Packet  # Import from features.py instead of duplicating

class FlowSession:
    """Manages network flow sessions and converts PacketInfo to Flow features"""
    
    def __init__(self, timeout: int = 120):
        self.flows: Dict[str, FlowFeatures] = {}
        self.timeout = timeout

    def _packet_to_flow(self, packet_info) -> Optional[Packet]:
        """Convert PacketInfo to Packet with safe defaults for missing attributes"""
        try:
            # Safely extract basic attributes with defaults
            time = getattr(packet_info, 'time', 0.0)
            length = getattr(packet_info, 'length', 0)
            src = getattr(packet_info, 'src', '0.0.0.0')
            sport = getattr(packet_info, 'sport', 0)
            dst = getattr(packet_info, 'dst', '0.0.0.0')
            dport = getattr(packet_info, 'dport', 0)
            protocol = getattr(packet_info, 'protocol', '0')
            flags = getattr(packet_info, 'flags', '')
            
            # Get window size with fallbacks
            window = 0
            for attr in ['window', 'window_size', 'win_size', 'tcp_window']:
                if hasattr(packet_info, attr):
                    window = getattr(packet_info, attr)
                    break
                    
            # Get header length with fallbacks
            ip_len = 0
            for attr in ['ip_len', 'header_length', 'ip_header_length', 'ihl']:
                if hasattr(packet_info, attr):
                    ip_len = getattr(packet_info, attr)
                    break

            # Determine flow direction
            forward_flow = True
            if hasattr(packet_info, 'forward_flow'):
                forward_flow = packet_info.forward_flow

            return Packet(
                time=float(time),
                length=int(length),
                src_ip=str(src),
                src_port=int(sport),
                dst_ip=str(dst),
                dst_port=int(dport),
                protocol=str(protocol),
                flags=flags,
                window=window,
                ip_len=ip_len,
                forward_flow=forward_flow
            )
        except Exception as e:
            print(f"Error converting packet: {str(e)}")
            return None

    def add_packet(self, packet_info) -> None:
        """Process a new packet and update corresponding flow"""
        packet = self._packet_to_flow(packet_info)
        if not packet:
            return

        flow_id = packet.get_flow_id()

        # Create new flow or update existing
        if flow_id not in self.flows:
            self.flows[flow_id] = FlowFeatures(packet)
        else:
            self.flows[flow_id].update(packet)

        # Clean up expired flows
        self._cleanup_flows()

    def _cleanup_flows(self) -> None:
        """Remove expired flows based on timeout"""
        current_time = datetime.now().timestamp()
        expired = []
        
        for flow_id, flow in self.flows.items():
            if current_time - flow.last_seen > self.timeout:
                expired.append(flow_id)
        
        for flow_id in expired:
            self.flows[flow_id].cleanup()
            del self.flows[flow_id]

    def get_flows(self) -> List[Dict]:
        """Get features for all flows"""
        return [flow.get_features() for flow in self.flows.values()]