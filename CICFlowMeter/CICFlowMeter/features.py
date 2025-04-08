from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
from .utils import preprocess_value

class Packet:
    def __init__(self, time: float, length: int, src_ip: str, src_port: int, 
                 dst_ip: str, dst_port: int, protocol: str, flags: str = '',
                 window: int = 0, ip_len: int = 0, forward_flow: bool = True):
        self.time = time
        self.length = length
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol
        self.flags = flags
        self.window = window
        self.ip_len = ip_len
        self.forward_flow = forward_flow

    def get_flow_id(self) -> str:
        return f"{self.src_ip}-{self.dst_ip}-{self.src_port}-{self.dst_port}-{self.protocol}"

class FlowFeatures:
    def __init__(self, packet: Packet):
        # Initialize basic flow attributes using the normalized flow direction
        self.flow_id = packet.get_flow_id()
        self.src_ip = packet.src_ip
        self.src_port = packet.src_port
        self.dst_ip = packet.dst_ip
        self.dst_port = packet.dst_port
        self.protocol = packet.protocol
        self.timestamp = packet.time
        self.start_time = packet.time
        self.last_seen = packet.time
        
        # Initialize packet lists
        self.forward_packets: List[Packet] = []
        self.backward_packets: List[Packet] = []
        
        # Initialize counters
        self.total_fwd_packets = 0
        self.total_backward_packets = 0
        self.total_length_of_fwd_packet = 0
        self.total_length_of_bwd_packet = 0
        
        # Initialize packet lengths
        self.fwd_packet_length_max = 0
        self.fwd_packet_length_min = float('inf')
        self.fwd_packet_length_mean = 0
        self.fwd_packet_length_std = 0
        self.bwd_packet_length_max = 0
        self.bwd_packet_length_min = float('inf')
        self.bwd_packet_length_mean = 0
        self.bwd_packet_length_std = 0
        
        # Initialize IAT stats
        self.flow_iat_mean = 0
        self.flow_iat_std = 0
        self.flow_iat_max = 0
        self.flow_iat_min = float('inf')
        self.fwd_iat_total = 0
        self.fwd_iat_mean = 0
        self.fwd_iat_std = 0
        self.fwd_iat_max = 0
        self.fwd_iat_min = float('inf')
        self.bwd_iat_total = 0
        self.bwd_iat_mean = 0
        self.bwd_iat_std = 0
        self.bwd_iat_max = 0
        self.bwd_iat_min = float('inf')
        
        # Initialize flag counters
        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0
        self.fin_flag_count = 0
        self.syn_flag_count = 0
        self.rst_flag_count = 0
        self.psh_flag_count = 0
        self.ack_flag_count = 0
        self.urg_flag_count = 0
        self.cwe_flag_count = 0
        self.ece_flag_count = 0
        
        # Initialize flow rates
        self.flow_bytes_s = 0
        self.flow_packets_s = 0
        self.fwd_packets_s = 0
        self.bwd_packets_s = 0
        
        # Initialize header lengths
        self.fwd_header_length = 0
        self.bwd_header_length = 0
        
        # Initialize window sizes
        self.init_win_bytes_forward = 0
        self.init_win_bytes_backward = 0
        
        # Initialize bulk stats
        self.fwd_bytes_bulk_avg = 0  # Changed from fwd_bulk_bytes_avg
        self.fwd_packet_bulk_avg = 0
        self.fwd_bulk_rate_avg = 0
        self.bwd_bytes_bulk_avg = 0  # Changed from bwd_bulk_bytes_avg
        self.bwd_packet_bulk_avg = 0
        self.bwd_bulk_rate_avg = 0
        
        # Initialize additional metrics
        self.down_up_ratio = 0
        self.average_packet_size = 0
        self.fwd_segment_size_avg = 0
        self.bwd_segment_size_avg = 0
        self.fwd_seg_size_min = float('inf')
        self.packet_length_min = float('inf')
        self.packet_length_max = 0
        self.packet_length_mean = 0
        self.packet_length_std = 0
        self.packet_length_variance = 0
        
        # Initialize subflow stats
        self.subflow_fwd_packets = 0
        self.subflow_fwd_bytes = 0
        self.subflow_bwd_packets = 0
        self.subflow_bwd_bytes = 0
        
        # Initialize activity tracking
        self.active_mean = 0
        self.active_std = 0
        self.active_max = 0
        self.active_min = float('inf')
        self.idle_mean = 0
        self.idle_std = 0
        self.idle_max = 0
        self.idle_min = float('inf')
        self.active_start_time = packet.time
        self.last_active_time = packet.time
        self.active_times: List[float] = []
        self.idle_times: List[float] = []
        self.idle_threshold = 2.0
        
        # Initialize the label
        self.label = "NeedManualLabel"
        
        # Additional packet size metrics
        self.min_seg_size_forward = float('inf')  # Add this initialization
        self.fwd_act_data_pkts = 0
        self.fwd_seg_size_min = float('inf')

        # Add missing byte counters
        self.total_fwd_bytes = 0
        self.total_bwd_bytes = 0
        self.fwd_iat_times = []
        self.bwd_iat_times = []

        # Add first packet based on its actual direction
        if packet.forward_flow:
            self.forward_packets.append(packet)
            self._update_forward_stats(packet)
        else:
            self.backward_packets.append(packet)
            self._update_backward_stats(packet)
        self._update_flags(packet, packet.forward_flow)

    def update(self, packet: Packet) -> None:
        """Update flow features with a new packet"""
        # Use packet's normalized direction instead of comparing IPs
        if packet.forward_flow:
            self.forward_packets.append(packet)
            self._update_forward_stats(packet)
        else:
            self.backward_packets.append(packet)
            self._update_backward_stats(packet)

        self._update_flags(packet, packet.forward_flow)
        self._update_iat_stats(packet, packet.forward_flow)
        self._update_flow_stats()
        self._update_activity_stats(packet.time)
        self.last_seen = packet.time

    def _update_forward_stats(self, packet: Packet) -> None:
        """Update forward packet statistics"""
        self.total_fwd_packets += 1
        self.total_length_of_fwd_packet += packet.length
        
        # Update length statistics
        self.fwd_packet_length_max = max(self.fwd_packet_length_max, packet.length)
        self.fwd_packet_length_min = min(self.fwd_packet_length_min, packet.length)
        self.min_seg_size_forward = min(self.min_seg_size_forward, packet.length)
        self.fwd_seg_size_min = min(self.fwd_seg_size_min, packet.length)

        self.total_fwd_bytes += packet.length

        # Update header length if available
        if hasattr(packet, 'ip_len') and packet.ip_len:
            self.fwd_header_length += packet.ip_len
            if packet.length > packet.ip_len:
                self.fwd_act_data_pkts += 1

        # Update window size
        if self.init_win_bytes_forward == 0:
            self.init_win_bytes_forward = packet.window

    def _update_backward_stats(self, packet: Packet) -> None:
        """Update backward packet statistics"""
        self.total_backward_packets += 1
        self.total_length_of_bwd_packet += packet.length
        self.bwd_packet_length_max = max(self.bwd_packet_length_max, packet.length)
        self.bwd_packet_length_min = min(self.bwd_packet_length_min, packet.length)
        self.total_bwd_bytes += packet.length
        # Only update header length if ip_len is available
        if hasattr(packet, 'ip_len') and packet.ip_len:
            self.bwd_header_length += packet.ip_len
        if self.init_win_bytes_backward == 0:
            self.init_win_bytes_backward = packet.window

    def _update_flow_stats(self) -> None:
        """Update flow-level statistics"""
        duration = self.get_duration()
        if duration > 0:
            total_bytes = self.total_length_of_fwd_packet + self.total_length_of_bwd_packet
            total_packets = self.total_fwd_packets + self.total_backward_packets
            
            # Calculate flow rates
            self.flow_bytes_s = (total_bytes / duration) * 1_000_000
            self.flow_packets_s = (total_packets / duration) * 1_000_000
            self.fwd_packets_s = (self.total_fwd_packets / duration) * 1_000_000
            self.bwd_packets_s = (self.total_backward_packets / duration) * 1_000_000

            # Calculate mean packet lengths safely
            if self.total_fwd_packets > 0:
                self.fwd_packet_length_mean = self.total_length_of_fwd_packet / self.total_fwd_packets
                # Calculate standard deviation for forward packets
                if len(self.forward_packets) > 0:
                    lengths = [p.length for p in self.forward_packets]
                    self.fwd_packet_length_std = np.std(lengths)

            if self.total_backward_packets > 0:
                self.bwd_packet_length_mean = self.total_length_of_bwd_packet / self.total_backward_packets
                # Calculate standard deviation for backward packets
                if len(self.backward_packets) > 0:
                    lengths = [p.length for p in self.backward_packets]
                    self.bwd_packet_length_std = np.std(lengths)

            # Calculate flow IAT statistics
            if len(self.forward_packets) + len(self.backward_packets) > 1:
                all_packets = sorted(self.forward_packets + self.backward_packets, key=lambda x: x.time)
                iats = [p2.time - p1.time for p1, p2 in zip(all_packets[:-1], all_packets[1:])]
                if iats:
                    self.flow_iat_mean = np.mean(iats)
                    self.flow_iat_std = np.std(iats)

    def get_features(self) -> Dict[str, Any]:
        """Get all flow features matching CSV format exactly"""
        self._finalize_calculations()
        
        return {
            "Flow ID": self.flow_id,
            "Src IP": self.src_ip, 
            "Src Port": self.src_port,
            "Dst IP": self.dst_ip,
            "Dst Port": self.dst_port,
            "Protocol": self.protocol,
            "Timestamp": self.timestamp,
            "Flow Duration": int(self.get_duration() * 1_000_000),
            "Total Fwd Packets": self.total_fwd_packets,  # Note slight name difference
            "Total Bwd packets": self.total_backward_packets,  # Note slight name difference
            "Total Length of Fwd Packet": self.total_length_of_fwd_packet,
            "Total Length of Bwd Packet": self.total_length_of_bwd_packet,
            "Fwd Packet Length Max": self.fwd_packet_length_max,
            "Fwd Packet Length Min": self.fwd_packet_length_min,
            "Fwd Packet Length Mean": self.fwd_packet_length_mean,
            "Fwd Packet Length Std": self.fwd_packet_length_std,
            "Bwd Packet Length Max": self.bwd_packet_length_max,
            "Bwd Packet Length Min": self.bwd_packet_length_min,
            "Bwd Packet Length Mean": self.bwd_packet_length_mean,
            "Bwd Packet Length Std": self.bwd_packet_length_std,
            "Flow Bytes/s": self.flow_bytes_s,
            "Flow Packets/s": self.flow_packets_s,
            "Flow IAT Mean": self.flow_iat_mean,
            "Flow IAT Std": self.flow_iat_std,
            "Flow IAT Max": self.flow_iat_max,
            "Flow IAT Min": self.flow_iat_min,
            "Fwd IAT Total": self.fwd_iat_total,
            "Fwd IAT Mean": self.fwd_iat_mean,
            "Fwd IAT Std": self.fwd_iat_std,
            "Fwd IAT Max": self.fwd_iat_max,
            "Fwd IAT Min": self.fwd_iat_min,
            "Bwd IAT Total": self.bwd_iat_total,
            "Bwd IAT Mean": self.bwd_iat_mean,
            "Bwd IAT Std": self.bwd_iat_std,
            "Bwd IAT Max": self.bwd_iat_max,
            "Bwd IAT Min": self.bwd_iat_min,
            "Fwd PSH Flags": self.fwd_psh_flags,
            "Bwd PSH Flags": self.bwd_psh_flags,
            "Fwd URG Flags": self.fwd_urg_flags,
            "Bwd URG Flags": self.bwd_urg_flags,
            "Fwd Header Length": self.fwd_header_length,
            "Bwd Header Length": self.bwd_header_length,
            "Fwd Packets/s": self.fwd_packets_s,
            "Bwd Packets/s": self.bwd_packets_s,
            "Min Packet Length": self.packet_length_min,
            "Max Packet Length": self.packet_length_max,
            "Packet Length Mean": self.packet_length_mean,
            "Packet Length Std": self.packet_length_std,
            "Packet Length Variance": self.packet_length_variance,
            "FIN Flag Count": self.fin_flag_count,
            "SYN Flag Count": self.syn_flag_count,
            "RST Flag Count": self.rst_flag_count,
            "PSH Flag Count": self.psh_flag_count,
            "ACK Flag Count": self.ack_flag_count,
            "URG Flag Count": self.urg_flag_count,
            "CWR Flag Count": self.cwe_flag_count,  # Note CWR vs CWE in name
            "ECE Flag Count": self.ece_flag_count,
            "Down/Up Ratio": self.down_up_ratio,
            "Average Packet Size": self.average_packet_size,
            "Avg Fwd Segment Size": self.fwd_segment_size_avg,
            "Avg Bwd Segment Size": self.bwd_segment_size_avg,
            "Fwd Bytes/Bulk Avg": self.fwd_bytes_bulk_avg,  # Note slight name difference
            "Fwd Packet/Bulk Avg": self.fwd_packet_bulk_avg,  # Note slight name difference
            "Fwd Bulk Rate Avg": self.fwd_bulk_rate_avg,
            "Bwd Bytes/Bulk Avg": self.bwd_bytes_bulk_avg,  # Note slight name difference
            "Bwd Packet/Bulk Avg": self.bwd_packet_bulk_avg,  # Note slight name difference
            "Bwd Bulk Rate Avg": self.bwd_bulk_rate_avg,
            "Subflow Fwd Packets": self.subflow_fwd_packets,
            "Subflow Fwd Bytes": self.subflow_fwd_bytes,
            "Subflow Bwd Packets": self.subflow_bwd_packets,
            "Subflow Bwd Bytes": self.subflow_bwd_bytes,
            "FWD Init Win Bytes": self.init_win_bytes_forward,  # Note name difference
            "Bwd Init Win Bytes": self.init_win_bytes_backward,  # Note name difference
            "Fwd Act Data Pkts": self.fwd_act_data_pkts,
            "Fwd Seg Size Min": self.fwd_seg_size_min,
            "Active Mean": self.active_mean,
            "Active Std": self.active_std,
            "Active Max": self.active_max,
            "Active Min": self.active_min,
            "Idle Mean": self.idle_mean,
            "Idle Std": self.idle_std,
            "Idle Max": self.idle_max,
            "Idle Min": self.idle_min,
            "Label": self.label
        }

    def get_duration(self) -> float:
        """Get flow duration in seconds"""
        return self.last_seen - self.start_time

    def _finalize_calculations(self) -> None:
        """Finalize all statistical calculations in proper order"""
        # Basic stats must be calculated first
        self._calculate_basic_stats()
        
        # Flow timing stats
        self._calculate_iat_stats()
        
        # Calculate all other stats
        self._calculate_flow_rates()
        self._calculate_packet_stats()
        self._calculate_bulk_stats()
        self._calculate_subflow_stats()
        self._calculate_ratios()
        self._finalize_flow_stats()

    def _calculate_basic_stats(self) -> None:
        """Calculate basic flow statistics"""
        # Reset min values for safety
        if self.fwd_packet_length_min == float('inf'):
            self.fwd_packet_length_min = 0
        if self.bwd_packet_length_min == float('inf'):
            self.bwd_packet_length_min = 0
        if self.flow_iat_min == float('inf'):
            self.flow_iat_min = 0
        if self.fwd_iat_min == float('inf'):
            self.fwd_iat_min = 0
        if self.bwd_iat_min == float('inf'):
            self.bwd_iat_min = 0
        if self.active_min == float('inf'):
            self.active_min = 0
        if self.idle_min == float('inf'):
            self.idle_min = 0

    def _calculate_flow_rates(self) -> None:
        """Calculate flow rates with proper scaling"""
        duration = self.get_duration()
        if duration > 0:
            total_bytes = self.total_length_of_fwd_packet + self.total_length_of_bwd_packet
            total_packets = self.total_fwd_packets + self.total_backward_packets
            
            # Convert to per second rates and apply preprocessing
            self.flow_bytes_s = preprocess_value(total_bytes / duration)
            self.flow_packets_s = preprocess_value(total_packets / duration)
            self.fwd_packets_s = preprocess_value(self.total_fwd_packets / duration)
            self.bwd_packets_s = preprocess_value(self.total_backward_packets / duration)
        else:
            self.flow_bytes_s = 0
            self.flow_packets_s = 0
            self.fwd_packets_s = 0
            self.bwd_packets_s = 0

    def _finalize_flow_stats(self) -> None:
        """Finalize any remaining flow statistics"""
        # Handle packet length edge cases
        if not self.forward_packets and not self.backward_packets:
            self.packet_length_min = 0
            self.packet_length_max = 0
            self.packet_length_mean = 0
            self.packet_length_std = 0
            self.packet_length_variance = 0
            return

        # Combine all packets for overall stats
        all_packets = self.forward_packets + self.backward_packets
        all_lengths = [p.length for p in all_packets]
        
        # Calculate overall packet stats
        self.packet_length_min = min(all_lengths) if all_lengths else 0
        self.packet_length_max = max(all_lengths) if all_lengths else 0
        self.packet_length_mean = np.mean(all_lengths) if all_lengths else 0
        self.packet_length_std = np.std(all_lengths) if all_lengths else 0
        self.packet_length_variance = np.var(all_lengths) if all_lengths else 0

        # Handle segment size edge cases
        if self.fwd_seg_size_min == float('inf'):
            self.fwd_seg_size_min = 0
        if self.min_seg_size_forward == float('inf'):
            self.min_seg_size_forward = 0

        # Ensure active/idle stats are properly set
        if not self.active_times:
            self.active_mean = 0
            self.active_std = 0
            self.active_max = 0
            self.active_min = 0

        if not self.idle_times:
            self.idle_mean = 0
            self.idle_std = 0
            self.idle_max = 0
            self.idle_min = 0

    def _calculate_packet_stats(self) -> None:
        """Calculate packet statistics according to CICFlowMeter specifications
        - Packet length stats are based on the entire packet length (headers + payload)
        - Segment size stats are based on the actual data size (excluding headers)
        """
        # Calculate forward packet stats
        if self.forward_packets:
            fwd_lengths = [p.length for p in self.forward_packets]
            fwd_data_lengths = [p.length - p.ip_len for p in self.forward_packets if p.ip_len <= p.length]
            
            # Standard packet length statistics
            self.fwd_packet_length_max = max(fwd_lengths)
            self.fwd_packet_length_min = min(fwd_lengths)
            self.fwd_packet_length_mean = np.mean(fwd_lengths)
            self.fwd_packet_length_std = np.std(fwd_lengths)
            
            # Segment size statistics (actual data without headers)
            if fwd_data_lengths:
                self.fwd_segment_size_avg = np.mean(fwd_data_lengths)
                self.fwd_seg_size_min = min(fwd_data_lengths)
            else:
                self.fwd_segment_size_avg = 0
                self.fwd_seg_size_min = 0
        
        # Similar calculations for backward packets
        if self.backward_packets:
            bwd_lengths = [p.length for p in self.backward_packets]
            bwd_data_lengths = [p.length - p.ip_len for p in self.backward_packets if p.ip_len <= p.length]
            
            self.bwd_packet_length_max = max(bwd_lengths)
            self.bwd_packet_length_min = min(bwd_lengths)
            self.bwd_packet_length_mean = np.mean(bwd_lengths)
            self.bwd_packet_length_std = np.std(bwd_lengths)
            
            if bwd_data_lengths:
                self.bwd_segment_size_avg = np.mean(bwd_data_lengths)
            else:
                self.bwd_segment_size_avg = 0

        # Calculate overall packet statistics
        all_packets = self.forward_packets + self.backward_packets
        if all_packets:
            all_lengths = [p.length for p in all_packets]
            self.packet_length_min = min(all_lengths)
            self.packet_length_max = max(all_lengths)
            self.packet_length_mean = np.mean(all_lengths)
            self.packet_length_std = np.std(all_lengths)
            self.packet_length_variance = np.var(all_lengths)
            self.average_packet_size = np.mean(all_lengths)

    def _calculate_bulk_stats(self) -> None:
        """Calculate bulk transfer statistics
        Bulk transfer is defined as consecutive packets in the same direction
        carrying a significant amount of data (>1000 bytes)
        """
        BULK_THRESHOLD = 1000  # bytes
        
        def analyze_bulk(packets):
            bulk_count = 0
            bulk_size = 0
            bulk_duration = 0
            current_bulk_start = None
            current_bulk_size = 0
            current_bulk_packets = 0
            
            for i, packet in enumerate(packets):
                if packet.length > BULK_THRESHOLD:
                    if current_bulk_start is None:
                        current_bulk_start = packet.time
                    current_bulk_size += packet.length
                    current_bulk_packets += 1
                elif current_bulk_start is not None:
                    if current_bulk_packets > 0:
                        bulk_count += 1
                        bulk_size += current_bulk_size
                        bulk_duration += packet.time - current_bulk_start
                    current_bulk_start = None
                    current_bulk_size = 0
                    current_bulk_packets = 0
            
            return bulk_size/bulk_count if bulk_count > 0 else 0, \
                   current_bulk_packets/bulk_count if bulk_count > 0 else 0, \
                   bulk_size/bulk_duration if bulk_duration > 0 else 0

        # Calculate forward and backward bulk statistics
        self.fwd_bytes_bulk_avg, self.fwd_packet_bulk_avg, self.fwd_bulk_rate_avg = \
            analyze_bulk(self.forward_packets)
        self.bwd_bytes_bulk_avg, self.bwd_packet_bulk_avg, self.bwd_bulk_rate_avg = \
            analyze_bulk(self.backward_packets)

    def _calculate_subflow_stats(self) -> None:
        """Calculate subflow statistics
        A subflow is a subset of packets within the main flow separated by an idle timeout
        Default subflow timeout is 1.0 seconds
        """
        SUBFLOW_TIMEOUT = 1.0  # seconds
        
        def get_subflows(packets):
            if not packets:
                return 0, 0
                
            subflow_count = 1
            subflow_packets = 0
            subflow_bytes = 0
            last_time = packets[0].time
            
            for packet in packets:
                if packet.time - last_time > SUBFLOW_TIMEOUT:
                    subflow_count += 1
                subflow_packets += 1
                subflow_bytes += packet.length
                last_time = packet.time
            
            # Average per subflow
            return subflow_packets/subflow_count, subflow_bytes/subflow_count
        
        # Calculate forward and backward subflow statistics
        self.subflow_fwd_packets, self.subflow_fwd_bytes = get_subflows(self.forward_packets)
        self.subflow_bwd_packets, self.subflow_bwd_bytes = get_subflows(self.backward_packets)

    def _calculate_ratios(self) -> None:
        """Calculate flow ratios with proper error handling"""
        try:
            # Down/Up ratio
            if self.total_length_of_fwd_packet > 0:
                self.down_up_ratio = self.total_length_of_bwd_packet / self.total_length_of_fwd_packet
            else:
                self.down_up_ratio = 0
            
            # Average packet sizes
            total_packets = self.total_fwd_packets + self.total_backward_packets
            total_bytes = self.total_length_of_fwd_packet + self.total_length_of_bwd_packet
            
            if total_packets > 0:
                self.average_packet_size = total_bytes / total_packets
            else:
                self.average_packet_size = 0
                
            # Segment sizes
            if self.total_fwd_packets > 0:
                self.fwd_segment_size_avg = self.total_length_of_fwd_packet / self.total_fwd_packets
            if self.total_backward_packets > 0:
                self.bwd_segment_size_avg = self.total_length_of_bwd_packet / self.total_backward_packets
                
        except ZeroDivisionError:
            self.down_up_ratio = 0
            self.average_packet_size = 0
            self.fwd_segment_size_avg = 0
            self.bwd_segment_size_avg = 0

    def _calculate_iat_stats(self) -> None:
        """Calculate IAT (Inter-Arrival Time) statistics with preprocessing"""
        if self.fwd_iat_times:
            self.fwd_iat_mean = preprocess_value(np.mean(self.fwd_iat_times))
            self.fwd_iat_std = preprocess_value(np.std(self.fwd_iat_times))
            self.fwd_iat_max = preprocess_value(max(self.fwd_iat_times))
            self.fwd_iat_min = preprocess_value(min(self.fwd_iat_times))
        else:
            self.fwd_iat_mean = 0
            self.fwd_iat_std = 0
            self.fwd_iat_max = 0
            self.fwd_iat_min = 0

        if self.bwd_iat_times:
            self.bwd_iat_mean = preprocess_value(np.mean(self.bwd_iat_times))
            self.bwd_iat_std = preprocess_value(np.std(self.bwd_iat_times))
            self.bwd_iat_max = preprocess_value(max(self.bwd_iat_times))
            self.bwd_iat_min = preprocess_value(min(self.bwd_iat_times))
        else:
            self.bwd_iat_mean = 0
            self.bwd_iat_std = 0
            self.bwd_iat_max = 0
            self.bwd_iat_min = 0

    def _update_iat_stats(self, packet: Packet, is_forward: bool) -> None:
        """Update IAT stats ensuring proper time calculations"""
        current_time = packet.time
        
        if hasattr(self, 'last_seen'):
            iat = current_time - self.last_seen
            
            # Update flow IAT stats
            if iat > 0:  # Only update if time difference is positive
                self.flow_iat_min = min(self.flow_iat_min, iat)
                self.flow_iat_max = max(self.flow_iat_max, iat)
            
            # Forward IAT stats
            if is_forward and len(self.forward_packets) > 1:
                fwd_iat = current_time - self.forward_packets[-2].time
                if fwd_iat > 0:
                    self.fwd_iat_total += fwd_iat
                    self.fwd_iat_times.append(fwd_iat)
                    self.fwd_iat_min = min(self.fwd_iat_min, fwd_iat)
                    self.fwd_iat_max = max(self.fwd_iat_max, fwd_iat)
            
            # Backward IAT stats
            elif not is_forward and len(self.backward_packets) > 1:
                bwd_iat = current_time - self.backward_packets[-2].time
                if bwd_iat > 0:
                    self.bwd_iat_total += bwd_iat
                    self.bwd_iat_times.append(bwd_iat)
                    self.bwd_iat_min = min(self.bwd_iat_min, bwd_iat)
                    self.bwd_iat_max = max(self.bwd_iat_max, bwd_iat)
        
        self.last_seen = current_time

    def _update_flags(self, packet: Packet, is_forward: bool) -> None:
        """Update TCP flag statistics"""
        if not packet.flags:
            return
        
        flags = str(packet.flags)
        
        # Update direction-specific flags
        if is_forward:
            if 'P' in flags:
                self.fwd_psh_flags += 1
            if 'U' in flags:
                self.fwd_urg_flags += 1
        else:
            if 'P' in flags:
                self.bwd_psh_flags += 1
            if 'U' in flags:
                self.bwd_urg_flags += 1
        
        # Update total flag counts
        if 'F' in flags:
            self.fin_flag_count += 1
        if 'S' in flags:
            self.syn_flag_count += 1
        if 'R' in flags:
            self.rst_flag_count += 1
        if 'P' in flags:
            self.psh_flag_count += 1
        if 'A' in flags:
            self.ack_flag_count += 1
        if 'U' in flags:
            self.urg_flag_count += 1
        if 'C' in flags:
            self.cwe_flag_count += 1
        if 'E' in flags:
            self.ece_flag_count += 1

    def _update_activity_stats(self, current_time: float) -> None:
        """Update flow activity statistics"""
        # Calculate idle time since last packet
        idle_time = current_time - self.last_active_time
        
        # If idle time exceeds threshold, end current active period
        if idle_time > self.idle_threshold:
            active_time = self.last_active_time - self.active_start_time
            if active_time > 0:
                self.active_times.append(active_time)
            self.idle_times.append(idle_time)
            self.active_start_time = current_time
            
        self.last_active_time = current_time
        
        # Update activity statistics if we have samples
        if self.active_times:
            self.active_mean = np.mean(self.active_times)
            self.active_std = np.std(self.active_times)
            self.active_max = max(self.active_times)
            self.active_min = min(self.active_times)
            
        if self.idle_times:
            self.idle_mean = np.mean(self.idle_times)
            self.idle_std = np.std(self.idle_times)
            self.idle_max = max(self.idle_times)
            self.idle_min = min(self.idle_times)

    def cleanup(self) -> None:
        """Clean up resources"""
        self.forward_packets.clear()
        self.backward_packets.clear()
        self.active_times.clear()
        self.idle_times.clear()
