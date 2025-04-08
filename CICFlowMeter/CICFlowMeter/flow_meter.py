import time
import os
import sqlite3
from scapy.layers.inet import TCP, UDP
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import sys
from .features import FlowFeatures
from .utils import PacketInfo, preprocess_value

class FlowMeter:
    """Manages network traffic flow metering and feature extraction"""
    
    def __init__(self, timeout=120, db_path='flows.db'):
        # Remove database deletion
        self.active_flows = {}  # Changed from defaultdict to regular dict
        self.completed_flows = []
        self.timeout = timeout
        self.packet_count = 0
        self.valid_packet_count = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 10  # Cleanup every 10 seconds
        self.db_path = db_path
        self.update_interval = 1  # Update DB every second
        self.last_db_update = time.time()
        self.updated_flows = set()  # Initialize set to track flows needing updates
        self.processed_flows = set()  # Track flows we've already saved to DB
        self.flow_order = []  # Track order of flows
        self._setup_database()

    def _setup_database(self):
        """Initialize SQLite database and create table if not exists"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS network_flows (
                        "Flow ID" TEXT ,
                        "Timestamp" DATETIME,
                        "Src IP" TEXT,
                        "Src Port" REAL,
                        "Dst IP" TEXT,
                        "Dst Port" REAL,
                        "Protocol" TEXT,
                        "Flow Duration" REAL,
                        "Total Fwd Packets" REAL,
                        "Total Bwd packets" REAL,
                        "Total Length of Fwd Packet" REAL,
                        "Total Length of Bwd Packet" REAL,
                        "Fwd Packet Length Max" REAL,
                        "Fwd Packet Length Min" REAL,
                        "Fwd Packet Length Mean" REAL,
                        "Fwd Packet Length Std" REAL,
                        "Bwd Packet Length Max" REAL,
                        "Bwd Packet Length Min" REAL,
                        "Bwd Packet Length Mean" REAL,
                        "Bwd Packet Length Std" REAL,
                        "Flow Bytes/s" REAL,
                        "Flow Packets/s" REAL,
                        "Flow IAT Mean" REAL,
                        "Flow IAT Std" REAL,
                        "Flow IAT Max" REAL,
                        "Flow IAT Min" REAL,
                        "Fwd IAT Total" REAL,
                        "Fwd IAT Mean" REAL,
                        "Fwd IAT Std" REAL,
                        "Fwd IAT Max" REAL,
                        "Fwd IAT Min" REAL,
                        "Bwd IAT Total" REAL,
                        "Bwd IAT Mean" REAL,
                        "Bwd IAT Std" REAL,
                        "Bwd IAT Max" REAL,
                        "Bwd IAT Min" REAL,
                        "Fwd PSH Flags" REAL,
                        "Bwd PSH Flags" REAL,
                        "Fwd URG Flags" REAL,
                        "Bwd URG Flags" REAL,
                        "Fwd Header Length" REAL,
                        "Bwd Header Length" REAL,
                        "Fwd Packets/s" REAL,
                        "Bwd Packets/s" REAL,
                        "Min Packet Length" REAL,
                        "Max Packet Length" REAL,
                        "Packet Length Mean" REAL,
                        "Packet Length Std" REAL,
                        "Packet Length Variance" REAL,
                        "FIN Flag Count" REAL,
                        "SYN Flag Count" REAL,
                        "RST Flag Count" REAL,
                        "PSH Flag Count" REAL,
                        "ACK Flag Count" REAL,
                        "URG Flag Count" REAL,
                        "CWR Flag Count" REAL,
                        "ECE Flag Count" REAL,
                        "Down/Up Ratio" REAL,
                        "Average Packet Size" REAL,
                        "Avg Fwd Segment Size" REAL,
                        "Avg Bwd Segment Size" REAL,
                        "Fwd Bytes/Bulk Avg" REAL,
                        "Fwd Packet/Bulk Avg" REAL,
                        "Fwd Bulk Rate Avg" REAL,
                        "Bwd Bytes/Bulk Avg" REAL,
                        "Bwd Packet/Bulk Avg" REAL,
                        "Bwd Bulk Rate Avg" REAL,
                        "Subflow Fwd Packets" REAL,
                        "Subflow Fwd Bytes" REAL,
                        "Subflow Bwd Packets" REAL,
                        "Subflow Bwd Bytes" REAL,
                        "FWD Init Win Bytes" REAL,
                        "Bwd Init Win Bytes" REAL,
                        "Fwd Act Data Pkts" REAL,
                        "Fwd Seg Size Min" REAL,
                        "Active Mean" REAL,
                        "Active Std" REAL,
                        "Active Max" REAL,
                        "Active Min" REAL,
                        "Idle Mean" REAL,
                        "Idle Std" REAL,
                        "Idle Max" REAL,
                        "Idle Min" REAL,
                        "Label" TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            print(f"Error setting up database: {str(e)}")

    def process_packet(self, packet):
        """Process a new packet and update flow statistics"""
        try:
            self.packet_count += 1
            
            # Convert raw packet to PacketInfo
            packet_info = PacketInfo(packet)
            
            if not packet_info.is_valid_flow():
                return
            
            self.valid_packet_count += 1
            flow_id = packet_info.get_flow_id()
            
            current_time = time.time()
            
            # Periodic cleanup of expired flows
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._check_timeouts()
                self.last_cleanup = current_time

            # Create or update flow
            if flow_id not in self.active_flows:
                self.active_flows[flow_id] = FlowFeatures(packet_info)
            else:
                self.active_flows[flow_id].update(packet_info)
            self.updated_flows.add(flow_id)  # Mark flow as needing update

            # Real-time database update every second
            if current_time - self.last_db_update > self.update_interval:
                self._update_active_flows_in_db()
                self.last_db_update = current_time
                self.updated_flows.clear()  # Clear after update

            if self.packet_count % 1000 == 0:
                print(f"\rProcessed: {self.packet_count} packets ({self.valid_packet_count} valid)", end='')
                sys.stdout.flush()
            
            if len(self.active_flows) % 100 == 0:
                print(f"\rActive flows: {len(self.active_flows)}", end="")
                sys.stdout.flush()
            
        except Exception as e:
            print(f"\nError processing packet: {str(e)}")

    def _check_timeouts(self):
        """Check and remove expired flows"""
        current_time = time.time()
        expired_flows = []
        
        for flow_id, flow in self.active_flows.items():
            if (current_time - flow.last_seen) > self.timeout:
                # Mark flow as needing update if not already processed
                if flow_id not in self.processed_flows:
                    self.updated_flows.add(flow_id)
                expired_flows.append(flow_id)

        # Update database before removing flows
        self._update_active_flows_in_db()

        # Remove expired flows
        for flow_id in expired_flows:
            self.active_flows[flow_id].cleanup()
            del self.active_flows[flow_id]

    def _update_active_flows_in_db(self):
        """Save current state of active flows to database, maintaining order"""
        try:
            total_flows = len(self.updated_flows)
            with tqdm(total=total_flows, desc="Processing flows", unit="flow") as pbar:
                for flow_id in self.updated_flows:
                    if flow_id in self.active_flows and flow_id not in self.processed_flows:
                        try:
                            flow = self.active_flows[flow_id]
                            flow._finalize_calculations()
                            features = flow.get_features()
                            
                            # Convert timestamp to datetime string format
                            features['Timestamp'] = pd.to_datetime(features['Timestamp'], unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')
                            
                            # Maintain flow order
                            if flow_id not in self.flow_order:
                                self.flow_order.append(flow_id)
                            
                            with sqlite3.connect(self.db_path) as conn:
                                # Configure SQLite to handle datetime properly
                                conn.execute("PRAGMA datetime_format = 'YYYY-MM-DD HH:mm:ss.SSS'")
                                cursor = conn.cursor()
                                
                                # Complete ordered column list matching CSV format
                                columns = [
                                    "Flow ID", "Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
                                    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", 
                                    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
                                    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                                    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
                                    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
                                    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
                                    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
                                    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                                    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
                                    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
                                    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
                                    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
                                    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
                                    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
                                    "URG Flag Count", "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio",
                                    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
                                    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
                                    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
                                    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
                                    "Subflow Bwd Bytes", "FWD Init Win Bytes", "Bwd Init Win Bytes",
                                    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std",
                                    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
                                    "Idle Min", "Label"
                                ]
                                
                                # Prepare values in correct order
                                values = [features[col] for col in columns]
                                placeholders = ','.join(['?' for _ in columns])
                                columns_sql = ','.join([f'"{col}"' for col in columns])
                                
                                cursor.execute(f'''
                                    INSERT INTO network_flows ({columns_sql})
                                    VALUES ({placeholders})
                                ''', values)
                                conn.commit()
                                
                                self.processed_flows.add(flow_id)
                                self.completed_flows.append(features)
                                pbar.update(1)
                                sys.stdout.flush()
                        except Exception as e:
                            print(f"\nError updating flow in database: {str(e)}")
        except Exception as e:
            print(f"\nError processing flows: {str(e)}")

    def save_flows(self, output_file: str) -> bool:
        """Save remaining flows to CSV file and ensure DB synchronization"""
        try:
            # Update any remaining flows
            remaining_flows = set(self.active_flows.keys()) - self.processed_flows
            self.updated_flows.update(remaining_flows)
            self._update_active_flows_in_db()
            
            if not self.completed_flows:
                print("No flows to save")
                return False
            
            # Create DataFrame preserving order
            df = pd.DataFrame(self.completed_flows)
            
            # Convert timestamp string back to datetime object if it's a string
            if df['Timestamp'].dtype == 'object':
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Save CSV with proper timestamp formatting
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            df.to_csv(output_file, index=False,
                     float_format='%.6f')
                     

            print(f"\nSuccessfully saved {len(df)} flows to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving flows: {str(e)}")
            return False

    def get_stats(self) -> dict:
        """Get current flow meter statistics"""
        return {
            'total_packets': self.packet_count,
            'valid_packets': self.valid_packet_count,
            'active_flows': len(self.active_flows),
            'completed_flows': len(self.completed_flows)
        }
