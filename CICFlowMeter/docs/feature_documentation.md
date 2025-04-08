# Network Flow Features Documentation (84 Features)

## Group 1: Basic Flow Information (7 features)
- `Flow ID`: Unique flow identifier {srcIP-dstIP-srcPort-dstPort-protocol}
- `Src IP`: Source IP address
- `Src Port`: Source port number
- `Dst IP`: Destination IP address
- `Dst Port`: Destination port number
- `Protocol`: Transport protocol (e.g., TCP=6, UDP=17)
- `Timestamp`: Start time of the flow

## Group 2: Duration and Packet Counts (5 features)
- `Flow Duration`: Duration of the flow in microseconds
- `Total Fwd Packets`: Total packets in forward direction
- `Total Backward Packets`: Total packets in backward direction
- `Total Length of Fwd Packets`: Total size of packets in forward direction (bytes)
- `Total Length of Bwd Packets`: Total size of packets in backward direction (bytes)

## Group 3: Packet Length Statistics (8 features)
- `Fwd Packet Length Max`: Maximum packet length in forward direction
- `Fwd Packet Length Min`: Minimum packet length in forward direction
- `Fwd Packet Length Mean`: Average packet length in forward direction
- `Fwd Packet Length Std`: Standard deviation of packet length in forward direction
- `Bwd Packet Length Max`: Maximum packet length in backward direction
- `Bwd Packet Length Min`: Minimum packet length in backward direction
- `Bwd Packet Length Mean`: Average packet length in backward direction
- `Bwd Packet Length Std`: Standard deviation of packet length in backward direction

## Group 4: Flow Rate Statistics (6 features)
- `Flow Bytes/s`: Number of flow bytes per second (scaled by 1,000,000 for precision)
- `Flow Packets/s`: Number of flow packets per second (scaled by 1,000,000 for precision)
- `Flow IAT Mean`: Mean time between two packets in the flow
- `Flow IAT Std`: Standard deviation of packet inter-arrival times
- `Flow IAT Max`: Maximum inter-arrival time between packets
- `Flow IAT Min`: Minimum inter-arrival time between packets

## Group 5: Inter-Arrival Time (IAT) Statistics (10 features)
Forward Direction:
- `Fwd IAT Total`: Total time between two packets in forward direction
- `Fwd IAT Mean`: Average time between two packets in forward direction
- `Fwd IAT Std`: Standard deviation time between two packets in forward direction
- `Fwd IAT Max`: Maximum time between two packets in forward direction
- `Fwd IAT Min`: Minimum time between two packets in forward direction

Backward Direction:
- `Bwd IAT Total`: Total time between two packets in backward direction
- `Bwd IAT Mean`: Average time between two packets in backward direction
- `Bwd IAT Std`: Standard deviation time between two packets in backward direction
- `Bwd IAT Max`: Maximum time between two packets in backward direction
- `Bwd IAT Min`: Minimum time between two packets in backward direction

## Group 6: TCP Flags Statistics (12 features)
- `Fwd PSH Flags`: Number of PSH flags in forward direction
- `Bwd PSH Flags`: Number of PSH flags in backward direction
- `Fwd URG Flags`: Number of URG flags in forward direction
- `Bwd URG Flags`: Number of URG flags in backward direction
- `FIN Flag Count`: Number of FIN flags
- `SYN Flag Count`: Number of SYN flags
- `RST Flag Count`: Number of RST flags
- `PSH Flag Count`: Number of PSH flags
- `ACK Flag Count`: Number of ACK flags
- `URG Flag Count`: Number of URG flags
- `CWE Flag Count`: Number of CWE flags
- `ECE Flag Count`: Number of ECE flags

## Group 7: Header and Packet Rate (4 features)
- `Fwd Header Length`: Total bytes used for headers in forward direction
- `Bwd Header Length`: Total bytes used for headers in backward direction
- `Fwd Packets/s`: Number of forward packets per second
- `Bwd Packets/s`: Number of backward packets per second

## Group 8: Packet Length Aggregates (5 features)
- `Min Packet Length`: Length of smallest packet in flow
- `Max Packet Length`: Length of largest packet in flow
- `Packet Length Mean`: Average length of all packets
- `Packet Length Std`: Standard deviation of packet lengths
- `Packet Length Variance`: Variance of packet lengths

## Group 9: Ratio and Segment Features (4 features)
- `Down/Up Ratio`: Download and upload ratio
- `Average Packet Size`: Average size of each packet in flow
- `Avg Fwd Segment Size`: Average segment size in forward direction
- `Avg Bwd Segment Size`: Average segment size in backward direction

## Group 10: Bulk Statistics (6 features)
These features analyze consecutive packets carrying significant data (>1000 bytes):
- `Fwd Avg Bytes/Bulk`: Average number of bytes in bulk in forward direction
- `Fwd Avg Packets/Bulk`: Average number of packets in bulk in forward direction
- `Fwd Avg Bulk Rate`: Average number of bulk bytes per second in forward direction
- `Bwd Avg Bytes/Bulk`: Average number of bytes in bulk in backward direction
- `Bwd Avg Packets/Bulk`: Average number of packets in bulk in backward direction
- `Bwd Avg Bulk Rate`: Average number of bulk bytes per second in backward direction

## Group 11: Subflow Statistics (4 features)
Subflows are portions of the flow divided by an idle timeout of 1.0 seconds:
- `Subflow Fwd Packets`: Average packets per subflow in forward direction
- `Subflow Fwd Bytes`: Average bytes per subflow in forward direction
- `Subflow Bwd Packets`: Average packets per subflow in backward direction
- `Subflow Bwd Bytes`: Average bytes per subflow in backward direction

## Group 12: TCP Window Statistics (2 features)
- `Init_Win_bytes_forward`: Initial window bytes in forward direction
- `Init_Win_bytes_backward`: Initial window bytes in backward direction

## Group 13: Additional Features (2 features)
- `Fwd Act Data Pkts`: Count of packets with at least 1 byte of TCP data
- `Fwd Seg Size Min`: Minimum segment size in forward direction

## Group 14: Active and Idle Statistics (8 features)
Active time is the time a flow was continuously sending data. Idle time is the time between active periods.
Threshold for idle time is 2.0 seconds.

Active Time:
- `Active Mean`: Mean time a flow was active before becoming idle
- `Active Std`: Standard deviation of active times
- `Active Max`: Longest active time
- `Active Min`: Shortest active time

Idle Time:
- `Idle Mean`: Mean time a flow was idle before becoming active
- `Idle Std`: Standard deviation of idle times
- `Idle Max`: Longest idle time
- `Idle Min`: Shortest idle time

## Group 15: Label (1 feature)
- `Label`: Traffic class label (default: "NeedManualLabel")

## Implementation Notes

1. All time-based features are in seconds unless specified otherwise
2. Flow duration is in microseconds for better precision
3. Rates (bytes/s, packets/s) are scaled by 1,000,000 for precision
4. Bulk transfer threshold is 1000 bytes
5. Idle threshold is 2.0 seconds
6. Forward/Backward direction is normalized (smaller IP/port is always source)

Total Features: 84
