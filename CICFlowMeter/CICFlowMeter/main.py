from scapy.all import sniff, conf
import argparse
import signal
import sys
import os
from .flow_meter import FlowMeter

# Global variables
running = True
DEFAULT_TIMEOUT = 120  # Default flow timeout in seconds
DEFAULT_OUTPUT = 'flows.csv'
DEFAULT_PACKET_COUNT = 0  # 0 means infinite

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping packet capture...")
    running = False

def packet_callback(packet, flow_meter):
    """Process packet and update progress"""
    global running
    if running:
        try:
            flow_meter.process_packet(packet)
            # Update progress less frequently and with cleaner format
            if flow_meter.packet_count % 100 == 0:
                print(f'\rProcessing packets: {flow_meter.packet_count:,} ({flow_meter.valid_packet_count:,} valid)', end='')
        except Exception as e:
            print(f"\nError processing packet: {e}")
    # Only return True when we need to stop
    if not running:
        return True

def list_interfaces():
    """List all available network interfaces with details"""
    print("\nAvailable Network Interfaces:")
    print("-----------------------------")
    for iface in conf.ifaces.values():
        # Add more interface details
        description = getattr(iface, 'description', 'No description')
        mac = getattr(iface, 'mac', 'No MAC')
        print(f"- Name: {iface.name}")
        print(f"  Description: {description}")
        print(f"  MAC: {mac}")
    print()

def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description='CICFlowMeter-Python - Network Traffic Flow Feature Extractor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--interface', help='Network interface to capture')
    parser.add_argument('-f', '--file', help='PCAP file to process')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT, help='Output CSV file')
    parser.add_argument('-l', '--list', action='store_true', help='List available interfaces')
    parser.add_argument('-c', '--count', type=int, default=DEFAULT_PACKET_COUNT, 
                       help='Number of packets to capture (0 = infinite)')
    parser.add_argument('-t', '--timeout', type=int, default=DEFAULT_TIMEOUT,
                       help='Flow timeout in seconds')
    parser.add_argument('-m', '--model',
                       help='Path to trained CyberBERT model for real-time classification')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list and not (args.interface or args.file):
        parser.error("Either --interface or --file must be specified")
    
    if args.file and not os.path.exists(args.file):
        parser.error(f"PCAP file not found: {args.file}")
        
    return args

def main():
    """Main program entry point"""
    try:
        args = parse_arguments()

        if args.list:
            list_interfaces()
            return 0

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize flow meter with optional model
        flow_meter = FlowMeter(timeout=args.timeout, model_path=args.model)
        print("\nCICFlowMeter-Python")
        print("-------------------")
        print("Starting packet capture... Press Ctrl+C to stop\n")

        try:
            if args.file:
                print(f"Processing PCAP file: {args.file}")
                sniff(offline=args.file, 
                     prn=lambda x: packet_callback(x, flow_meter),
                     store=False,
                     count=args.count)
            elif args.interface:
                print(f"Capturing on interface: {args.interface}")
                sniff(iface=args.interface, 
                     prn=lambda x: packet_callback(x, flow_meter),
                     store=False,
                     stop_filter=lambda x: not running,
                     count=args.count)
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        except Exception as e:
            print(f"\nError during capture: {e}")
            return 1

        # Print final statistics
        stats = flow_meter.get_stats()
        print("\nCapture Statistics:")
        print(f"- Total packets: {stats['total_packets']}")
        print(f"- Valid packets: {stats['valid_packets']}")
        print(f"- Active flows: {stats['active_flows']}")
        print(f"- Completed flows: {stats['completed_flows']}")

        # Save flows
        if flow_meter.save_flows(args.output):
            print(f"\nFlows saved to: {args.output}")
            return 0
        else:
            print("\nNo flows were captured")
            return 1

    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
