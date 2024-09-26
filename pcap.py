import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from datetime import datetime

# Configuration for attack IPs and victims
DOS2019_FLOWS = {'attackers': ['172.16.0.5'], 'victims': ['192.168.50.1', '192.168.50.4']}

# Path to pcap files
pcap_path = './data/pcapmini/'

# Feature extraction function
def extract_packet_features(pkt):
    """Extracts comprehensive features from a packet."""
    features = {}

    # Timestamp
    features['timestamp'] = pkt.time

    # IP Layer
    if IP in pkt:
        ip_src = pkt[IP].src
        ip_dst = pkt[IP].dst
        features['src_ip'] = ip_src
        features['dst_ip'] = ip_dst
        features['ip_len'] = pkt[IP].len
        features['ttl'] = pkt[IP].ttl

        # Labeling the packet
        if ip_src in DOS2019_FLOWS['attackers'] and ip_dst in DOS2019_FLOWS['victims']:
            features['label'] = 1  # Attack
        else:
            features['label'] = 0  # Normal

    # Transport Layer (TCP/UDP)
    if TCP in pkt:
        features['protocol'] = 'TCP'
        features['src_port'] = pkt[TCP].sport
        features['dst_port'] = pkt[TCP].dport
        features['flags'] = str(pkt[TCP].flags)  # TCP flags
        features['window_size'] = pkt[TCP].window  # TCP window size
        features['tcp_ack'] = pkt[TCP].ack
        features['tcp_seq'] = pkt[TCP].seq
    elif UDP in pkt:
        features['protocol'] = 'UDP'
        features['src_port'] = pkt[UDP].sport
        features['dst_port'] = pkt[UDP].dport
    elif ICMP in pkt:
        features['protocol'] = 'ICMP'
        features['icmp_type'] = pkt[ICMP].type
        features['icmp_code'] = pkt[ICMP].code
    else:
        features['protocol'] = 'Other'

    # Packet length and payload size
    features['pkt_len'] = len(pkt)
    features['payload_len'] = len(pkt[IP].payload) if IP in pkt else 0

    return features

# Extract flow-level features (e.g., total packets, bytes, inter-arrival time, etc.)
def extract_flow_features(packets):
    """Aggregate flow-level features from the captured packets."""
    flows = {}
    flow_features = []

    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else None
            dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else None
            protocol = pkt[TCP].flags if TCP in pkt else 'UDP' if UDP in pkt else 'Other'

            flow_id = (src_ip, dst_ip, src_port, dst_port, protocol)

            # Initialize flow if new
            if flow_id not in flows:
                flows[flow_id] = {
                    'start_time': pkt.time,
                    'end_time': pkt.time,
                    'packet_count': 0,
                    'total_bytes': 0,
                    'packet_lengths': [],
                    'inter_arrival_times': [],
                    'last_packet_time': pkt.time,
                }

            # Update flow
            flow = flows[flow_id]
            flow['packet_count'] += 1
            flow['total_bytes'] += len(pkt)
            flow['packet_lengths'].append(len(pkt))
            inter_arrival_time = pkt.time - flow['last_packet_time']
            if inter_arrival_time > 0:
                flow['inter_arrival_times'].append(inter_arrival_time)
            flow['end_time'] = pkt.time
            flow['last_packet_time'] = pkt.time

    # Extract flow-level features
    for flow_id, flow in flows.items():
        duration = flow['end_time'] - flow['start_time']
        if flow['inter_arrival_times']:
            avg_inter_arrival = sum(flow['inter_arrival_times']) / len(flow['inter_arrival_times'])
        else:
            avg_inter_arrival = 0
        
        flow_features.append({
            'src_ip': flow_id[0],
            'dst_ip': flow_id[1],
            'src_port': flow_id[2],
            'dst_port': flow_id[3],
            'protocol': flow_id[4],
            'flow_duration': duration,
            'packet_count': flow['packet_count'],
            'total_bytes': flow['total_bytes'],
            'avg_pkt_size': sum(flow['packet_lengths']) / len(flow['packet_lengths']) if flow['packet_lengths'] else 0,
            'std_pkt_size': pd.Series(flow['packet_lengths']).std() if len(flow['packet_lengths']) > 1 else 0,
            'pkt_per_sec': flow['packet_count'] / duration if duration > 0 else 0,
            'bytes_per_sec': flow['total_bytes'] / duration if duration > 0 else 0,
            'avg_inter_arrival': avg_inter_arrival,
            'label': 1 if flow_id[0] in DOS2019_FLOWS['attackers'] and flow_id[1] in DOS2019_FLOWS['victims'] else 0
        })

    return flow_features

# Process each pcap file and extract features
def process_pcap_file(pcap_file):
    packets = rdpcap(pcap_file)
    packet_features = []
    flow_features = []

    for pkt in packets:
        try:
            # Extract packet-level features
            pkt_features = extract_packet_features(pkt)
            packet_features.append(pkt_features)
        except Exception as e:
            print(f"Error processing packet: {e}")

    # Extract flow-level features
    flow_features = extract_flow_features(packets)
    
    return packet_features, flow_features

# Main function to process all pcap files in a directory
def generate_dataset_from_pcap(pcap_dir):
    all_packet_features = []
    all_flow_features = []

    for file_name in os.listdir(pcap_dir):
        if file_name.endswith('.pcap'):
            pcap_file_path = os.path.join(pcap_dir, file_name)
            print(f"Processing {pcap_file_path} ...")
            packet_features, flow_features = process_pcap_file(pcap_file_path)
            all_packet_features.extend(packet_features)
            all_flow_features.extend(flow_features)
    
    # Convert packet and flow features to DataFrame
    packet_df = pd.DataFrame(all_packet_features)
    flow_df = pd.DataFrame(all_flow_features)
    
    # Save datasets to CSV
    packet_output_file = 'ddos_packet_dataset.csv'
    flow_output_file = 'ddos_flow_dataset.csv'
    packet_df.to_csv(packet_output_file, index=False)
    flow_df.to_csv(flow_output_file, index=False)

    print(f"Packet-level dataset saved to {packet_output_file}")
    print(f"Flow-level dataset saved to {flow_output_file}")

# Run the process
generate_dataset_from_pcap(pcap_path)
