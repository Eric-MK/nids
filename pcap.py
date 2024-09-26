import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from collections import Counter
import statistics
import math

# Configuration for attack IPs and victims
DOS2019_FLOWS = {'attackers': ['172.16.0.5'], 'victims': ['192.168.50.1', '192.168.50.4']}

# Path to pcap files
pcap_path = './data/pcapmini/'

def calculate_entropy(values):
    """Calculate Shannon entropy for a list of values."""
    counter = Counter(values)
    total = sum(counter.values())
    probabilities = [count / total for count in counter.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def extract_packet_features(pkt):
    features = {}
    features['timestamp'] = float(pkt.time)

    if IP in pkt:
        ip_src = pkt[IP].src
        ip_dst = pkt[IP].dst
        features['src_ip'] = ip_src
        features['dst_ip'] = ip_dst
        features['ip_len'] = int(pkt[IP].len)
        features['ttl'] = int(pkt[IP].ttl)
        features['ip_flags'] = int(pkt[IP].flags)
        features['ip_frag'] = int(pkt[IP].frag)

        features['label'] = 1 if ip_src in DOS2019_FLOWS['attackers'] and ip_dst in DOS2019_FLOWS['victims'] else 0

    if TCP in pkt:
        features['protocol'] = 'TCP'
        features['src_port'] = int(pkt[TCP].sport)
        features['dst_port'] = int(pkt[TCP].dport)
        features['flags'] = str(pkt[TCP].flags)
        features['window_size'] = int(pkt[TCP].window)
        features['tcp_ack'] = int(pkt[TCP].ack)
        features['tcp_seq'] = int(pkt[TCP].seq)
        features['tcp_dataofs'] = int(pkt[TCP].dataofs)
        features['tcp_urgptr'] = int(pkt[TCP].urgptr)
    elif UDP in pkt:
        features['protocol'] = 'UDP'
        features['src_port'] = int(pkt[UDP].sport)
        features['dst_port'] = int(pkt[UDP].dport)
        features['udp_len'] = int(pkt[UDP].len)
    elif ICMP in pkt:
        features['protocol'] = 'ICMP'
        features['icmp_type'] = int(pkt[ICMP].type)
        features['icmp_code'] = int(pkt[ICMP].code)
    else:
        features['protocol'] = 'Other'

    features['pkt_len'] = len(pkt)
    features['payload_len'] = len(pkt[IP].payload) if IP in pkt else 0
    
    return features

def extract_flow_features(packets):
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

            if flow_id not in flows:
                flows[flow_id] = {
                    'start_time': float(pkt.time),
                    'end_time': float(pkt.time),
                    'packet_count': 0,
                    'total_bytes': 0,
                    'packet_lengths': [],
                    'inter_arrival_times': [],
                    'last_packet_time': float(pkt.time),
                    'tcp_flags': [],
                    'payload_sizes': [],
                }

            flow = flows[flow_id]
            flow['packet_count'] += 1
            flow['total_bytes'] += len(pkt)
            flow['packet_lengths'].append(len(pkt))
            inter_arrival_time = float(pkt.time) - flow['last_packet_time']
            if inter_arrival_time > 0:
                flow['inter_arrival_times'].append(inter_arrival_time)
            flow['end_time'] = float(pkt.time)
            flow['last_packet_time'] = float(pkt.time)
            
            if TCP in pkt:
                flow['tcp_flags'].append(str(pkt[TCP].flags))
            
            flow['payload_sizes'].append(len(pkt[IP].payload) if IP in pkt else 0)

    for flow_id, flow in flows.items():
        duration = flow['end_time'] - flow['start_time']
        avg_inter_arrival = statistics.mean(flow['inter_arrival_times']) if flow['inter_arrival_times'] else 0
        
        # Calculate entropy of packet lengths and payload sizes
        pkt_len_entropy = calculate_entropy(flow['packet_lengths'])
        payload_entropy = calculate_entropy(flow['payload_sizes'])
        
        # Calculate TCP flag distribution
        tcp_flag_dist = Counter(flow['tcp_flags'])
        syn_rate = tcp_flag_dist['S'] / flow['packet_count'] if 'S' in tcp_flag_dist else 0
        fin_rate = tcp_flag_dist['F'] / flow['packet_count'] if 'F' in tcp_flag_dist else 0
        
        flow_features.append({
            'src_ip': flow_id[0],
            'dst_ip': flow_id[1],
            'src_port': flow_id[2],
            'dst_port': flow_id[3],
            'protocol': flow_id[4],
            'flow_duration': duration,
            'packet_count': flow['packet_count'],
            'total_bytes': flow['total_bytes'],
            'avg_pkt_size': statistics.mean(flow['packet_lengths']),
            'std_pkt_size': statistics.stdev(flow['packet_lengths']) if len(flow['packet_lengths']) > 1 else 0,
            'pkt_len_entropy': pkt_len_entropy,
            'payload_entropy': payload_entropy,
            'pkt_per_sec': flow['packet_count'] / duration if duration > 0 else 0,
            'bytes_per_sec': flow['total_bytes'] / duration if duration > 0 else 0,
            'avg_inter_arrival': avg_inter_arrival,
            'syn_rate': syn_rate,
            'fin_rate': fin_rate,
            'label': 1 if flow_id[0] in DOS2019_FLOWS['attackers'] and flow_id[1] in DOS2019_FLOWS['victims'] else 0
        })

    return flow_features

def process_pcap_file(pcap_file):
    packets = rdpcap(pcap_file)
    packet_features = []
    
    for pkt in packets:
        try:
            pkt_features = extract_packet_features(pkt)
            packet_features.append(pkt_features)
        except Exception as e:
            print(f"Error processing packet: {e}")

    flow_features = extract_flow_features(packets)
    
    return packet_features, flow_features

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
    
    packet_df = pd.DataFrame(all_packet_features)
    flow_df = pd.DataFrame(all_flow_features)
    
    packet_output_file = 'ddos_packet_dataset.csv'
    flow_output_file = 'ddos_flow_dataset.csv'
    packet_df.to_csv(packet_output_file, index=False)
    flow_df.to_csv(flow_output_file, index=False)

    print(f"Packet-level dataset saved to {packet_output_file}")
    print(f"Flow-level dataset saved to {flow_output_file}")

# Run the process
generate_dataset_from_pcap(pcap_path)