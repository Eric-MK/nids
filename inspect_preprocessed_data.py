import pickle
import numpy as np
from util_functions import feature_list

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def inspect_data(data, num_samples=5):
    print(f"Total number of flows: {len(data)}")
    
    feature_names = list(feature_list.keys())
    
    for i, (flow_id, flow_data) in enumerate(data[:num_samples]):
        print(f"\nFlow {i+1}:")
        print(f"Flow ID: {flow_id}")
        print(f"Label: {flow_data['label']}")
        
        for time_window, packets in flow_data.items():
            if time_window != 'label':
                print(f"\nTime window: {time_window}")
                print(f"Number of packets: {len(packets)}")
                print("Features of first packet:")
                for j, feature_name in enumerate(feature_names):
                    if j < len(packets[0]):
                        print(f"{feature_name}: {packets[0][j]}")
                    else:
                        print(f"{feature_name}: Not present in data")
                if len(packets[0]) > len(feature_names):
                    for j in range(len(feature_names), len(packets[0])):
                        print(f"Unknown feature {j}: {packets[0][j]}")
                break  # Only show the first time window for brevity

def main():
    file_path = './data/pcapmini/10t-10n-DOS2019-preprocess.data'
    data = load_preprocessed_data(file_path)
    inspect_data(data)

if __name__ == "__main__":
    main()