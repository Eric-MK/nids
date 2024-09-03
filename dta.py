import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def sample_from_csv(file_path, sample_size=300000, chunk_size=10000):
    benign_samples = []
    attack_samples = []
    total_rows = 0
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)
        
        # Separate BENIGN and attack samples
        benign_chunk = chunk[chunk[' Label'] == 'BENIGN']
        attack_chunk = chunk[chunk[' Label'] != 'BENIGN']
        
        benign_samples.append(benign_chunk)
        attack_samples.append(attack_chunk)
    
    # Combine all benign and attack samples
    benign_data = pd.concat(benign_samples, ignore_index=True)
    attack_data = pd.concat(attack_samples, ignore_index=True)
    
    # If there are fewer than sample_size rows, return all rows
    if total_rows <= sample_size:
        return pd.concat([benign_data, attack_data], ignore_index=True)
    
    # Calculate how many attack samples we need
    attack_sample_size = sample_size - len(benign_data)
    
    # If we need more attack samples than available, take all attack samples
    if attack_sample_size >= len(attack_data):
        return pd.concat([benign_data, attack_data], ignore_index=True)
    
    # Otherwise, sample from attack samples
    sampled_attacks = attack_data.sample(n=attack_sample_size, random_state=42)
    
    # Combine BENIGN and sampled attack data
    return pd.concat([benign_data, sampled_attacks], ignore_index=True)

def process_directory(directory, total_sample_size=300000):
    all_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            
            sampled_data = sample_from_csv(file_path, total_sample_size)
            all_data.append(sampled_data)
    
    # Combine all sampled data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Split into training and simulation sets
    train_data, simulation_data = train_test_split(combined_data, 
                                                   test_size=100000, 
                                                   stratify=combined_data[' Label'],
                                                   random_state=42)
    
    return train_data, simulation_data

# Usage
data_directory = 'data/raw/01-12'  # Replace with your actual directory path
train_data, simulation_data = process_directory(data_directory)

# Print information about the datasets
print(f"Training data shape: {train_data.shape}")
print("Label distribution in training data:")
print(train_data[' Label'].value_counts(normalize=True))

print(f"\nSimulation data shape: {simulation_data.shape}")
print("Label distribution in simulation data:")
print(simulation_data[' Label'].value_counts(normalize=True))

# Save the datasets
train_data.to_csv('ddos_train_data.csv', index=False)
simulation_data.to_csv('ddos_simulation_data.csv', index=False)
print("\nTraining dataset saved to 'ddos_train_data.csv'")
print("Simulation dataset saved to 'ddos_simulation_data.csv'")