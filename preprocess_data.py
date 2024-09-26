import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(packet_file, flow_file):
    # Load the datasets
    packet_df = pd.read_csv(packet_file)
    flow_df = pd.read_csv(flow_file)

    # Function to preprocess a dataframe
    def preprocess_df(df):
        # Remove any duplicate rows
        df = df.drop_duplicates()

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        # For numeric columns, impute with median
        imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # For categorical columns, impute with most frequent value
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

        return df

    # Preprocess both dataframes
    packet_df = preprocess_df(packet_df)
    flow_df = preprocess_df(flow_df)

    # Separate features and labels
    packet_labels = packet_df['label']
    packet_features = packet_df.drop(['label', 'src_ip', 'dst_ip', 'timestamp'], axis=1)

    flow_labels = flow_df['label']
    flow_features = flow_df.drop(['label', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)

    # Convert categorical variables to one-hot encoding
    packet_features = pd.get_dummies(packet_features)
    flow_features = pd.get_dummies(flow_features)

    # Normalize numerical features
    packet_scaler = StandardScaler()
    flow_scaler = StandardScaler()

    packet_features_scaled = packet_scaler.fit_transform(packet_features)
    flow_features_scaled = flow_scaler.fit_transform(flow_features)

    # Split the data into training and testing sets
    packet_X_train, packet_X_test, packet_y_train, packet_y_test = train_test_split(
        packet_features_scaled, packet_labels, test_size=0.2, random_state=42)

    flow_X_train, flow_X_test, flow_y_train, flow_y_test = train_test_split(
        flow_features_scaled, flow_labels, test_size=0.2, random_state=42)

    return (packet_X_train, packet_X_test, packet_y_train, packet_y_test, 
            flow_X_train, flow_X_test, flow_y_train, flow_y_test,
            packet_scaler, flow_scaler)

# Usage
packet_file = 'ddos_packet_dataset.csv'
flow_file = 'ddos_flow_dataset.csv'

(packet_X_train, packet_X_test, packet_y_train, packet_y_test, 
 flow_X_train, flow_X_test, flow_y_train, flow_y_test,
 packet_scaler, flow_scaler) = load_and_preprocess_data(packet_file, flow_file)

print("Packet dataset shape:", packet_X_train.shape)
print("Flow dataset shape:", flow_X_train.shape)

# Save preprocessed data
np.save('packet_X_train.npy', packet_X_train)
np.save('packet_X_test.npy', packet_X_test)
np.save('packet_y_train.npy', packet_y_train)
np.save('packet_y_test.npy', packet_y_test)
np.save('flow_X_train.npy', flow_X_train)
np.save('flow_X_test.npy', flow_X_test)
np.save('flow_y_train.npy', flow_y_train)
np.save('flow_y_test.npy', flow_y_test)

print("Preprocessed data saved to .npy files")