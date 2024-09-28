import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from util_functions import feature_list

def load_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        X_cnn = np.array(hf['X_cnn'])
        y = np.array(hf['y'])
    return X_cnn, y

def trim_outliers(data, lower_percentile=1, upper_percentile=99):
    lower, upper = np.percentile(data, [lower_percentile, upper_percentile])
    return np.clip(data, lower, upper)

def plot_feature_distributions(X, y, feature_names):
    n_features = X.shape[2]
    fig, axes = plt.subplots(n_features, 2, figsize=(20, 5*n_features))
    for i in range(n_features):
        benign_data = trim_outliers(X[y == 0, :, i].flatten())
        ddos_data = trim_outliers(X[y == 1, :, i].flatten())
        
        sns.histplot(benign_data, ax=axes[i, 0], kde=True, color='blue', label='Benign', bins=50, stat='density')
        sns.histplot(ddos_data, ax=axes[i, 0], kde=True, color='red', label='DDoS', bins=50, stat='density')
        axes[i, 0].set_title(f'{feature_names[i]} Distribution')
        axes[i, 0].legend()
        
        sns.boxplot(data=[benign_data, ddos_data], ax=axes[i, 1])
        axes[i, 1].set_xticklabels(['Benign', 'DDoS'])
        axes[i, 1].set_title(f'{feature_names[i]} Box Plot')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def plot_correlation_heatmap(X):
    print(f"Shape of X: {X.shape}")
    if X.ndim != 2:
        print(f"Warning: Expected 2D array, got {X.ndim}D. Reshaping...")
        X = X.reshape(X.shape[0], -1)
    print(f"Shape after reshape: {X.shape}")
    
    try:
        corr_matrix = np.corrcoef(X.T)
        print(f"Shape of correlation matrix: {corr_matrix.shape}")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                    xticklabels=feature_list.keys(), yticklabels=feature_list.keys())
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_correlation_heatmap: {str(e)}")

def plot_tsne(X, y, n_samples=5000):
    indices = np.random.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    X_flat = X_sample.reshape(X_sample.shape[0], -1)
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_flat)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample, palette={0: 'blue', 1: 'red'})
    plt.title('t-SNE Visualization of Flows (Sampled)')
    plt.legend(['Benign', 'DDoS'])
    plt.savefig('tsne_visualization.png')
    plt.close()

def main():
    file_path = './output_cnn_lstm/10t-10n-DOS2019-dataset-train-cnn-lstm.hdf5'
    X, y = load_data(file_path)
    
    feature_names = list(feature_list.keys())
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    plot_feature_distributions(X, y, feature_names)
    plot_correlation_heatmap(X[:, 0, :])  # Use first packet of each flow for correlation
    plot_tsne(X, y)
    
    print("Total samples:", X.shape[0])
    print("Benign samples:", np.sum(y == 0))
    print("DDoS samples:", np.sum(y == 1))
    print("Packets per flow:", X.shape[1])
    print("Number of features:", X.shape[2])

if __name__ == "__main__":
    main()