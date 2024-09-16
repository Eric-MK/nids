import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the simulation data
simulation_data = pd.read_csv('data/processed/ddos_simulation_data.csv')

# Prepare the features (X) and labels (y) from the simulation data
X_sim = simulation_data.drop(' Label', axis=1)
y_sim = simulation_data[' Label']

# Create a new scaler and fit it to the simulation data
# Note: This is not ideal, but it's a workaround since we don't have the original scaler
scaler = StandardScaler()
X_sim_scaled = scaler.fit_transform(X_sim)

# Reshape the input for the LSTM model (assuming your model expects 3D input)
X_sim_reshaped = X_sim_scaled.reshape((X_sim_scaled.shape[0], 1, X_sim_scaled.shape[1]))

print(f"Simulation data shape: {X_sim_reshaped.shape}")

# Load the saved model
model = load_model('models/anomallyrnnlstmmodel.keras')  # Replace with your actual model filename

# Make predictions
y_sim_pred = model.predict(X_sim_reshaped)
y_sim_pred_classes = (y_sim_pred > 0.5).astype(int)

# Evaluate the model's performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Model Performance on Simulation Data:")
print(f"Accuracy: {accuracy_score(y_sim, y_sim_pred_classes):.4f}")
print("\nClassification Report:")
print(classification_report(y_sim, y_sim_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_sim, y_sim_pred_classes))

tn, fp, fn, tp = confusion_matrix(y_sim, y_sim_pred_classes).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1_score = 2 * (precision * recall) / (precision + recall)

print("\nDetailed Metrics:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1_score:.4f}")