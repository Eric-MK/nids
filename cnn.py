import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import h5py
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = "./data/DOS2019_pcaps/"
MODEL_PATH = "./models/"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001

def load_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        X = np.array(hf['set_x'])
        y = np.array(hf['set_y'])
    return X, y

def create_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, 'best_model.keras'), 
                                       save_best_only=True, monitor='val_accuracy')

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint])
    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

def main():
    # Load data
    X_train, y_train = load_data(os.path.join(DATA_PATH, "10t-100n-DOS2019-dataset-train.hdf5"))
    X_val, y_val = load_data(os.path.join(DATA_PATH, "10t-100n-DOS2019-dataset-val.hdf5"))
    X_test, y_test = load_data(os.path.join(DATA_PATH, "10t-100n-DOS2019-dataset-test.hdf5"))

    print("Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Create and train the model
    model = create_model(X_train.shape[1:])
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()