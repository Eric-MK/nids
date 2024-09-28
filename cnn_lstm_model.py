import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LSTM, Dense, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from HDF5 files
def load_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        X_cnn = np.array(hf['X_cnn'])
        X_lstm = np.array(hf['X_lstm'])
        y = np.array(hf['y'])
    return X_cnn, X_lstm, y

# Load the datasets
train_file = './output_cnn_lstm/10t-100n-DOS2019-dataset-train-cnn-lstm.hdf5'
val_file = './output_cnn_lstm/10t-100n-DOS2019-dataset-val-cnn-lstm.hdf5'
test_file = './output_cnn_lstm/10t-100n-DOS2019-dataset-test-cnn-lstm.hdf5'

X_train_cnn, X_train_lstm, y_train = load_data(train_file)
X_val_cnn, X_val_lstm, y_val = load_data(val_file)
X_test_cnn, X_test_lstm, y_test = load_data(test_file)

print("CNN input shape:", X_train_cnn.shape)
print("LSTM input shape:", X_train_lstm.shape)

# Define the CNN+LSTM hybrid model
def create_cnn_lstm_model(input_shape_cnn, input_shape_lstm):
    # CNN part
    cnn_input = Input(shape=input_shape_cnn, name='cnn_input')
    x = Conv2D(64, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # LSTM part
    lstm_input = Input(shape=input_shape_lstm, name='lstm_input')
    y = LSTM(64, return_sequences=True)(lstm_input)
    y = LSTM(64)(y)

    # Combine CNN and LSTM
    combined = concatenate([x, y])
    z = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[cnn_input, lstm_input], outputs=output)
    return model

# Create and compile the model
input_shape_cnn = X_train_cnn.shape[1:]
input_shape_lstm = X_train_lstm.shape[1:]
model = create_cnn_lstm_model(input_shape_cnn, input_shape_lstm)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Train the model
history = model.fit(
    [X_train_cnn, X_train_lstm], y_train,
    validation_data=([X_val_cnn, X_val_lstm], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test_cnn, X_test_lstm], y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict([X_test_cnn, X_test_lstm])
y_pred_classes = (y_pred > 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')

# Plot training history
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')

print("Training complete. Model evaluated and results saved.")