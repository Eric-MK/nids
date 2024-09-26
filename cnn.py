import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
import glob
import h5py
import argparse
import time
from collections import OrderedDict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, Dropout, GlobalMaxPooling2D, LSTM, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Constants
SEED = 1
MAX_FLOW_LEN = 100
TIME_WINDOW = 10
TRAIN_SIZE = 0.90
OUTPUT_FOLDER = "./output/"
PATIENCE = 10
DEFAULT_EPOCHS = 1000

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

# Configure TensorFlow
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True

# Hyperparameters
hyperparameters = {
    "learning_rate": [0.1, 0.01],
    "batch_size": [1024, 2048],
    "kernels": [32, 64],
    "lstm_units": [32, 64],
    "regularization": [None, 'l1'],
    "dropout": [None, 0.2]
}

def LSTMCNNModel(model_name, input_shape, kernel_col, lstm_units=64, kernels=64, kernel_rows=3, learning_rate=0.01, regularization=None, dropout=None):
    tf.keras.backend.clear_session()

    model = Sequential(name=model_name)
    regularizer = regularization

    # CNN layers
    model.add(Conv2D(kernels, (kernel_rows, kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Activation('relu'))
    
    # Reshape for LSTM
    model.add(Reshape((input_shape[0], -1)))  # Flatten the CNN output for each time step

    # LSTM layer
    model.add(LSTM(lstm_units, return_sequences=True, name='lstm0'))
    
    # Additional layers
    model.add(GlobalMaxPooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid', name='fc1'))

    print(model.summary())
    compileModel(model, learning_rate)
    return model

def compileModel(model, lr):
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def load_dataset(path):
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])
    set_y_orig = np.array(dataset["set_y"][:])
    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig
    return X_train, Y_train

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])
    if Y_true is not None and len(Y_true.shape) > 0:
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1),
               'TPR': '{:05.4f}'.format(tpr), 'FPR': '{:05.4f}'.format(fpr), 'TNR': '{:05.4f}'.format(tnr), 'FNR': '{:05.4f}'.format(fnr), 'Source': data_source}
    else:
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
               'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source}
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)

def main(argv):
    parser = argparse.ArgumentParser(description='DDoS detection with LSTM-CNN hybrid model')
    parser.add_argument('-t', '--train', nargs='+', type=str, help='Start the training process')
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int, help='Training iterations')
    parser.add_argument('-cv', '--cross_validation', default=0, type=int, help='Number of folds for cross-validation')
    parser.add_argument('-p', '--predict', nargs='?', type=str, help='Perform a prediction on pre-preprocessed data')
    parser.add_argument('-m', '--model', type=str, help='File containing the model')
    args = parser.parse_args()

    if args.train is not None:
        for dataset_folder in args.train:
            X_train, Y_train = load_dataset(dataset_folder + "/*-train.hdf5")
            X_val, Y_val = load_dataset(dataset_folder + "/*-val.hdf5")

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            train_file = glob.glob(dataset_folder + "/*-train.hdf5")[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print(f"\nCurrent dataset folder: {dataset_folder}")

            model_name = f"{dataset_name}-LSTM-CNN-HYBRID"
            keras_classifier = KerasClassifier(
                build_fn=LSTMCNNModel,
                model_name=model_name,
                input_shape=X_train.shape[1:],
                kernel_col=X_train.shape[2]
            )

            grid_search_cv = GridSearchCV(
                keras_classifier,
                hyperparameters,
                cv=args.cross_validation if args.cross_validation > 1 else [(slice(None), slice(None))],
                refit=True,
                return_train_score=True
            )

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            best_model_filename = f"{OUTPUT_FOLDER}{time_window}t-{max_flow_len}n-{model_name}"
            mc = ModelCheckpoint(f"{best_model_filename}.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

            grid_search_cv.fit(X_train, Y_train, epochs=args.epochs, validation_data=(X_val, Y_val), callbacks=[es, mc])

            best_model = grid_search_cv.best_estimator_.model
            best_model.save(f"{best_model_filename}.h5")

            Y_pred_val = (best_model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            print(f"Best parameters: {grid_search_cv.best_params_}")
            print(f"Best model path: {best_model_filename}")
            print(f"F1 Score of the best model on the validation set: {f1_score_val}")

    if args.predict is not None:
        predict_file = open(f"{OUTPUT_FOLDER}predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv", 'w', newline='')
        predict_writer = csv.DictWriter(predict_file, fieldnames=['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR', 'TNR', 'FNR', 'Source'])
        predict_writer.writeheader()

        dataset_filelist = glob.glob(f"{args.predict}/*test.hdf5")
        model_list = [args.model] if args.model is not None else glob.glob(f"{args.predict}/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            model_name_string = model_filename.split('-')[-1].strip().split('.')[0].strip()
            model = load_model(model_path)

            for dataset_file in dataset_filelist:
                X, Y = load_dataset(dataset_file)
                
                pt0 = time.time()
                Y_pred = (model.predict(X, batch_size=2048) > 0.5).squeeze()
                pt1 = time.time()
                prediction_time = pt1 - pt0

                report_results(Y, Y_pred, X.shape[0], model_name_string, dataset_file, prediction_time, predict_writer)

        predict_file.close()

if __name__ == "__main__":
    main([])