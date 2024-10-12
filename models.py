import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import os
from config import MODEL_FILE

def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def create_model(lookback=6):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_mse)
    return model

def prepare_data(data, lookback=6):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def train_model(model, data, epochs=20, batch_size=32):
    X, y = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return model

def load_or_train_model(data):
    if os.path.exists(MODEL_FILE):
        print("Loading existing model...")
        try:
            custom_objects = {'custom_mse': custom_mse}
            return load_model(MODEL_FILE, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
    else:
        print("Training new model...")
    
    model = create_model()
    model = train_model(model, data)
    model.save(MODEL_FILE)
    print("Model trained and saved.")
    return model