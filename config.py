import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'wind_speed_lstm_model_retrained.h5')
DATA_FILE = r'scaled_turbine1.csv' #PATH TO DATA
COLUMN_NAME = 'Wind speed (m/s)'