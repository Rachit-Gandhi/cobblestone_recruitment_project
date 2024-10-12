import pandas as pd
from config import DATA_FILE, COLUMN_NAME

def load_data():
    df = pd.read_csv(DATA_FILE)
    return df[COLUMN_NAME].values

def get_initial_sequence(data, lookback=6):
    return data[-lookback:]