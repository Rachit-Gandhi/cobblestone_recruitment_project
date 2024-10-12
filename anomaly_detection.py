from collections import deque
import numpy as np

class RealtimeAdaptiveExponentialSmoothing:
    def __init__(self, alpha=0.3, beta=0.1, anomaly_threshold=2.0, window_size=30):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        self.anomaly_threshold = anomaly_threshold
        self.forecast_errors = deque(maxlen=window_size)
        self.actual_values = deque(maxlen=window_size)
        self.window_size = window_size

    def initialize(self, initial_data):
        self.level = initial_data[0]
        self.trend = initial_data[1] - initial_data[0]

    def update(self, actual):
        if self.level is None:
            self.initialize([actual, actual])
            return actual, 0, False

        forecast = self.level + self.trend
        error = actual - forecast
        
        new_level = self.alpha * actual + (1 - self.alpha) * (self.level + self.trend)
        new_trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend
        
        self.level = new_level
        self.trend = new_trend
        
        self.forecast_errors.append(error)
        self.actual_values.append(actual)
        
        is_anomaly = self.is_anomaly(actual, error)
        
        self.adapt_parameters()
        
        return forecast, error, is_anomaly

    def is_anomaly(self, actual, error):
        if len(self.forecast_errors) < self.window_size:
            return False
        
        error_std = np.std(self.forecast_errors)
        value_std = np.std(self.actual_values)
        
        # Check for anomaly based on forecast error
        if abs(error) > self.anomaly_threshold * error_std:
            return True
        
        # Check for anomaly based on sudden change in value
        if len(self.actual_values) > 1:
            last_value = self.actual_values[-2]
            if abs(actual - last_value) > self.anomaly_threshold * value_std:
                return True
        
        return False

    def adapt_parameters(self):
        if len(self.forecast_errors) >= self.window_size:
            recent_errors = list(self.forecast_errors)[-10:]
            mape = np.mean(np.abs(recent_errors))
            self.alpha = max(0.1, min(0.9, 1 / (1 + np.exp(-mape + 5))))
            self.beta = self.alpha / 2