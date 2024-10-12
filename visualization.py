from collections import deque
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class RealtimePlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.actual_values = deque(maxlen=max_points)
        self.forecast_values = deque(maxlen=max_points)
        self.anomalies = deque(maxlen=max_points)
        
        self.fig = make_subplots(rows=1, cols=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual', line=dict(color='blue')))
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
        
        self.fig.update_layout(
            title='Real-time Wind Speed Anomaly Detection',
            xaxis_title='Time',
            yaxis_title='Wind Speed (scaled)',
            height=600
        )
    
    def update(self):
        self.fig.data[0].x = list(self.times)
        self.fig.data[0].y = list(self.actual_values)
        
        self.fig.data[1].x = list(self.times)
        self.fig.data[1].y = list(self.forecast_values)
        
        anomaly_times = [t for t, a in zip(self.times, self.anomalies) if a]
        anomaly_values = [v for v, a in zip(self.actual_values, self.anomalies) if a]
        self.fig.data[2].x = anomaly_times
        self.fig.data[2].y = anomaly_values
        
        self.fig.update_xaxes(type='date')
        return self.fig
    
    def add_point(self, time, actual, forecast, is_anomaly):
        self.times.append(time)
        self.actual_values.append(actual)
        self.forecast_values.append(forecast)
        self.anomalies.append(is_anomaly)