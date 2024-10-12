from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import numpy as np
from datetime import datetime, timedelta
from models import load_or_train_model
from data_processing import load_data, get_initial_sequence
from anomaly_detection import RealtimeAdaptiveExponentialSmoothing
from data_generation import data_stream_generator
import threading

app = Flask(__name__)
socketio = SocketIO(app)

background_thread = None
thread_lock = threading.Lock()

# Global variables
trained_model = None
initial_sequence = None
anomaly_model = None
data_generator = None
start_time = None
noise_level = 0.5
introduce_spike = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_simulation')
def start_simulation():
    global background_thread, start_time, noise_level, anomaly_model
    noise_level = float(request.args.get('noise_level', 0.5))
    with thread_lock:
        if background_thread is None:
            start_time = datetime.now()
            # Update anomaly threshold based on initial noise level
            anomaly_model.anomaly_threshold = max(1.5, 2.0 - noise_level * 0.1)
            background_thread = socketio.start_background_task(background_task)
    return jsonify({"status": "Simulation started"})

@socketio.on('update_noise_level')
def handle_noise_update(data):
    global noise_level, anomaly_model
    noise_level = data['noise_level']
    # Update anomaly threshold based on new noise level
    anomaly_model.anomaly_threshold = max(1.5, 2.0 - noise_level * 0.1)
    print(f"Noise level updated to: {noise_level}, Anomaly threshold: {anomaly_model.anomaly_threshold}")

def background_task():
    global initial_sequence, start_time, noise_level, introduce_spike
    data_point_count = 0
    while True:
        try:
            wind_speed = next(data_generator)
            
            # Add noise based on the current noise level
            wind_speed += np.random.normal(0, noise_level * 2)  # Increased noise effect
            
            # Introduce spike if requested
            if introduce_spike:
                wind_speed += np.random.choice([-1, 1]) * np.random.uniform(5, 10)
                introduce_spike = False
            
            model_input = np.array(initial_sequence).reshape(1, 6, 1)
            lstm_forecast = trained_model.predict(model_input, verbose=0)[0][0]
            
            forecast, error, is_anomaly = anomaly_model.update(wind_speed)
            
            current_time = start_time + timedelta(minutes=10 * data_point_count)
            
            socketio.emit('new_data_point', {
                'time': current_time.isoformat(),
                'actual': float(wind_speed),
                'forecast': float(lstm_forecast),
                'is_anomaly': bool(is_anomaly)
            })
            
            initial_sequence = np.roll(initial_sequence, -1)
            initial_sequence[-1] = wind_speed
            
            data_point_count += 1
            print(f"Emitted data point {data_point_count}: Time={current_time.isoformat()}, Actual={wind_speed}, Forecast={lstm_forecast}, Anomaly={is_anomaly}")
            
            socketio.sleep(5)  # Wait for 5 seconds before next update
        except Exception as e:
            print(f"Error in background task: {e}")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('introduce_spike')
def handle_introduce_spike():
    global introduce_spike
    introduce_spike = True
    print("Spike will be introduced in the next data point")

if __name__ == '__main__':
    data = load_data()
    trained_model = load_or_train_model(data)
    initial_sequence = get_initial_sequence(data)
    
    anomaly_model = RealtimeAdaptiveExponentialSmoothing(anomaly_threshold=3.0)
    data_generator = data_stream_generator(initial_sequence)
    
    print("\nServer is ready. Click 'Start Simulation' to begin.")
    socketio.run(app, debug=True)