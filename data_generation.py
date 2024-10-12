import numpy as np

def generate_wind_speed(time, base=10, amplitude=3, noise_level=0.5, seasonal_amplitude=2, seasonal_period=24):
    wind = base + amplitude * np.sin(2 * np.pi * time / 24)
    wind += seasonal_amplitude * np.sin(2 * np.pi * time / (seasonal_period * 24))
    wind += np.random.normal(0, noise_level)
    return max(0, wind)

def data_stream_generator(initial_sequence, interval=10):
    time_counter = 0
    current_sequence = initial_sequence.copy()
    
    while True:
        next_point = generate_wind_speed(time_counter)
        yield next_point
        
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_point
        
        time_counter += interval / 3600  # Convert seconds to hours