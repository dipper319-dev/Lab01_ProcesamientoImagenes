import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data from deteccion.py
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Calculate velocity and acceleration
def calculate_kinematics(data):
    time = np.array(data['time'])
    positions = np.array(data['positions'])  # Assuming 'positions' key holds position data

    # Calculate velocity and acceleration
    velocity = np.diff(positions) / np.diff(time)
    acceleration = np.diff(velocity) / np.diff(time[:-1])

    return velocity, acceleration

# Identify type of movement based on the velocity
def identify_movement(velocity):
    if np.all(velocity >= 0):
        return 'Accelerating'
    elif np.all(velocity <= 0):
        return 'Decelerating'
    else:
        return 'Variable'

# Generate graphs
def generate_graphs(positions, velocity, acceleration):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(positions, label='Position')
    plt.title('Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Position')
    
    plt.subplot(1, 3, 2)
    plt.plot(velocity, label='Velocity', color='orange')
    plt.title('Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity')

    plt.subplot(1, 3, 3)
    plt.plot(acceleration, label='Acceleration', color='green')
    plt.title('Acceleration vs Time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')

    plt.tight_layout()
    plt.show()

# Save results to CSV
def save_results_to_csv(velocity, acceleration, filename):
    df = pd.DataFrame({
        'Velocity': np.concatenate(([0], velocity)),  # Adding initial velocity
        'Acceleration': np.concatenate(([0], [0], acceleration))  # Adding initial acceleration
    })
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    data = load_data('deteccion.json')  # Adjust the filename as necessary
    velocity, acceleration = calculate_kinematics(data)
    movement_type = identify_movement(velocity)
    print(f'Movement Type: {movement_type}')
    
    generate_graphs(data['positions'], velocity, acceleration)
    save_results_to_csv(velocity, acceleration, 'results.csv')