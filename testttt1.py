import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(F.shape[0])  # Initial state covariance
        self.x = np.zeros((F.shape[0], 1))  # Initial state

    def predict(self):
        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Compute association probabilities
        association_probabilities = self.compute_association_probabilities(S)

        # Update state based on association probabilities
        predicted_states = []
        for i, prob in enumerate(association_probabilities):
            K = np.dot(np.dot(self.P, self.H.T), inv(S))
            x_i = self.x + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)
            predicted_states.append(x_i)
        
        # Select the predicted state with highest association probability
        max_prob_index = np.argmax(association_probabilities)
        self.x = predicted_states[max_prob_index]

    def compute_association_probabilities(self, S):
        # Compute association probabilities using Mahalanobis distance
        num_measurements = self.H.shape[0]
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        association_probabilities = np.zeros(num_measurements)
        for i in range(num_measurements):
            d = np.dot(np.dot((self.H[i] - np.dot(self.H[i], self.x)).T, inv_S), (self.H[i] - np.dot(self.H[i], self.x)))
            association_probabilities[i] = np.exp(-0.5 * d) / ((2 * np.pi) ** (self.H.shape[0] / 2) * np.sqrt(det_S))
        return association_probabilities

def main():
    # Provided predicted values
    predicted_values = [
        [94805.44, 217.89, 2.0831, 21486.916],
        [27177.54, 153.5201, 2.086, 21487.166],
        [85834.72, 226.6823, 4.7109, 21487.189],
        [26591.4, 120.7162, 1.3585, 21487.24],
        [67521.98, 295.1252, 2.8341, 21487.256],
        [64726.5, 341.2639, 4.6564, 21487.332],
        [24220.79, 89.6023, 3.1265, 21487.369],
        [3768.37, 129.798, 12.6818, 21487.381],
        [20474.44, 27.3968, 0.6826, 21487.557],
        [94854.33, 217.6161, 2.4473, 21487.693],
        [27184.91, 153.4323, 1.7675, 21487.863],
        [64715.11, 341.1737, 4.6514, 21487.971],
        [70434.91, 325.4155, 3.0297, 21488.012],
        [26844.95, 301.2844, 4.9459, 21488.039],
        [80301.8, 352.2547, 4.7756, 21488.08],
        [87872.73, 46.1141, 6.5272, 21488.119],
        [66776.26, 104.3781, 3.9765, 21488.057]
    ]

    # Convert the provided values to NumPy array
    measurements = np.array(predicted_values)

    # Define state transition matrix
    F = np.eye(4)  # Assume constant velocity model for simplicity

    # Define measurement matrix
    H = np.eye(4)  # Identity matrix since measurement directly reflects state

    # Define process noise covariance matrix
    Q = np.eye(4) * 0.1  # Process noise covariance

    # Define measurement noise covariance matrix
    R = np.eye(4) * 0.01  # Measurement noise covariance, adjusted variance

    # Initialize Kalman filter
    kf = KalmanFilter(F, H, Q, R)

    # Lists to store predicted values for all variables
    predicted_ranges = []
    predicted_azimuths = []
    predicted_elevations = []
    predicted_times = []

    # Predict and update for each measurement
    for i, z in enumerate(measurements, start=1):
        # Predict
        kf.predict()

        # Update with measurement
        kf.update(z[:, np.newaxis])

        # Get predicted state
        predicted_state = kf.x.squeeze()

        # Append predicted values for all variables
        predicted_ranges.append(predicted_state[0])
        predicted_azimuths.append(predicted_state[1])
        predicted_elevations.append(predicted_state[2])
        predicted_times.append(z[3])  # Use the provided time value

        # Print predicted values
        print(f"Measurement {i}:")
        print("Predicted Range:", predicted_state[0])
        print("Predicted Azimuth:", predicted_state[1])
        print("Predicted Elevation:", predicted_state[2])
        print()  # Add an empty line for separation

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot measured and predicted ranges
    plt.plot(predicted_times, measurements[:, 0], label='Measured Range', marker='o')
    plt.plot(predicted_times, predicted_ranges, label='Predicted Range', linestyle='--', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Range')
    plt.title('Range Prediction vs. Range Measurement')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Plot measured and predicted azimuths
    plt.plot(predicted_times, measurements[:, 1], label='Measured Azimuth', marker='o')
    plt.plot(predicted_times, predicted_azimuths, label='Predicted Azimuth', linestyle='--', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Azimuth')
    plt.title('Azimuth Prediction vs. Azimuth Measurement')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Plot measured and predicted elevations
    plt.plot(predicted_times, measurements[:, 2], label='Measured Elevation', marker='o')
    plt.plot(predicted_times, predicted_elevations, label='Predicted Elevation', linestyle='--', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Elevation')
    plt.title('Elevation Prediction vs. Elevation Measurement')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
