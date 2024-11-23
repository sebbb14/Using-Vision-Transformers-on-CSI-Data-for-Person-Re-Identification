import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def lstm_features(phase_matrix, lstm_units=50, output_dim=64):
    """
    Process a phase matrix with shape (packets, subcarriers) through an LSTM model.

    Parameters:
    - phase_matrix (numpy.ndarray): Input phase matrix of shape (packets, subcarriers).
    - lstm_units (int): Number of units in the LSTM layer.
    - output_dim (int): Dimension of the output feature vector.

    Returns:
    - feature_vector (numpy.ndarray): Extracted feature vector of shape (output_dim,).
    """
    # Validate the input
    if not isinstance(phase_matrix, np.ndarray):
        raise ValueError("phase_matrix must be a numpy ndarray.")
    if len(phase_matrix.shape) != 2:
        raise ValueError("phase_matrix must have shape (packets, subcarriers).")

    # Reshape to 3D for a single sample
    phase_matrix_3d = np.expand_dims(phase_matrix, axis=0)  # Shape: (1, packets, subcarriers)

    # Define the LSTM model
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(phase_matrix_3d.shape[1], phase_matrix_3d.shape[2])),
        Dense(output_dim)  # Output layer for feature vector
    ])

    # Compile the model (dummy compilation as no training is performed here)
    model.compile(optimizer='adam', loss='mse')

    # Pass the phase matrix through the model
    feature_vector = model.predict(phase_matrix_3d)

    # Return the feature vector for the single input
    return feature_vector[0]


# Example: Generate a dummy phase matrix (packets=20, subcarriers=64)
# packets, subcarriers = 20, 64
# phase_matrix = np.random.randn(packets, subcarriers)  # Shape: (packets, subcarriers)

# Process the phase matrix
# output_dim = 64  # Desired output feature vector dimension
# feature_vector = lstm_features(phase_matrix)

# Output the resulting feature vector
# print("Feature Vector Shape:", feature_vector.shape)
# print("Feature Vector:", type(feature_vector))
