
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt

def sanitize_and_median_filter(data, window_size, filter_size):
    """
    Performs a two-step processing procedure:
    1. Detects and replaces outliers in CSI data (artifact reduction).
    2. Applies a median filter over the entire matrix.

    Parameters:
        data (ndarray): Input 2D data array (packets × subcarriers).
        window_size (int): Size of the sliding window for outlier detection.
        filter_size (int): Size of the window for median filtering.

    Returns:
        ndarray: Processed 2D data after artifact reduction and median filtering.
    """
    # Step 1: Outlier detection and replacement
    def sanitize_column(column, window_size):
        """
        Detects and replaces local outliers in a single column (subcarrier).
        """
        sanitized_column = column.copy()
        for i in range(len(column)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(column), i + window_size // 2 + 1)
            window = column[start_idx:end_idx]

            # Compute local median and MAD
            local_median = np.median(window)
            mad = np.median(np.abs(window - local_median))

            # Detect outlier
            if mad == 0:
                continue
            threshold = 3 * mad
            if abs(column[i] - local_median) > threshold:
                sanitized_column[i] = sanitized_column[i - 1] if i > 0 else local_median
        return sanitized_column

    # Apply sanitization column-wise
    data = np.array(data)
    for col_idx in range(data.shape[1]):
        data[:, col_idx] = sanitize_column(data[:, col_idx], window_size)

    # Step 2: Apply median filtering to the entire matrix
    processed_data = median_filter(data, size=filter_size, mode='reflect')

    return processed_data


def sanitize_phase_matrix(phase_matrix, subcarrier_indices, filter_kernel_size=3):
    """
    Perform phase sanitization on a 2D phase matrix by removing linear distortion (slope and offset)
    and applying median filtering.

    Args:
        phase_matrix (np.ndarray): 2D array (packets x subcarriers) of measured phases.
        subcarrier_indices (np.ndarray): 1D array of corresponding indices (m_k) for subcarriers.

    Returns:
        np.ndarray: Sanitized phase matrix after calibration and filtering.
    """

    # Ensure subcarrier indices are a numpy array
    subcarrier_indices = np.array(subcarrier_indices)

    # Initialize the sanitized matrix
    sanitized_matrix = np.zeros_like(phase_matrix)

    # Iterate over each row (packet) in the phase matrix
    for i, phases in enumerate(phase_matrix):
        # Compute the slope (a) for the current packet
        slope = (phases[-1] - phases[0]) / (subcarrier_indices[-1] - subcarrier_indices[0])

        # Compute the offset (b) for the current packet
        offset = np.mean(phases)

        # Calibrate the phases for the current packet
        calibrated_phases = phases - slope * subcarrier_indices - offset

        # Apply median filtering (using a simple 1D convolution for smoothing)
        sanitized_phases = medfilt(calibrated_phases, kernel_size=filter_kernel_size)

        # Store the sanitized phases in the matrix
        sanitized_matrix[i, :] = sanitized_phases

    return sanitized_matrix
