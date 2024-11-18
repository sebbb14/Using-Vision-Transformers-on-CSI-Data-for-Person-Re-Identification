

import numpy as np
from scipy.ndimage import median_filter

def sanitize_amplitude_matrix(amplitude_matrix, clip_stddev=3, filter_size=5):
    """
    Sanitizes a matrix of amplitude values by removing outliers and applying a median filter.
    
    Parameters:
    - amplitude_matrix: 2D numpy array of amplitude values (rows: time frames or antenna pairs, columns: subcarriers)
    - clip_stddev: Standard deviation multiplier for outlier clipping (default: 3)
    - filter_size: Size of the median filter window (default: 3)
    
    Returns:
    - sanitized_matrix: 2D numpy array of sanitized amplitude values
    """

    # Step 1: Outlier Removal (Clipping)
    mean = np.mean(amplitude_matrix)
    std = np.std(amplitude_matrix)
    lower_bound = mean - clip_stddev * std
    upper_bound = mean + clip_stddev * std
    
    # Clip values outside the range
    clipped_matrix = np.clip(amplitude_matrix, lower_bound, upper_bound)
    
    # Step 2: Median Filtering
    # Apply median filter row-wise to reduce noise while preserving the main trends
    sanitized_matrix = median_filter(clipped_matrix, size=(1, filter_size))

    return sanitized_matrix


def sanitize_phase_data(phase_data, clip_range=0.5, median_filter_size=5):
    """
    Sanitizes phase data by clipping outliers and applying a median filter.
    
    Parameters:
    - phase_data: 1D numpy array of phase values for a single subcarrier over time.
    - clip_range: Range around the mean for clipping outliers (default: 0.5).
    - median_filter_size: Size of the window for the median filter (default: 5).
    
    Returns:
    - sanitized_phase_data: Phase data after clipping.
    - filtered_phase_data: Phase data after applying median filter.
    """
    # Step 1: Clip phase data around the mean to reduce outliers
    mean_phase = np.mean(phase_data)
    sanitized_phase_data = np.clip(phase_data, mean_phase - clip_range, mean_phase + clip_range)
    
    # Step 2: Apply median filter to smooth the phase data
    filtered_phase_data = median_filter(sanitized_phase_data, size=median_filter_size)
    
    return sanitized_phase_data, filtered_phase_data
