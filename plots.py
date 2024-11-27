import matplotlib.pyplot as plt
import numpy as np
import os
from sanification import *
from PIL import Image

def plot_phase_processing(phase_matrix, file_name, directory, subcarrier_index=0):
    """
    Plots the raw, sanitized, and median-filtered phases of a single subcarrier from the phase matrix.
    
    Parameters:
    - phase_matrix: 2D numpy array of phase values (rows: packets or time frames, columns: subcarriers).
    - subcarrier_index: Index of the subcarrier to visualize (default: 0).
    - clip_range: Range around the mean for clipping outliers (default: 0.5).
    - median_filter_size: Size of the window for the median filter (default: 5).
    """

    # Define the path for the heatmap image
    plot_path = os.path.join(directory, f"{file_name}_phase_plot.png")
    
    # Check if the plot already exists
    if os.path.exists(plot_path):
        print(f"Plot already exists at: {plot_path}")
        return  # Exit the function if the plot already exists
    
    # Extract raw phase data for the specified subcarrier
    raw_phase_data = phase_matrix[:, subcarrier_index]

    # Sanitize the phase data
    sanitized_phase_data = sanitize_phase_matrix(phase_matrix, np.arange(0, phase_matrix.shape[1]))[:, subcarrier_index]


    # Step 1: Plot Raw Phase Data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1, polar=True)
    plt.scatter(raw_phase_data, np.ones_like(raw_phase_data), color='blue', s=10)
    plt.title("(a) Raw Phase")

    # Step 2: Plot Sanitized Phase Data
    plt.subplot(1, 3, 2, polar=True)
    plt.scatter(sanitized_phase_data, np.ones_like(sanitized_phase_data), color='red', s=10)
    plt.title("(b) Sanitized Phase")

    # Save the plot
    # plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    
    # Display the path where the heatmap was saved
    print(f"Plot saved at: {plot_path}")

    plt.show(block=False)  # Show plot without blocking code execution
    plt.pause(2)           # Pause for 2 seconds
    plt.close()    


def heatmap_plot_processing(amplitude_matrix, file_name, directory):

    # Define the path for the heatmap image
    heatmap_path = os.path.join(directory, f"{file_name}_heatmap.png")
    
    # Check if the plot already exists
    if os.path.exists(heatmap_path):
        print(f"Plot already exists at: {heatmap_path}")
        return
    
    # Generate the heatmap
    plt.imshow(amplitude_matrix, cmap='jet', interpolation='nearest', aspect='auto')
    
    # Add color bar to show the amplitude scale
    plt.colorbar(label='Amplitude')
    
    # Labeling
    plt.xlabel('Subcarriers')
    plt.ylabel('Packets')
    plt.title('CSI Amplitude Heatmap: ' + file_name)
    
    # Save the plot
    plt.savefig(heatmap_path, format='png', dpi=300, bbox_inches='tight')
    
    # Display the path where the heatmap was saved
    print(f"Plot saved at: {heatmap_path}")
    
    # Show the plot briefly
    plt.show(block=False)  # Show plot without blocking code execution
    plt.pause(2)           # Pause for 2 seconds
    plt.close()            # Close the plot window
    

# This function takes the heatmap amplitude as matrix (already sanitized) and convert it into a (224, 224) RGB image
def heatmap_vgg16(amplitude_matrix): 
    plt.imshow(amplitude_matrix, cmap='jet')  # Use a color map like 'viridis' or 'hot'
    plt.axis('off')

    # saving at the same path the png will change after each csv processing
    plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0)  # Save as image without axes

    # Load saved image and prepare as input for VGG-16 -> the resolution it's always 125x369?
    heatmap_image = Image.open('heatmap.png').convert('RGB')  # Convert to 3-channel RGB image
    heatmap_image = heatmap_image.resize((224, 224))  # Resize to 224x224 for VGG-16

    
    return heatmap_image