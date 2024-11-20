

import csv
import numpy as np

from csi_data import *
from scipy.ndimage import gaussian_filter
from cnn import *

# libraries for clustering
from elbow import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

from kmeans import kmeans_algorithm

N = 128 # from 384 values to 128 (csi data)
n_packets = 200 # re-dimension the raw data
subcarriers = 64
all_signatures = []
all_labels = []

def calculate_signature(csv_original):

    # initialize the amplitude and phase matrix
    amplitude_matrix = []
    phase_matrix = []
    directory = os.path.dirname(csv_original)
    
    # read the csv file
    file_name = csv_original.split("/")[-1][0:-4]
    csv_original = open(csv_original, mode = 'r')

    # create a new csv file and save the csi_data rows inside
    csi_data_csv = open(directory + "/" + (file_name + "_csi_data.csv"), mode = "w", newline='')
    csv_writer = csv.writer(csi_data_csv)

    # creating the csi_data csv
    for i, line in enumerate(csv_original):
        try:
            if i == 0: # intestation row
                continue 

            line = line.split("\"")

            # delete this line if you work with original csv and uncomment csi_data = line[-2][1:-1].split(",")
            line = line[0].split(",")


            if len(line) < 2:
                print(f"Skipping row {i} due to unexpected format: {line}")
                continue

            # csi_data = line[-2][1:-1].split(",")

            # calculate_raw_amplitudes_and_phases returns the amplitude and phases for each trasmitted packets
            packet_amplitudes, packet_phases = calculate_raw_amplitudes_and_phases(line[10:N], subcarriers) # do not consider the initial 5 values (img, real)

            # add the packet amplitudes to amplitude_matrix
            amplitude_matrix.append(packet_amplitudes)

            # add the packet phases to phase_matrix
            phase_matrix.append(packet_phases)

            # format again in str and save only 128 values into new csv
            result = ",".join(line[0:N])
            csv_writer.writerow([result])
        
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # close the files
    csv_original.close()
    csi_data_csv.close()

    # sanitize the amplitude matrix -> delete outliers and apply median filter
    sanitized_amplitude_matrix = sanitize_amplitude_matrix(np.array(amplitude_matrix))
    smooth_matrix = gaussian_filter(sanitized_amplitude_matrix, sigma=1.5)  # Additional smoothing step

    # plot heatmap
    # heatmap_plot_processing(smooth_matrix[0:n_packets], file_name, directory)

    # convert the image into a 224x224 png for the vgg-16 method
    heatmap_resized = heatmap_vgg16(smooth_matrix[0:n_packets])

    # calculate the feature vector with a cnn vgg-16 and convert it into numpy array
    vgg_features = feature_map_vector(heatmap_resized).detach().numpy().flatten()
    signature = list(vgg_features) # create a signature

    for i in range(np.array(phase_matrix).shape[1]):  # Loop through each subcarrier (column)
        subcarrier_values = np.array(phase_matrix)[:, i]  # Extract column (subcarrier data)

        # Apply your sanitization function here
        _, filtered_subcarrier = sanitize_phase_data(subcarrier_values)

        stats = [np.mean(filtered_subcarrier),
                np.var(filtered_subcarrier),
                skew(filtered_subcarrier),
                kurtosis(filtered_subcarrier),
                np.min(filtered_subcarrier),
                np.max(filtered_subcarrier),
                np.std(filtered_subcarrier)
        ]

        signature += stats

    # phase plot (original, sanitized, filtered) with index subcarrier = 0
    # plot_phase_processing(np.array(phase_matrix), file_name, directory)

    return signature

def process_all_csv_in_folder(folder_path):
    """
    Function to iterate over all CSV files in a folder and process them.
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return  # Folder doesn't exist, return early

    # Check if any filtered file already exists in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_csi_data.csv"):
            print(f"Filtered file already exists: {filename}")
            continue  # Skip to the next file

    # If a csi_data_csv file exists, do not stop but continue processing other files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".csv") and not filename.endswith("_csi_data.csv"):

            signature = calculate_signature(file_path)  # return the signature for a single csv_file
            all_signatures.append(signature) # creating the matrix with all the signatures
            all_labels.append(file_path.split("/")[-2:-1][0] + "-" + filename.split("/")[-1])
            print(all_labels)

            print(f"Processing {file_path}")

        elif os.path.isdir(file_path):
            process_all_csv_in_folder(file_path)


if __name__ == "__main__":

    # process_all_csv_in_folder("/Users/sebastiandinu/Desktop/Tesi-Triennale/Dataset Re-Id") # change with your dataset's path

    # Replace NaNs values with zeros
    # signatures = np.nan_to_num(np.array(all_signatures), nan=0.0, posinf=0.0, neginf=0.0)

    # save the signatures on a txt file
    # np.savetxt("signatures.txt", signatures, fmt='%f', delimiter=',')
    # np.savetxt("labels.txt", np.array(all_labels), fmt='%s')

    signatures = np.loadtxt("signatures.txt", delimiter=',')

    # Read the file to identify problematic rows
    with open("signatures.txt", "r") as file:
        for i, line in enumerate(file, start=1):
            columns = line.strip().split()
            print(f"Row {i}: {len(columns)} columns")

    print(signatures.shape)
    # clustering code
    # Normalize the features
    scaler = StandardScaler()
    signatures_normalized = scaler.fit_transform(signatures)

    # Apply PCA Principal Component Analysis
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(signatures_normalized) # return a numpy vector shape(poses, n_components)

    # this method shows the elbow graph to decide the best number of clusters
    elbow_method_plot(pca_features)

    # shows kmeans algorithm plot
    kmeans_algorithm(pca_features, 10)