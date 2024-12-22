import csv
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from tensorboard.plugins.image.summary import image
import torch
from csi_data import *
from sanification import *
from plots import *

N = 128 # from 384 values to 128 (csi data)
n_packets = 200 # re-dimension the raw data
subcarriers = 64
all_phase_matrix = []
all_labels = []
all_heatmaps = []


# the function below updated the phase matrix changing the seq_lenght to max_len
def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        # Calculate how many zeros need to be added
        padding_size = max_len - seq.shape[0]

        if padding_size > 0:
            # Create a tensor of zeros to pad the sequence
            padding = torch.zeros(padding_size, seq.shape[1])  # (padding_size, feature_dim)
            padded_seq = torch.cat((seq, padding), dim=0)  # Append padding
        else:
            padded_seq = seq  # No padding needed if seq_len >= max_len
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences)

# the function below calculate the phase and amplitude matrix and sanitize them
def calculate_amplitude_phase_matrix(csv_original):

    # initialize the amplitude and phase matrix
    amplitude_matrix = []
    phase_matrix = []

    # save the current directory
    directory = os.path.dirname(csv_original)
    
    # read the original csv file
    file_name = csv_original.split("/")[-1][0:-4]
    csv_original = open(csv_original, mode = 'r')

    # create a new csv file and save the csi_data rows inside
    csi_data_csv = open(directory + "/" + (file_name + "_csi_data.csv"), mode = "w", newline='')
    csv_writer = csv.writer(csi_data_csv)

    for i, line in enumerate(csv_original):
        try:
            if i == 0: # intestation row
                continue 

            line = line.split("\"")

            # delete this line if you work with original csv and uncomment csi_data = line[-2][1:-1].split(",")
            # line = line[0].split(",")

            csi_data = line[-2][1:-1].split(",")[:N] # csi_data contains 128 values

            # calculate_raw_amplitudes_and_phases returns the amplitude and phases for each trasmitted packet
            packet_amplitudes, packet_phases = calculate_raw_amplitudes_and_phases(csi_data)

            # filter and delete the initial and final parts of phase and amplitudes values because 0
            # packet_amplitudes = packet_amplitudes[subcarriers//2 - 15: subcarriers//2 + 15]
            # packet_phases = packet_phases[subcarriers//2 - 15 : subcarriers//2 + 15]

            packet_amplitudes = packet_amplitudes[0:subcarriers]
            packet_phases = packet_phases[0:subcarriers]

            # add the packet amplitudes to amplitude_matrix
            amplitude_matrix.append(packet_amplitudes)

            # add the packet phases to phase_matrix
            phase_matrix.append(packet_phases)

            # format again in str and save only 128 values into new csv
            result = ",".join(csi_data)
            csv_writer.writerow([result])
        
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # close the files
    csv_original.close()
    csi_data_csv.close()

    # here the phase and amplitude matrix will have (packet_trasmitted x 30) shape
    # sanification amplitude and phase matrix (add slice with n_packets if you want to filter the heatmap)
    amplitude_matrix = sanitize_and_median_filter(amplitude_matrix, 5, 5)
    smooth_matrix = gaussian_filter(amplitude_matrix, sigma=1.5)  # Additional smoothing step

    # plot heatmap and save it into the person folder
    heatmap_plot_processing(smooth_matrix[:n_packets], file_name, directory)
    heatmap_image = heatmap_png(smooth_matrix)

    # work on phase matrix -> np.arange create a 0,N sorted array
    phase_matrix_sanitize = sanitize_phase_matrix(np.array(phase_matrix), np.arange(0, len(phase_matrix[0])))

    return heatmap_image, phase_matrix_sanitize

def process_all_csv_in_folder(folder_path):

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
            heatmap_image, phase_matrix_sanitize = calculate_amplitude_phase_matrix(file_path)

            print(phase_matrix_sanitize.shape)
            all_phase_matrix.append(torch.tensor(phase_matrix_sanitize))
            # save the heatmap
            heatmap_image.save("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/dataset_heatmap/" + folder_path.split("/")[-1] + "/" + filename[:-4] + ".png")

            label = file_path.split("/")[-2] # Francesca
            # save the heatmaps and the label
            all_heatmaps.append(heatmap_image)
            print(all_heatmaps)
            all_labels.append(label)
            # complete folder dataset_heatmap creation

            print(f"Processing {file_path}")

        elif os.path.isdir(file_path):
            process_all_csv_in_folder(file_path)

if __name__ == "__main__":

    process_all_csv_in_folder("/Users/sebastiandinu/Desktop/Tesi-Triennale/dataset") # change with your dataset's path

    # work on all_phase_matrix -> padding the arrays to the max trasmitted_packets value
    max_seq_len = max(seq.shape[0] for seq in all_phase_matrix)  # Determine max seq_len (10 in this case)
    padded_data = pad_sequences(all_phase_matrix, max_seq_len)
    torch.save(padded_data, "phase_padded.pt") # save the batch matrix (35, max_len_seq, 64)

    print("phase_padded max len_seq: ", padded_data.shape[1])
    # save all the labels into a txt file as numpy array
    np.save("labels", all_labels)
    print(all_labels)
    np.save("heatmaps", np.array(all_heatmaps))

    # Load the tensor from the .pt file
    loaded_tensor = torch.load("phase_padded.pt")

    # 0 Francesca
    # 1 Federica
    # 2 Federico
    # 3 Sebastian
    # 4 Mara
















