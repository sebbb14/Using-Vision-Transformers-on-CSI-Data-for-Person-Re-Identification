import csv

from scipy.ndimage import gaussian_filter

from cnn import *
from csi_data import *
from lstm import *
from plots import *

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

    for i, line in enumerate(csv_original):
        try:
            if i == 0: # intestation row
                continue 

            line = line.split("\"")

            # delete this line if you work with original csv and uncomment csi_data = line[-2][1:-1].split(",")
            line = line[0].split(",")

            # csi_data = line[-2][1:-1].split(",")

            # calculate_raw_amplitudes_and_phases returns the amplitude and phases for each trasmitted packet
            packet_amplitudes, packet_phases = calculate_raw_amplitudes_and_phases(line, subcarriers)

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

    # sanification amplitude and phase matrix
    amplitude_matrix = sanitize_and_median_filter(amplitude_matrix[0:n_packets], 5, 5)
    smooth_matrix = gaussian_filter(amplitude_matrix, sigma=1.5)  # Additional smoothing step

    # plot heatmap and save it into the person folder
    heatmap_plot_processing(smooth_matrix, file_name, directory)

    # convert the image into a 224x224 png for the vgg-16 method even the png is saved at a different resolution
    heatmap_resized = heatmap_vgg16(smooth_matrix)

    # calculate the feature vector with a cnn vgg-16 and convert it into numpy array
    vgg_features = feature_map_vector(heatmap_resized).detach().numpy().flatten()
    signature = list(vgg_features) # create a signature

    # work on phase matrix
    phase_matrix_sanitize = sanitize_phase_matrix(np.array(phase_matrix), np.arange(0, subcarriers-1))

    # phase plot (original, sanitized, filtered) with index subcarrier = 0
    # plot_phase_processing(np.array(phase_matrix), file_name, directory)

    lstm_feature = lstm_features(phase_matrix_sanitize)

    signature.extend(list(lstm_feature))
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
            all_labels.append(file_path.split("/")[-2])
            print(f"Processing {file_path}")

        elif os.path.isdir(file_path):
            process_all_csv_in_folder(file_path)


if __name__ == "__main__":

    process_all_csv_in_folder("/Users/sebastiandinu/Desktop/Tesi-Triennale/dataset ridotto") # change with your dataset's path

    # Replace NaNs values with zeros
    # signatures = np.nan_to_num(np.array(all_signatures), nan=0.0, posinf=0.0, neginf=0.0)

    # save the signatures on a txt file
    np.savetxt("signatures.txt", all_signatures, fmt='%f', delimiter=',')
    np.savetxt("labels.txt", np.array(all_labels), fmt='%s')



