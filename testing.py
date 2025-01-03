import numpy as np
from torch.ao.nn.quantized.functional import threshold

from combined_transformers import *
from main import *
from main2 import *
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# check if a pose have been already seen or not
# stored labels contains numbers
def re_identification(embedding, stored_embeddings, stored_labels):
    similitary = cosine_similarity(embedding, stored_embeddings).flatten()
    max_similitary = similitary.max()

    if max_similitary < 0.8:
        print("Persona mai vista")
        stored_embeddings = np.vstack((stored_embeddings, embedding))
        # find a new label number
        all_numbers = set(range(11))
        valid_numbers = list(all_numbers - set(stored_labels))
        random_number = random.choice(valid_numbers)
        stored_labels = np.concatenate((stored_labels, [random_number]))
        return

    if max_similitary > 0.8:
        index = np.where(similitary == max_similitary)
        print("Persona già vista: " + str(stored_labels[index]))





def calculate_rank1(query_embeddings, query_labels, stored_embeddings, stored_labels):
    """
    Calculate the Rank-1 accuracy for person re-identification.

    Args:
        query_embeddings (numpy.ndarray): Query embeddings (NxD).
        query_labels (numpy.ndarray): Labels for query embeddings (N).
        stored_embeddings (numpy.ndarray): Stored embeddings (MxD).
        stored_labels (numpy.ndarray): Labels for stored embeddings (M).

    Returns:
        float: The Rank-1 accuracy.
    """
    correct = 0

    for i, query_embedding in enumerate(query_embeddings):
        query_label = query_labels[i]

        # Compute cosine similarity (higher is more similar)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings).flatten()

        # Get the index of the most similar embedding (rank-1)
        rank1_index = np.argmax(similarities)
        rank1_label = stored_labels[rank1_index]

        # Check if the rank-1 label matches the query label
        if rank1_label == query_label:
            correct += 1

    # Rank-1 accuracy
    rank1_accuracy = correct / len(query_embeddings)
    return rank1_accuracy

def calculate_map(query_embeddings, query_labels, stored_embeddings, stored_labels):
    all_ap = []

    for i, query_embedding in enumerate(query_embeddings):
        query_label = query_labels[i]

        # Compute cosine similarity (higher is more similar)
        similarities = cosine_similarity(query_embedding, stored_embeddings).flatten()

        # Sort stored embeddings by similarity (descending)
        sorted_indices = np.argsort(-similarities)
        sorted_labels = stored_labels[sorted_indices]

        # Compute precision and recall
        relevant = (sorted_labels == query_label).astype(int)
        precision_at_k = np.cumsum(relevant) / (np.arange(1, len(relevant) + 1))
        recall_at_k = np.cumsum(relevant) / relevant.sum()

        # Average precision for this query
        ap = np.sum(precision_at_k * relevant) / relevant.sum() if relevant.sum() > 0 else 0
        all_ap.append(ap)

    # Mean Average Precision
    mean_ap = np.mean(all_ap)
    return mean_ap


if "__main__" == __name__:

    # load the final embeddings and labels data
    training_data = torch.load("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/combined/checkpoints_199.pt")
    stored_embeddings = training_data["embeddings"].detach().numpy()
    stored_labels = training_data["labels"].detach().numpy()

    stored_embeddings = torch.load("all_embeddings.pt")
    stored_embeddings = np.array([x.squeeze() for x in stored_embeddings])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
    ])

    # create the model
    vit_model = HeatmapViT(image_size=224, patch_size=16, num_classes=5)  # ViT model
    temporal_transformer = TemporalTransformer(feature_dim=256)  # temporal transformer
    model = CombinedReIDModel(vit_model, temporal_transformer, 512)

    # load the model
    model.load_state_dict(torch.load(
        "/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/heatmap_vit_model.pth",
        map_location=torch.device('cpu')))
    model.eval()

    # calculate the heatmap and phase matrix of a pose -> here I should use the testing dataset instead of a single csv file
    sequences = []
    heatmap1, phase_matrix1 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/posa1.csv")
    heatmap2, phase_matrix2 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/mossa1.csv")
    heatmap3, phase_matrix3 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/posa1_federico.csv")

    # mara e federica
    heatmap4, phase_matrix4 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/mossa5_mara.csv")
    heatmap5, phase_matrix5 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/mossa2_federica.csv")

    # sconosciuto
    heatmap6, phase_matrix6 = calculate_amplitude_phase_matrix2("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/saluto2_sconosciuto.csv")
    heatmap7, phase_matrix7 = calculate_amplitude_phase_matrix2("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/tPose2_sconosciuto.csv")

    # individuo 40
    heatmap8, phase_matrix8 = calculate_amplitude_phase_matrix2("/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/tPose3.csv")

    sebastian_tensor1 = transform(heatmap1) # heatmap already as an image
    sebastian_tensor2 = transform(heatmap2)
    federico_tensor3 = transform(heatmap3)
    mara_tensor4 = transform(heatmap4)
    federica_tensor5 = transform(heatmap5)
    sconosciuto_tensor6 = transform(heatmap6)
    sconosciuto_tensor7 = transform(heatmap7)
    sconosciuto_tensor8 = transform(heatmap8)


    # create the list of tensors
    sequences.append(torch.Tensor(phase_matrix1))
    sequences.append(torch.Tensor(phase_matrix2))
    sequences.append(torch.Tensor(phase_matrix3))
    sequences.append(torch.Tensor(phase_matrix4))
    sequences.append(torch.Tensor(phase_matrix5))
    sequences.append(torch.Tensor(phase_matrix6))
    sequences.append(torch.Tensor(phase_matrix7))
    sequences.append(torch.Tensor(phase_matrix8))


    # padded the phase_matrix to max number of seq_len in training
    max_seq_len = torch.load("phase_padded.pt").shape[1]
    phase_padded = pad_sequences(sequences, max_seq_len)
    print(phase_padded[0].shape)
    print(phase_padded[1].shape)

    # extract the embedding
    with torch.no_grad():
        sebastian_embedding1 = model(phase_padded[0].unsqueeze(0), sebastian_tensor1.unsqueeze(0))
        sebastian_embedding2 = model(phase_padded[1].unsqueeze(0), sebastian_tensor2.unsqueeze(0))
        federico_embeddings3 = model(phase_padded[2].unsqueeze(0), federico_tensor3.unsqueeze(0))
        mara_embeddings4 = model(phase_padded[3].unsqueeze(0), mara_tensor4.unsqueeze(0))
        federica_embeddings5 = model(phase_padded[4].unsqueeze(0), federica_tensor5.unsqueeze(0))
        sconosciuto_embeddings6 = model(phase_padded[5].unsqueeze(0), sconosciuto_tensor6.unsqueeze(0))
        sconosciuto_embeddings7 = model(phase_padded[6].unsqueeze(0), sconosciuto_tensor7.unsqueeze(0))
        sconosciuto_embeddings8 = model(phase_padded[7].unsqueeze(0), sconosciuto_tensor8.unsqueeze(0))

    # calculate the distance between 2 embeddings and see the results
    similarity = cosine_similarity(sebastian_embedding1.detach().numpy(), sebastian_embedding2.detach().numpy())[0, 0]
    print("sebastian mossa1 e sebastian mossa2: ", similarity) # expected value close to 1
    similarity = cosine_similarity(federico_embeddings3.detach().numpy(), sebastian_embedding1.detach().numpy())[0, 0]
    print("federico posa1 e sebastian posa1: ", similarity) # expected value close to 0
    similarity = cosine_similarity(mara_embeddings4.detach().numpy(), federica_embeddings5.detach().numpy())[0, 0]
    print("mara mossa5 e federica mossa5: ", similarity) # expected value close to 0

    # add a new person never seen before
    similarity = cosine_similarity(sebastian_embedding1.detach().numpy(), sconosciuto_embeddings6.detach().numpy())[0, 0]
    print("sebastian posa1 e sconosciuto saluto2: ", similarity)
    similarity = cosine_similarity(sconosciuto_embeddings6.detach().numpy(), sconosciuto_embeddings7.detach().numpy())[0, 0]
    print("sconosciuto saluto2 e sconosciuto saluto3: ", similarity)


    # Calculate cosine similarity
    distances = cosine_similarity(federico_embeddings3.detach().numpy(), stored_embeddings).flatten()
    max_similitary = distances.max()
    index = np.where(distances == max_similitary)
    labels = np.load("labels.npy")
    unique_names = np.unique(labels)
    # Create a dictionary mapping names to integers
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}
    # Convert the original list to integers -> [0,0,0,..., 4,4,4,...]
    labels = np.array([name_to_int[name] for name in labels])
    tensor_labels = torch.tensor(labels)

    # testing the mAP
    query_embeddings = [
        sebastian_embedding1.detach().numpy(),
        sebastian_embedding2.detach().numpy(),
        federico_embeddings3.detach().numpy(),
        mara_embeddings4.detach().numpy(),
        federica_embeddings5.detach().numpy()]

    query_labels = np.array([4,4,1,3,0])


    # Calculate mAP
    map_score = calculate_map(query_embeddings, query_labels, stored_embeddings, labels)
    print(f"Mean Average Precision (mAP): {map_score}")

    # Calculate Rank-1 accuracy
    rank1_accuracy = calculate_rank1(query_embeddings, query_labels, stored_embeddings, labels)
    print(f"Rank-1 Accuracy: {rank1_accuracy}")

    if max_similitary >= 0.8:
        print("Persona già vista")



    # testing the re-identification process
    result = re_identification(sebastian_embedding1.detach().numpy(), stored_embeddings, labels)
    result = re_identification(federico_embeddings3.detach().numpy(), stored_embeddings, labels)












