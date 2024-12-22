from torch.ao.nn.quantized.functional import threshold

from combined_transformers import *
from main import *
from main2 import *
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


if "__main__" == __name__:

    # load the final embeddings and labels data
    training_data = torch.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/combined/checkpoints_199.pt")
    stored_embeddings = training_data["embeddings"].detach().numpy()
    stored_labels = training_data["labels"].detach().numpy()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
    ])

    # create the model
    vit_model = HeatmapViT(image_size=224, patch_size=16, num_classes=5)  # ViT model
    temporal_transformer = TemporalTransformer(feature_dim=128)  # temporal transformer
    model = CombinedReIDModel(vit_model, temporal_transformer, 512)

    # load the model
    torch.save(model.state_dict(), "heatmap_vit_model.pth")

    model.eval()

    # calculate the heatmap and phase matrix of a pose -> here I should use the testing dataset instead of a single csv file
    sequences = []
    heatmap1, phase_matrix1 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/posa1.csv")
    heatmap2, phase_matrix2 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/mossa1.csv")
    heatmap3, phase_matrix3 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/posa1_federico.csv")

    # mara e federica
    heatmap4, phase_matrix4 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/mossa5_mara.csv")
    heatmap5, phase_matrix5 = calculate_amplitude_phase_matrix("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/mossa2_federica.csv")

    # sconosciuto
    heatmap6, phase_matrix6 = calculate_amplitude_phase_matrix2("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/saluto2_sconosciuto.csv")
    heatmap7, phase_matrix7 = calculate_amplitude_phase_matrix2("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/saluto3_sconosciuto.csv")

    sebastian_tensor1 = transform(heatmap1) # heatmap already as an image
    sebastian_tensor2 = transform(heatmap2)
    federico_tensor3 = transform(heatmap3)
    mara_tensor4 = transform(heatmap4)
    federica_tensor5 = transform(heatmap5)
    sconosciuto_tensor6 = transform(heatmap6)
    sconosciuto_tensor7 = transform(heatmap7)


    # create the list of tensors
    sequences.append(torch.Tensor(phase_matrix1))
    sequences.append(torch.Tensor(phase_matrix2))
    sequences.append(torch.Tensor(phase_matrix3))
    sequences.append(torch.Tensor(phase_matrix4))
    sequences.append(torch.Tensor(phase_matrix5))
    sequences.append(torch.Tensor(phase_matrix6))
    sequences.append(torch.Tensor(phase_matrix7))


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
    distances = cosine_similarity(sebastian_embedding1.detach().numpy(), stored_embeddings).flatten()
    max_similitary = distances.max()


    if max_similitary >= 0.8:
        print("Persona già vista")










