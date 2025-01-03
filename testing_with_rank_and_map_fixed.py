from torch.ao.nn.quantized.functional import threshold
from combined_transformers import *
from main import *
from main2 import *
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import average_precision_score
def compute_rank_and_map(stored_embeddings, stored_labels, query_embeddings, query_labels, top_k=5):

    rank_results = []
    ap_scores = []

    for query_emb, query_label in zip(query_embeddings, query_labels):
        # Ensure query_emb is 2D
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Ensure stored_embeddings is 2D
        if stored_embeddings.ndim == 1:
            stored_embeddings = stored_embeddings.reshape(1, -1)

        # Compute cosine similarity between the query and stored embeddings.
        similarities = cosine_similarity(query_emb, stored_embeddings).flatten()

        # Rank stored items by descending similarity.
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_labels = stored_labels[ranked_indices]

        # Find the rank of the first correct match.
        correct_indices = np.where(ranked_labels == query_label)[0]
        if correct_indices.size > 0:
            rank_results.append(correct_indices[0] + 1)  # Convert to 1-based rank.
        else:
            rank_results.append(float('inf'))  # No correct match.

        # Compute Average Precision (AP) for this query.
        relevant = (ranked_labels == query_label).astype(int)
        ap = average_precision_score(relevant, similarities[ranked_indices])
        ap_scores.append(ap)

    # Compute mAP as the mean of Average Precision scores.
    mean_ap = np.mean(ap_scores)

    return rank_results, mean_ap


if "__main__" == __name__:

    # sequential order of labels following the recursive function
    stored_labels = np.load("labels.npy")
    unique_names = np.unique(stored_labels)
    # Create a dictionary mapping names to integers
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}
    # Convert the original list to integers -> [0,0,0,..., 4,4,4,...]
    stored_labels = np.array([name_to_int[name] for name in stored_labels])

    stored_embeddings = torch.load("all_embeddings.pt")
    stored_embeddings = np.array([x.squeeze() for x in stored_embeddings])


    # create the model
    vit_model = HeatmapViT(image_size=224, patch_size=16, num_classes=5)  # ViT model
    temporal_transformer = TemporalTransformer(feature_dim=256)  # temporal transformer
    model = CombinedReIDModel(vit_model, temporal_transformer, 512)

    # load the model
    model.load_state_dict(torch.load(
        "/Users/sebastiandinu/Library/Mobile Documents/com~apple~CloudDocs/01 Projects/Tesi/re-identification-csi/heatmap_vit_model.pth",
        map_location=torch.device('cpu')))
    model.eval()

    # Testing phase using the same function used for creating the dataloader during training
    testing_loader = get_data(35)
    # the phase matrix is already padded
    for batch_idx, (phase_matrix, heatmap_input, label_batch) in enumerate(testing_loader):
        print(batch_idx, label_batch, heatmap_input.shape, phase_matrix.shape)

        embeddings = model(phase_matrix, heatmap_input)  # (batch_size, embedding_dim)

        # Compute Rank and mAP
        rank_results, mean_ap = compute_rank_and_map(
            stored_embeddings, stored_labels, embeddings.detach().numpy(), label_batch.detach().numpy()
        )

    print("Rank Results (per query):", rank_results)
    print("Mean Average Precision (mAP):", mean_ap)



