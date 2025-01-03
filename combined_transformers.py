import numpy as np
import torch
import torch.nn as nn

from vit import HeatmapViT
from temporal_transformer import TemporalTransformer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import random

def get_data(batch_size):

    # import the phase matrices
    input_tensor = torch.load("phase_padded.pt") # already padded
    input_tensor = input_tensor.float()  # Convert to float32 (default dtype)
    phase_matrices = normalize_tensor(input_tensor)
    # phase_matrices follow targets [Francesca, ...] recursive method

    # import the heatmap dataset
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
    ])

    heatmaps = np.load("heatmaps.npy")
    tensor_heatmaps = torch.from_numpy(heatmaps).float()
    tensor_heatmaps = tensor_heatmaps.permute(0, 3, 1, 2)
    tensor_heatmaps = transform(tensor_heatmaps)

    # load the labels -> recursive
    labels = np.load("labels.npy")
    unique_names = np.unique(labels)
    # Create a dictionary mapping names to integers
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}
    # Convert the original list to integers -> [0,0,0,..., 4,4,4,...]
    labels = np.array([name_to_int[name] for name in labels])
    tensor_labels = torch.tensor(labels)

    # Create a dataset that contains the phase_matrix, heatmap, and labels
    dataset = TensorDataset(phase_matrices, tensor_heatmaps, tensor_labels)

    # Create a DataLoader that will shuffle the data at every epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def normalize_tensor(tensor):

    # Compute mean and std across batch and sequence length dimensions
    mean = tensor.mean(dim=(0, 1), keepdim=True)  # mean across batch and sequence
    std = tensor.std(dim=(0, 1), keepdim=True)    # std across batch and sequence
    return (tensor - mean) / std

class CombinedReIDModel(nn.Module):
    # the model takes in input the previous transofmers created
    def __init__(self, vit_model, temporal_transformer, output_dim):
        super(CombinedReIDModel, self).__init__()
        self.vit_model = vit_model
        self.temporal_transformer = temporal_transformer

        # final dense layer
        self.fc = nn.Linear(vit_model.vit.config.hidden_size + temporal_transformer.linear.out_features, output_dim)
        # self.fc = nn.Linear(vit_model.vit.config.hidden_size, output_dim)

    def forward(self, phase_matrix, heatmap_input):
        # Step 1: Extract temporal features
        temporal_features = self.temporal_transformer(phase_matrix)  # (batch_size, temporal_feature_dim)

        # Step 2: Extract ViT embeddings from the heatmap input
        vit_embeddings = self.vit_model(heatmap_input)  # (batch_size, vit_embedding_dim)

        # Step 3: Concatenate the ViT embeddings and temporal features
        combined_features = torch.cat((vit_embeddings, temporal_features), dim=1)  # (batch_size, combined_feature_dim)

        # Step 4: Pass through a fully connected layer (optional: for dimensionality reduction)
        output = self.fc(combined_features)  # (batch_size, output_dim)

        return output

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Euclidean distance between anchor-positive and anchor-negative
        positive_distance = torch.norm(anchor - positive, p=2) # apply euclidean distance
        negative_distance = torch.norm(anchor - negative, p=2)

        # Triplet loss
        loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0))
        return loss

def generate_triplets(batch_embeddings, batch_labels):
    triplets = []
    for i, anchor_label in enumerate(batch_labels):
        anchor = batch_embeddings[i]

        # Find all positives (same label) and negatives (different label)
        positives = [j for j in range(len(batch_labels)) if batch_labels[j] == anchor_label and j != i]
        negatives = [j for j in range(len(batch_labels)) if batch_labels[j] != anchor_label]

        if not positives or not negatives:
            continue

        # Randomly pick one positive and one negative
        positive = random.choice(positives)
        negative = random.choice(negatives)

        triplets.append((anchor, batch_embeddings[positive], batch_embeddings[negative]))

    return triplets

# Function to sample triplets (anchor, positive, negative)
def sample_triplets(embeddings, labels):
    """
    Given a batch of embeddings and labels, create triplets (anchor, positive, negative).
    """
    triplets = []
    label_set = torch.unique(labels)  # Unique class labels in the batch

    for idx, label in enumerate(label_set): # 5 classes so the triplets contains 5 tuples of 3 embeddings
        # Find all embeddings corresponding to this label
        positive_samples = embeddings[labels == label]
        negative_samples = embeddings[labels != label]

        # Ensure at least one positive and one negative sample exist in the batch
        if positive_samples.size(0) > 0 and negative_samples.size(0) > 0:
            # Randomly sample positive and negative samples (taking the first one here)
            anchor = positive_samples[0]  # For simplicity, taking the first positive sample as anchor
            positive = positive_samples[1]  # Another positive sample
            negative = negative_samples[0]  # One negative sample
            triplets.append((anchor, positive, negative))

    return triplets

if __name__ == '__main__':

    # Initialize models
    vit_model = HeatmapViT(image_size=224, patch_size=16, num_classes=5) # ViT model
    temporal_transformer = TemporalTransformer(feature_dim=256) # temporal transformer
    combined_model = CombinedReIDModel(vit_model, temporal_transformer, output_dim=512)  # Final combined model

    # Loss function (triplet loss or contrastive loss for re-identification)
    triplet_loss = TripletLoss(margin=0.5)

    # Optimizer
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.0001)

    batch_size = 35
    train_loader = get_data(batch_size)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0

        # used train_loader to shuffle the labels
        for batch_idx, (phase_matrix, heatmap_input, label_batch) in enumerate(train_loader):
            # Forward pass
            embeddings = combined_model(phase_matrix, heatmap_input)  # (batch_size, embedding_dim)

            # Sample triplets from the embeddings and labels
            triplets = generate_triplets(embeddings, label_batch)

            # Accumulate loss for the entire batch
            total_loss = 0.0
            for anchor, positive, negative in triplets:
                total_loss += triplet_loss(anchor, positive, negative)

            # Perform the backward pass only once
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item()}")

        save_dict = {"epoch": epoch, "embeddings": embeddings, "labels": label_batch}
        torch.save(save_dict, f'combined/checkpoints_{epoch}.pt')
        torch.save(combined_model.state_dict(), "heatmap_vit_model.pth")

