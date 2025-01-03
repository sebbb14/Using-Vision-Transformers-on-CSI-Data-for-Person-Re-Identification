import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorflow.python.autograph.converters.functions import transform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=8, num_layers=4, dropout_rate=0.3):
        super(TemporalTransformer, self).__init__()

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 349, d_model))  # Adjust length as needed

        # Transformer Encoder Layer with Dropout
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,  # Feedforward network size
            #dropout=dropout_rate  # Dropout in attention and feedforward sublayers
        )

        # Stack multiple Transformer Encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Linear layer to project the output into feature space (for downstream tasks)
        self.linear = nn.Linear(d_model, feature_dim)  # Adjust the output dimension as needed

        # Output Dropout
        # self.output_dropout = nn.Dropout(dropout_rate)

        # Positional Encoding Dropout
        # self.positional_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape

        # Adding Positional Encoding to Input
        x = x + self.positional_encoding[:, :seq_len, :]  # Adding positional encoding
        # x = self.positional_dropout(x)  # Apply positional encoding dropout

        # Prepare Input for Transformer Encoder
        x = x.permute(1, 0, 2)  # Shape becomes (seq_len, batch_size, feature_dim)

        # Pass Through Transformer Encoder
        x = self.transformer_encoder(x)

        # Take the output corresponding to the last time step
        x = x[-1, :, :]  # Last time step

        # Project the output through a linear layer
        x = self.linear(x)
        # x = self.output_dropout(x)  # Apply output dropout

        return x

def normalize_tensor(tensor):
    # Compute mean and std across batch and sequence length dimensions
    mean = tensor.mean(dim=(0, 1), keepdim=True)  # mean across batch and sequence
    std = tensor.std(dim=(0, 1), keepdim=True)    # std across batch and sequence
    return (tensor - mean) / std

if __name__ == "__main__":

    # load my phase matrices tensor
    input_tensor = torch.load("phase_padded.pt")
    input_tensor = input_tensor.float()  # Convert to float32 (default dtype)
    labels = np.load("labels.npy")


    # convert labels into numerical
    # Convert to a numpy array and get unique values
    unique_names = np.unique(labels)
    # Create a dictionary mapping names to integers
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}
    # Convert the original list to integers -> [0,0,0,..., 4,4,4,...]
    labels = np.array([name_to_int[name] for name in labels])
    tensor_labels = torch.tensor(labels)


    # normalize the tensor
    normalized_tensor = normalize_tensor(input_tensor)

    dataset = TensorDataset(normalized_tensor, tensor_labels)

    # Create DataLoader for batching
    batch_size = 35 # Set the desired batch size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(train_loader.dataset)

    # Create the model class
    temporal_transformer = TemporalTransformer(feature_dim=128)

    # Optimizer
    optimizer = optim.Adam(temporal_transformer.parameters(), lr=1e-4)

    # Loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        temporal_transformer.train()  # Set model to training mode

        # Loop over batches from the DataLoader
        for phase_matrix, labels in train_loader:
            # phase_matrix shape: (batch_size, seq_len, feature_dim)
            # labels shape: (batch_size,)

            # Forward pass -> here is saved the embedding
            outputs = temporal_transformer(phase_matrix)  # Shape: (batch_size, feature_dim)
            print(outputs.shape)

            # Compute loss (Cross-Entropy for classification)
            loss = criterion(outputs, labels)  # Ensure labels and outputs have the same batch size

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
