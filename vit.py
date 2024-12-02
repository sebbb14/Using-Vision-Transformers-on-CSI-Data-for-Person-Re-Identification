import torch
import numpy as np
import os
from tensorflow.python.keras.saving.save import load_model
from torch import nn
from transformers import ViTModel, ViTConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class HeatmapViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, pretrained=True):
        super().__init__()
        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,  # Single-channel heatmap images
            num_labels=num_classes,
            hidden_size=252  # Set your desired number of features here (multiple of 12 = attention heads)
        )
        self.vit = ViTModel(vit_config)
        self.fc = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        # Take the CLS token representation (first token)
        cls_token_embedding = outputs.last_hidden_state[:, 0]
        return cls_token_embedding

def training_loop(epochs, dataset):

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)  # find all subfolder's names and assing a numerical label to it
    model = HeatmapViT(image_size=224, patch_size=16, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        all_embeddings = []  # To store embeddings for the entire epoch
        model.train()
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images) # outputs contains the embeddings
            embeddings = outputs.cpu()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Collect embeddings from the current batch
            all_embeddings.append(embeddings.detach().numpy())

        # Concatenate all embeddings for the epoch
        all_embeddings = np.concatenate(all_embeddings, axis=0)  # Combine all batches

        # Save concatenated embeddings
        np.save(os.path.join("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/saved_features_concatenated", f"epoch_{epoch + 1}_embeddings"), all_embeddings)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

    torch.save(model.state_dict(), "heatmap_vit_model.pth")



def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)  # Extract embeddings
            embeddings.append(outputs.cpu())
            labels.append(targets.cpu())

    return torch.cat(embeddings), torch.cat(labels)

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
    ])

    # Replace 'path_to_heatmaps' with the directory containing your heatmaps
    dataset = datasets.ImageFolder(root='/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/dataset_heatmap', transform=transform)


    # Training loop
    # training_loop(10, dataset)
    embeddings = np.load("./saved_features_concatenated/epoch_10_embeddings.npy")


    model = HeatmapViT(image_size=224, patch_size=16, num_classes=num_classes).to(device)







