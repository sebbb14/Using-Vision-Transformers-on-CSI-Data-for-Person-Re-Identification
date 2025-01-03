import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from transformers import ViTModel, ViTConfig
from torch import nn, optim


class HeatmapDataset(Dataset):
    def __init__(self, heatmaps, labels=None, transform=None):
        """
        Args:
            heatmaps (list of numpy arrays): List of heatmap images.
            labels (list or None): List of labels (optional, for supervised training).
            transform (callable, optional): Transform to be applied on an image.
        """
        self.heatmaps = heatmaps
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, idx):
        heatmap = self.heatmaps[idx]

        if self.transform:
            heatmap = self.transform(heatmap)

        if self.labels is not None:
            return heatmap, self.labels[idx]
        else:
            return heatmap


class ViTEmbeddingExtractor(nn.Module):
    def __init__(self, image_size=224, num_classes=None):
        super().__init__()
        config = ViTConfig(image_size=image_size)
        self.vit_model = ViTModel(config)
        self.num_classes = num_classes

        # Add classification head if supervised learning is desired
        if num_classes is not None:
            self.classification_head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, images):
        outputs = self.vit_model(pixel_values=images)
        embeddings = outputs.last_hidden_state[:, 0]  # Extract [CLS] token embeddings

        if self.num_classes is not None:
            logits = self.classification_head(embeddings)
            return embeddings, logits

        return embeddings


def train_vit_model(dataset, image_size=224, batch_size=32, num_epochs=10, learning_rate=1e-4, num_classes=None):
    """
    Args:
        dataset: PyTorch Dataset object containing heatmaps and (optionally) labels.
        image_size: Image size expected by the ViT model.
        batch_size: Batch size for DataLoader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        num_classes: Number of classes for supervised classification (optional).
    Returns:
        Trained model and embeddings.
    """
    # Data preprocessing
    transform = Compose([
        ToTensor(),
        Resize((image_size, image_size)),
        Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale heatmaps
    ])
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = ViTEmbeddingExtractor(image_size=image_size, num_classes=num_classes)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() if num_classes else None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if num_classes:
                images, labels = batch
                labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                images = batch

            images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer.zero_grad()

            if num_classes:
                embeddings, logits = model(images)
                loss = criterion(logits, labels)
            else:
                embeddings = model(images)
                loss = 0  # In case of unsupervised embedding generation

            if loss:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    return model


def extract_embeddings(model, dataset, image_size=224, batch_size=32):
    """
    Extracts embeddings for a given dataset using the trained ViT model.

    Args:
        model: Trained ViT model.
        dataset: Dataset to extract embeddings from.
        image_size: Image size expected by the ViT model.
        batch_size: Batch size for DataLoader.

    Returns:
        List of embeddings.
    """
    model.eval()
    transform = Compose([
        ToTensor(),
        Resize((image_size, image_size)),
        Normalize(mean=[0.5], std=[0.5])
    ])
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.cpu().numpy())

    return torch.cat(embeddings, dim=0)


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Example heatmap dataset
    heatmaps = [np.random.rand(64, 64) for _ in range(1000)]  # Replace with actual heatmaps
    labels = np.random.randint(0, 10, size=(1000,))  # Replace with actual labels if supervised learning is needed

    dataset = HeatmapDataset(heatmaps, labels)

    # Train the model
    model = train_vit_model(dataset, num_classes=10)

    # Extract embeddings
    embeddings = extract_embeddings(model, dataset)
    print("Embeddings shape:", embeddings.shape)
