import torch
import numpy as np
import os
from torch import nn
from transformers import ViTModel, ViTConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class HeatmapViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes):
        super().__init__()
        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,  # Single-channel heatmap images
            num_labels=num_classes,
            num_attention_heads=8,  # Explicitly set attention heads
            hidden_size=256  # Set your desired number of features here (multiple of 12 = attention heads)
        )
        self.vit = ViTModel(vit_config)
        self.fc = nn.Linear(self.vit.config.hidden_size, num_classes) # classification layer
        # Add dropout for regularization
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        # Take the CLS token representation (first token)
        cls_token_embedding = outputs.last_hidden_state[:, 0]
        # Apply dropout to embeddings
        # cls_token_embedding = self.dropout(cls_token_embedding)

        return cls_token_embedding

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

def training_loop(epochs, dataset):

    # takes 32 shuffle heatmaps
    data_loader = DataLoader(dataset, batch_size=35, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)  # find all subfolder's names and assing a numerical label to it: 5 with my dataset

    model = HeatmapViT(image_size=224, patch_size=16, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # criterion = CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)
    saved_labels = []
    for epoch in range(epochs):
        all_embeddings = []  # To store embeddings for the entire epoch
        model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            print("label person: ", labels)

            # Forward pass
            outputs = model(images) # outputs contains the embeddings
            embeddings = outputs.cpu()
            # loss = criterion(outputs, labels)

            # Generate triplets -> contains batch * 3 embeddings (anchor, positive, negative) -> 96 total
            triplets = generate_triplets(embeddings, labels)

            # Compute triplet loss
            if triplets:
                anchors, positives, negatives = zip(*triplets)
                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)
                loss = triplet_loss_fn(anchors, positives, negatives)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Collect embeddings from the current batch
            all_embeddings.append(embeddings.detach().numpy())

            # Save embeddings and labels during the last epoch
            if epoch == epochs - 1:
                saved_labels.append(labels.detach().cpu())

        # Concatenate all embeddings for the epoch
        all_embeddings = np.concatenate(all_embeddings, axis=0)  # Combine all batches

        # Save concatenated embeddings
        np.save(os.path.join("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/saved_features_concatenated", f"epoch_{epoch + 1}_embeddings"), all_embeddings)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

    torch.save(model.state_dict(), "heatmap_vit_model.pth")
    return saved_labels

def compute_similarity_scores(model, pairs, metric='cosine'):
    similarities = []
    labels = []
    for (img1, img2, label) in pairs:  # label: 1 for same person, 0 for different people
        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0))
            emb2 = model(img2.unsqueeze(0))

        if metric == 'cosine':
            similarity = cosine_similarity(emb1.detach().numpy(), emb2.detach().numpy())[0, 0]
        elif metric == 'euclidean':
            similarity = -torch.norm(emb1 - emb2, p=2).item()  # Negative because smaller is better

        similarities.append(similarity)
        labels.append(label)
    return similarities, labels

if __name__ == "__main__":

    # create transform for normalize the input heatmap image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
    ])

    # Replace 'path_to_heatmaps' with the directory containing your heatmaps and apply the transform
    dataset = datasets.ImageFolder(root='/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/dataset_heatmap', transform=transform)

    # Training loop
    last_labels = training_loop(100, dataset) # return the last labels epoch used
    np.savetxt("last_labels.txt", np.array(last_labels), fmt='%d')


    stored_embeddings = np.load("./saved_features_concatenated/epoch_100_embeddings.npy")
    print("Stored embeddings lenght: ", stored_embeddings.shape)
    # testing phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapViT(image_size=224, patch_size=16, num_classes=len(dataset.classes)).to(device)
    # load the pre-trained model into model class
    model.load_state_dict(torch.load("/Users/sebastiandinu/Desktop/Tesi-Triennale/re-identification-csi/heatmap_vit_model.pth"))

    # Set the model to evaluation mode
    model.eval()

    # Test with a new heatmap
    heatmap_path1 = "posa1 copy.png"
    heatmap_path2 = "posa1 copy.png"
    heatmap_path3 = "posa1 copy.png"
    heatmap_path4 = "posa1 copy.png"
    heatmap_path5 = "posa1 copy.png"


    # delete the batch dimension
    heatmap_png1 = Image.open(heatmap_path1)
    input_tensor1 = transform(heatmap_png1) # Add batch dimension
    heatmap_png2 = Image.open(heatmap_path2)
    input_tensor2 = transform(heatmap_png2)  # Add batch dimension
    heatmap_png3 = Image.open(heatmap_path3)
    input_tensor3 = transform(heatmap_png3)  # Add batch dimension
    heatmap_png4 = Image.open(heatmap_path4)
    input_tensor4 = transform(heatmap_png4)  # Add batch dimension
    heatmap_png5 = Image.open(heatmap_path5)
    input_tensor5 = transform(heatmap_png5)  # Add batch dimension

    # visualize the embeddings
    with torch.no_grad():
        embedding1 = model(input_tensor1.unsqueeze(0))
        embedding2 = model(input_tensor2.unsqueeze(0))
        embedding3 = model(input_tensor3.unsqueeze(0))
        embedding4 = model(input_tensor4.unsqueeze(0))
        embedding5 = model(input_tensor5.unsqueeze(0))


    embeddings = [np.array(embedding1.squeeze(0)), np.array(embedding2.squeeze(0)), np.array(embedding3.squeeze(0)), np.array(embedding4.squeeze(0)), np.array(embedding5.squeeze(0))]  # All embeddings

    tsne = TSNE(n_components=2,random_state=42, perplexity=2)
    reduced = tsne.fit_transform(np.array(embeddings))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=[0,1,2,1,2])  # color by person ID
    plt.show()

    similitaries, labels = compute_similarity_scores(model, [(input_tensor2, input_tensor4, 1), (input_tensor1, input_tensor4, 0)])

    input_tensor1 = transform(heatmap_png1).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        new_embedding = model(input_tensor1)
        print("Extracted Embedding:", new_embedding)
        print("Embedding Shape:", new_embedding.shape)

    # compare the embedding with the stored ones
    print(type(stored_embeddings))
    print(type(np.array(new_embedding)))
    print(stored_embeddings.shape)
    print(np.array(new_embedding).shape)


    distances = cosine_similarity(np.array(new_embedding), stored_embeddings)
    max_similarity = max(distances[0])

    # Decision
    threshold = 0.006  # Define a similarity threshold
    if max_similarity > threshold:
        print("Person identified")
    else:
        print("New person detected!")

    print(stored_embeddings[0], stored_embeddings[1])