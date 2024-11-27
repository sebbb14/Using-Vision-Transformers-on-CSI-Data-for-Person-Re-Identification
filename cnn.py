import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights


# Define the VGG-16 based model
class VGG16_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16_FeatureExtractor, self).__init__()
        # Load the pre-trained VGG-16 model
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # Use the layers up to the last max-pooling layer
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])  # Exclude classification layers
        
        # Output layer to generate feature map vector
        # Flatten the feature maps and reduce dimensionality
        self.fc = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),  # 25088 matches the VGG-16 output feature size for 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)  # Final feature map vector size (256 as an example)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x
 

def feature_map_vector(heatmap_RGB): 
    # Instantiate the model
    model = VGG16_FeatureExtractor()

    # transform RGB image to tensor for vgg-16
    transform = transforms.Compose([transforms.ToTensor()])

    # Apply the transformations
    input_tensor = transform(heatmap_RGB).unsqueeze(0)  # Add a batch dimension
    # the input for the model must have these characteristics (1, 3, 224, 224) -> batch, channels, width, height

    # Forward pass
    output = model(input_tensor)

    return output

