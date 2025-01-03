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