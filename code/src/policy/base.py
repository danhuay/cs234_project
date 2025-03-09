import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_height,
        input_width,
        conv1_channels=16,
        conv2_channels=32,
        conv3_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.features(x)
        flattend_x = x.view(x.size(0), -1)
        return flattend_x

    def get_output_size(self, input_height, input_width):
        dummy_input = torch.zeros(
            1, 3, input_height, input_width
        )  # (batch_size, channels, height, width)
        # If your model is on the GPU, move the dummy input there:
        dummy_input = dummy_input.to(next(self.features.parameters()).device)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        return flattened_size


class MLPPolicy(nn.Module):
    def __init__(self, input_size, action_dim, hidden_units=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

    def forward(self, x):
        return self.classifier(x)


class CNNPolicy(nn.Module):
    def __init__(
        self,
        input_height,
        input_width,
        action_dim,
        conv1_channels=16,
        conv2_channels=32,
        conv3_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        hidden_units=512,
    ):
        super().__init__()
        self.features = CNNFeatureExtractor(
            input_height,
            input_width,
            conv1_channels,
            conv2_channels,
            conv3_channels,
            kernel_size,
            stride,
            padding,
        )
        feature_dim = self.features.get_output_size(input_height, input_width)
        self.classifier = MLPPolicy(feature_dim, action_dim, hidden_units)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
        return torch.argmax(logits, dim=1)
