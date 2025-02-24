import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        input_height,
        input_width,
        action_dim,
        conv1_channels=16,
        conv2_channels=32,
        conv3_channels=64,
        fc1_units=512,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(BehaviorCloningPolicy, self).__init__()

        # Store input size
        self.input_height = input_height
        self.input_width = input_width
        self.action_dim = action_dim

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                conv1_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),  # 1st Conv
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1st Pooling
            nn.Conv2d(
                conv1_channels,
                conv2_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),  # 2nd Conv
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 2nd Pooling
            nn.Conv2d(
                conv2_channels,
                conv3_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),  # 3rd Conv
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 3rd Pooling
        )

        # Compute the output size after convolutions & pooling dynamically
        output_height = self.compute_conv_output_size(
            input_height, 3, kernel_size, stride, padding
        )
        output_width = self.compute_conv_output_size(
            input_width, 3, kernel_size, stride, padding
        )
        # Compute the flattened size
        self.flattened_size = conv3_channels * output_height * output_width

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, action_dim),
        )

    @staticmethod
    def compute_conv_output_size(input_size, num_conv_layers, kernel, stride, padding):
        size = input_size
        for _ in range(num_conv_layers):
            size = (size - kernel + 2 * padding) // stride + 1
            size = size // 2
        return size

    def forward(self, x):
        x = self.features(x)  # Apply convolutions and pooling
        x = x.view(-1, self.flattened_size)  # Flatten for fully connected layers
        x = self.classifier(x)  # Apply fully connected layers
        return x

    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
        pred = torch.argmax(logits, dim=1)
        return pred
