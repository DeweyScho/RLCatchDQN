#Dewey Schoenfelder
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # input is grayscale (1 channel), output is 32 feature maps, kernel size 5x5
        self.pool = nn.MaxPool2d(2, 2)    # defined once, reused
        self.conv2 = nn.Conv2d(32, 6, 5)  # input channels = 32 (from conv1), output = 6 channels

        self.fc1 = nn.Linear(6 * 22 * 22, 100)  # flattened size based on output of conv layers
        self.fc2 = nn.Linear(100, 3)            # output size = 3 (e.g., left, stay, right)
        self.sm = nn.Softmax(dim=1)             # softmax over actions

    def forward(self, x):
        # x expected shape: (batch_size, 1, 100, 100)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.sm(self.fc2(x))  # softmax activation on final layer
        return x

