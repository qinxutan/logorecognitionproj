import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, lastLayer=False, pretrained=True):
        super(SiameseNetwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        if self.lastLayer:
            self.extraL = nn.Linear(4096, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward_once(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, input1, input2):
        # Forward pass through the network for two inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.lastLayer:
            # compute l1 distance (similarity) between the 2 encodings
            diff = torch.abs(output1 - output2)
            scores = self.extraL(diff)
            return scores
        else:
            return output1, output2