import numpy as np
from abc import ABC

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


    

class CNN(nn.Module) :

    def __init__(self, n_classes=5, in_channels=3, dropout=0.5):
        super().__init__()

        #convolution 1 : low level caracteristic detection
        self.conv1 = nn.Sequential(
            #convolution 1.1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #convolution 1.2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        #convolution 2 : intermediate level patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        #convolution 3 : high level 
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )


    def forward(self, images) :

        images = self.conv1(images)
        images = self.conv2(images)
        images = self.conv3(images)

        images = self.global_pool(images)

        images = images.view(images.size(0), -1)

        images = self.dropout(images)
        images = self.fc(images)

        return images




    