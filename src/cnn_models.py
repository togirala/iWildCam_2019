from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import torch

class FirstModel(nn.Module):
    """Some Information about FirstModel"""
    def __init__(self, features_size=175, weights=None):
        super (FirstModel, self).__init__()
        self.resnet = models.resnet152(pretrained=False)
        if weights:
            self.resnet.load_state_dict(torch.load(weights))
        self.resnet.fc = nn.Linear(in_features = self.resnet.fc.in_features, out_features = 23)
        self.fc1 = nn.Linear(in_features=features_size, out_features=23)
        self.out = nn.Linear(in_features=46, out_features=23)

    def forward(self, image, features):
        image_out = self.resnet(image)
        features_out = self.fc1(features)
        x = F.relu(torch.cat((image_out, features_out), dim=1))
        x = self.out(x)

        return x