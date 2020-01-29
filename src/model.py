from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


# class FirstModel(nn.Module):
#     def __init__(self, num_categories, len_dense, weighs):
#         super(FirstModel, self).__init__()
#         self.model_conv = models.resnet152(pretrained=False)
#         if weighs:
#             self.model_conv.load_state_dict(torch.load(weighs))
#         self.model_conv.fc = nn.Linear(self.model_conv.fc.in_features, num_categories)
#         self.model_dense = nn.Linear(len_dense, num_categories)
#         self.model = nn.Linear(2*num_categories, num_categories)
    
#     def forward(self, x, y):
#         x1 = self.model_conv(x)
#         x2 = self.model_dense(y)
#         x = F.relu(torch.cat((x1, x2), 1))
#         return self.model(x)
    
    
    
class FirstModel(nn.Module):
    """Some Information about FirstModel"""
    def __init__(self, weights=None):
        super (FirstModel, self).__init__()
        self.resnet = models.resnet152(pretrained=False)
        if weights:
            self.resnet.load_state_dict(torch.load(weights))
        self.resnet.fc = nn.F.linear(in_features = self.resnet.fc.in_features, out_features = 23)
        self.fc1 = nn.Linear(in_features=k1, out_features=23)
        self.out = nn.Linear(in_features=k2, out_features=23)

    def forward(self, x, y):
        x1 = self.resnet(x)
        x2 = self.fc1(y)
        x = F.relu(torch.cat((x1, x2), dim=1))
        x = self.out(x)

        return x