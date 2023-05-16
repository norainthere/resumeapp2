import torch
import torch.nn as nn

class EQModel(nn.Module):
    def __init__(self):
        super(EQModel, self).__init__()
        # Define the layers of your model
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)
