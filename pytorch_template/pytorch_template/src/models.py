import torch.nn as nn

# TODO 2: Write your model architecture in a new model class here

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(24, 24, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(24, 12, kernel_size=3, padding=1))

    def forward(self, x):
        return self.model(x)

