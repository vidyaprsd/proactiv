import torch
from torch import nn as nn

class TrainableTensor(nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor         = nn.Parameter(tensor)

    def forward(self) -> torch.Tensor:
        return self.tensor