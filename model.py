import torch.nn as nn
import torch


class NeRF(nn.Module):

    def __init__(self,
            L: int,
            hidden_layers: int,
            hidden_dim: int
        ) -> None:
        super.__init__()
        self.L = L




    def positional_encoding(self, x):
        encoding = list(x)
        for power in range(self.L):
            encoding.extend([
                torch.sin((2 ** power) * torch.pi * x),
                torch.cos((2 ** power) * torch.pi * x)
            ])
        return torch.cat(encoding, dim=-1)

    def forward(self, x):
        return self.model(self.positional_encoding(x))

model = NeRF(2, 1, 1)

print(model.positional_encoding([1, 2, 3, 4, 5]))