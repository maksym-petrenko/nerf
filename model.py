import torch.nn as nn
import torch


class NeRF(nn.Module):

    def __init__(self,
            L: int,
        ) -> None:
        super().__init__()
        self.L = L

        self.block_1 = nn.Sequential(
            nn.Linear(self.L * 6 + 3, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Linear(self.L * 6 + 259, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

        self.block_3 = nn.Sequential(nn.Linear(self.L * 4 + 258, 128), nn.ReLU())

        self.block_4 = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())


    def positional_encoding(self, x):
        x = torch.tensor(x)

        if len(x.shape) != 1:
            raise ValueError("Input tensor must be 1-dimensional (shape: (n,)).")

        encoding = torch.zeros(x.shape[0], self.L * 2)

        for power in range(self.L):
            encoding[:, power * 2] = torch.sin((2 ** power) * torch.pi * x)
            encoding[:, power * 2 + 1] = torch.cos((2 ** power) * torch.pi * x)

        return torch.cat((torch.flatten(x), torch.flatten(encoding)))

    def forward(self, position, direction):
        encoded_pos = self.positional_encoding(position)
        encoded_dir = self.positional_encoding(direction)

        hidden_1 = self.block_1(encoded_pos)
        hidden_input_1 = torch.cat(hidden_1, encoded_pos)

        hidden_output_1 = self.block_2(hidden_input_1)
        hidden_input_2, sigma = hidden_output_1[:-1], hidden_output_1[-1]

        hidden_input_3 = torch.cat(hidden_input_2, encoded_dir)
        hidden_output_2 = self.block_3(hidden_input_3)

        color = self.block_4(hidden_output_2)

        return color, sigma

    @staticmethod
    def compute_t(alphas):
        transformed = 1 - alphas
        return torch.cumprod(transformed, dim=0)

    def render_image(self, ):
        pass


model = NeRF(1)

arr = torch.tensor([0.5,0.24,0.124])

print(model.compute_t(arr))
