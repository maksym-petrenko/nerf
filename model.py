import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Tensor = torch.Tensor

class NeRF(nn.Module):
    def __init__(self,
            pos_L: int = 10,
            cam_L: int = 4,
            device: str = "cpu"
        ) -> None:
        super().__init__()

        self.pos_L = pos_L
        self.cam_L = cam_L

        self.block_1 = nn.Sequential(
            nn.Linear(self.pos_L * 6 + 3, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Linear(self.pos_L * 6 + 259, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 257)
        )

        self.block_3 = nn.Sequential(nn.ReLU(), nn.Linear(self.cam_L * 6 + 259, 128), nn.Sigmoid())

        self.block_4 = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())

        self.device = device

    def positional_encoding(self, x: Tensor, L: int = 1) -> Tensor:
        size = x.shape[1]
        encoding = torch.zeros(x.shape[0], L * 2 * x.shape[1]).to(self.device)
        for power in range(L):
            encoding[:, power * size * 2:power * size * 2 + x.shape[1]] = torch.sin((2 ** (power + 1)) * torch.pi * x).to(self.device)
            encoding[:, power * size * 2 + x.shape[1]:power * size * 2 + 2 * x.shape[1]] = torch.cos((2 ** (power + 1)) * torch.pi * x).to(self.device)

        return torch.cat((x, encoding), dim=1).to(self.device)

    def forward(self, position: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
        encoded_pos = self.positional_encoding(position, self.pos_L)
        encoded_dir = self.positional_encoding(direction, self.cam_L)

        hidden_1 = self.block_1(encoded_pos)
        hidden_input_1 = torch.cat((hidden_1, encoded_pos), dim=1)

        hidden_output_1 = self.block_2(hidden_input_1)
        hidden_input_2, not_activated_sigma = hidden_output_1[:, :-1], hidden_output_1[:, -1]
        sigma = nn.ReLU()(not_activated_sigma)

        hidden_input_3 = torch.cat((hidden_input_2, encoded_dir), dim=1)
        hidden_output_2 = self.block_3(hidden_input_3)

        color = self.block_4(hidden_output_2)

        return color, sigma

    @torch.no_grad()
    def test(self, dataset, img_index: int = 0, epoch: int = 0, chunk_size: int = 10) -> None:
        origins = dataset[img_index * 160000: (img_index + 1) * 160000, :3]
        directions = dataset[img_index * 160000: (img_index + 1) * 160000, 3:6]

        data = []
        for i in range(int(np.ceil(400 / chunk_size))):
            chinked_origins = origins[i * 400 * chunk_size: (i + 1) * 400 * chunk_size].to(device)
            chinked_directions = directions[i * 400 * chunk_size: (i + 1) * 400 * chunk_size].to(device)
            regenerated_px_values = self.render_image(chinked_origins, chinked_directions)
            data.append(regenerated_px_values)

        img = torch.cat(data).data.cpu().numpy().reshape(400, 400, 3)

        plt.figure()
        plt.imshow(img)
        plt.savefig(f'generated_nerf/img_{epoch}_{img_index}.png')
        plt.close()


    def render_image(self, origins: Tensor, directions: Tensor) -> Tensor:
        device = self.device
        batch = origins.shape[0]

        t = torch.linspace(2, 6, 192, device=device).expand(batch, 192)

        mid = (t[:, :-1] + t[:, 1:]) / 2
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        u = torch.rand(t.shape, device=device)
        t = lower + (upper - lower) * u

        delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e12], device=device).expand(batch, 1)), -1)
        delta = delta.unsqueeze(2)

        coords = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)
        directions = directions.expand(192, directions.shape[0], 3).transpose(0, 1)

        colors, sigma = self(coords.reshape(-1, 3), directions.reshape(-1, 3))

        colors = colors.reshape(batch, 192, 3)
        sigma = sigma.reshape(batch, 192, 1)

        alpha = (1 - torch.exp(-(sigma * delta))).squeeze()

        ones = torch.ones(batch, 1, device=device)
        new_tensor = torch.cat((ones, (1 - alpha)[:, :-1]), dim=1)
        T = torch.cumprod(new_tensor, 1).to(device)

        weights = T * alpha

        prod = weights.reshape((batch, 192, 1)) * colors
        c = prod.sum(dim=1)

        weight_sum = weights.reshape((batch, 192, 1)).sum(-1).sum(-1)
        return c + 1 - weight_sum.unsqueeze(-1)


    def train(self, optimizer, scheduler, data_loader, device='cuda', epochs=16) -> list[float]:
        training_loss = []
        for epoch in range(epochs):
            for batch in tqdm(data_loader):
                origins = batch[:, :3].to(device)
                directions = batch[:, 3:6].to(device)
                pixels = batch[:, 6:].to(device)

                loss = ((pixels - self.render_image(origins, directions)) ** 2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())
            scheduler.step()

            for img_index in range(20):
                self.test(testing_dataset, img_index=img_index, epoch=epoch)
        return training_loss

device = 'cpu'

training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

model = NeRF().to(device)

model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
model.train(model_optimizer, scheduler, data_loader, epochs=16, device=device)
