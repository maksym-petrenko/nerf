import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class NeRF(nn.Module):

    def __init__(self,
            pos_L: int = 10,
            cam_L: int = 4,
            device: str = "cuda"
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
            nn.Linear(256, 256), nn.ReLU()
        )

        self.block_3 = nn.Sequential(nn.Linear(self.cam_L * 4 + 258, 128), nn.ReLU())

        self.block_4 = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())

        self.device = device


    @staticmethod
    def positional_encoding(x, L):
        encoding = torch.zeros(x.shape[0], L * 2 * x.shape[1])
        for power in range(L):
            encoding[:, power * 2 * L:(power * 2 + 1)*L] = torch.sin((2 ** power) * torch.pi * x)
            encoding[:, (power * 2 + 1)*L:(power * 2 + 2)*L] = torch.cos((2 ** power) * torch.pi * x)

        return torch.cat((x, encoding), dim=1)

    def forward(self, position, direction):
        encoded_pos = self.positional_encoding(position, self.pos_L)
        encoded_dir = self.positional_encoding(direction, self.cam_L)

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

    def render_image(self, origins, directions, tn = 0, tf = 1, samples = 128):

        device = origins.device
        t = torch.linspace(tn, tf, samples, device=device).expand(origins.shape[0], samples)

        mid = (t[:-1] + t[1:]) / 2.
        lower = torch.cat((t[:1], mid), -1)
        upper = torch.cat((mid, t[-1:]), -1)
        u = torch.rand(t.shape, device=device)
        t = lower + (upper - lower) * u

        delta = t[1:] - t[:-1]

        coords = origins + t * directions

        colors, sigma = self(coords, directions)

        alpha = 1 - torch.exp(-sigma * delta)

        weights = self.compute_t(alpha) * alpha
        c = (weights * colors).sum(dim=1)

        return c

    @torch.no_grad()
    def test(self, tn, tf, dataset, chunk_size=10, img_index=0, samples=128, height=400, width=400):
        device = self.device

        origins = dataset[img_index * height * width: (img_index + 1) * height * width, :3]
        directions = dataset[img_index * height * width: (img_index + 1) * height * width, 3:6]

        data = []
        for i in range(int(np.ceil(height / chunk_size))):
            ray_origins = origins[i * width * chunk_size: (i + 1) * width * chunk_size].to(device)
            ray_directions = directions[i * width * chunk_size: (i + 1) * width * chunk_size].to(device)

            regenerated_px_values = self.render_image(model, ray_origins, ray_directions, tn=tn, tf=tf, samples=samples)
            data.append(regenerated_px_values)
        img = torch.cat(data).data.cpu().numpy().reshape(height, width, 3)

        plt.figure()
        plt.imshow(img)
        plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
        plt.close()

    def train(self, optimizer, scheduler, train_data, test_data, tn=0, tf=1, epochs=1, samples=128, height=400, width=400):
        training_loss = []
        for _ in range(epochs):
            for batch in tqdm(train_data):
                origins = batch[:, :3].to(self.device)
                directions = batch[:, 3:6].to(self.device)
                ground_truth_px_values = batch[:, 6:].to(self.device)

                regenerated_px_values = self.render_image(origins, directions, tn=tn, tf=tf, samples=samples)
                loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())
            scheduler.step()

            for img_index in range(200):
                self.test(tn, tf, test_data, img_index=img_index, samples=samples, height=height, width=width)
        return training_loss


device = 'cpu'

training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

model = NeRF().to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1024, shuffle=True)

model.train(model, model_optimizer, scheduler, data_loader, epochs=16, tn=2, tf=6, samples=128, height=400, width=400)
