# brdf_nn_training.py : Training for the Neural BRDF model

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


class BRDFModel(nn.Module):
    def __init__(self, hidden=128, depth=6):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.eval()


class BRDFNeuralTraining:
    def __init__(self):
        img = Image.open("assets/brdf_ground_truth.png").convert("RGB")
        img_tensor = TF.to_tensor(img)
        dimension = img_tensor.shape[-1]

        # Build input grid: NdotV (x-axis) and roughness (y-axis)
        ndotv = torch.linspace(0.0, 1.0, dimension)
        roughness = torch.linspace(0.0, 1.0, dimension)
        grid_r, grid_n = torch.meshgrid(roughness, ndotv, indexing="ij")

        self.X = torch.stack([grid_n.flatten(), grid_r.flatten()], dim=1)

        scale = img_tensor[0].flatten()
        bias = img_tensor[1].flatten()
        self.Y = torch.stack([scale, bias], dim=1)

    def infer(self, model: BRDFModel, path: str):
        with torch.no_grad():
            y_pred = model(self.X)

        dimension = int(self.X.shape[0] ** 0.5)
        scale = y_pred[:, 0].reshape(dimension, dimension)
        bias = y_pred[:, 1].reshape(dimension, dimension)

        # Build RGB image (B channel = 0)
        img_tensor = torch.stack([scale, bias, torch.zeros_like(scale)], dim=0)
        img = TF.to_pil_image(img_tensor.clamp(0, 1))
        img.save(path)

    def train(self, model: BRDFModel, lr: float, epochs: int, batch_size: int = 4096):
        dataset = TensorDataset(self.X, self.Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, Y_batch in loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.6f}")
