from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms.functional as VF

from clapp.model import FilterNet
from clapp.data import ImageFilterStream


@dataclass
class TrainConfig:
    device: str = "auto"
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    output_dir: str = "output"
    output_log_interval: int = 100


def get_device(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        try:  # M1 MacOS support
            import torch.backends.mps as mps

            if mps.is_available():
                return torch.device("mps")
        except ImportError:
            pass
        return torch.device("cpu")
    return torch.device(name)


class SimpleTrainer:
    def __init__(
        self,
        config: TrainConfig,
        model: FilterNet,
        dataset: ImageFilterStream,
    ):
        self.config = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.dataset = dataset
        self.loader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.progress: tqdm = None

    def log_images(self, inputs, targets, outputs, count=3):
        if self.progress.n % self.config.output_log_interval != 0:
            return
        grids = []
        for i in range(count):
            grid = torch.cat(
                [
                    inputs[i],
                    targets[i].expand(3, -1, -1),
                    outputs[i].expand(3, -1, -1),
                ],
                dim=2,
            )
            grids.append(grid)
        grid = torch.cat(grids, dim=1)
        grid: Image.Image = VF.to_pil_image(grid)
        grid.save(f"{self.config.output_dir}/{self.step:05d}.png")

    def log_loss(self, loss):
        self.progress.set_postfix(loss=loss)

    def train_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss: torch.Tensor = self.loss_fn(outputs, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def train(self):
        self.progress = tqdm(self.loader)
        self.model.train()
        for batch in self.progress:
            loss = self.train_step(batch)
            self.log_loss(loss)
