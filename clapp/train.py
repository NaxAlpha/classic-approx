import os
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as VU
import torchvision.transforms.functional as VF

from clapp.model import FilterNet
from clapp.data import ImageFilterStream

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    device: str = "auto"
    batch_size: int = 16
    num_workers: int = 0
    num_iterations: int = 5000
    learning_rate: float = 1e-3
    output_dir: str = "outputs"
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
        self.optim = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.loss_fn = nn.MSELoss()

        self.dataset = dataset
        self.loader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.step_id = 0
        self.progress: tqdm = None
        self.output_file = self.configure_output_file()

    def configure_output_file(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        dir_id = len(os.listdir(self.config.output_dir))
        output_fn = os.path.join(self.config.output_dir, f"{dir_id:03d}")
        return output_fn

    def log_images(self, inputs, targets, outputs, count=3):
        if self.step_id % self.config.output_log_interval != 0:
            return

        targets = targets.expand(-1, 3, -1, -1)
        outputs = outputs.expand(-1, 3, -1, -1).clamp(0, 1)  # fixes noisy image output bug
        grid = torch.stack([inputs[:count], targets[:count], outputs[:count]], dim=1)
        grid = grid.view(-1, *grid.shape[2:])
        grid = VU.make_grid(grid.cpu(), nrow=3)  # .cpu() because of pytorch/vision#6533
        grid = VF.to_pil_image(grid)

        grid.save(f"{self.output_file}.png")
        if wandb and wandb.run:
            wandb.log({"images": wandb.Image(grid)}, step=self.step_id)

    def log_loss(self, loss):
        self.progress.set_postfix(loss=loss)
        if wandb and wandb.run:
            wandb.log({"loss": loss}, step=self.step_id)

    def train_step(self, batch):
        self.step_id += 1

        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss: torch.Tensor = self.loss_fn(outputs, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.log_images(inputs, targets, outputs)

        return loss.item()

    def train(self):
        self.step_id = 0

        self.progress = tqdm(self.loader)
        self.model.train()
        iterator = zip(range(self.config.num_iterations), self.progress)
        for _, batch in iterator:
            loss = self.train_step(batch)
            self.log_loss(loss)
