import os
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.utils as VU
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms.functional as VF
import torch.optim.lr_scheduler as lr_scheduler

from clapp.ema import EMA
from clapp.model import FilterNet
from clapp.data import ImageFilterStream

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 0
    max_iterations: int = 1000
    stop_l2_loss: float = 3e-3
    stop_consecutives: int = 10
    stop_loss_ema: float = 0.99
    learning_rate: float = 1e-3
    output_dir: str = "outputs"
    output_log_interval: int = 100
    min_lr: float = 1e-5
    lr_cycle: int = 100
    loss_type: str = "all"


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


def log_cosh_loss(preds, targets):
    # adapted from: https://github.com/tuantle/regression-losses-pytorch
    return torch.mean(torch.log(torch.cosh(preds - targets + 1e-12)))


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
        self.optim = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim,
            T_0=config.lr_cycle,
            T_mult=2,
            eta_min=config.min_lr,
        )

        self.dataset = dataset
        self.loader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        self.ema = EMA(config.stop_loss_ema)
        self.stop_counter = 0
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
        outputs = outputs.expand(-1, 3, -1, -1).clamp(0, 1)
        # ^ fixes noisy image output bug
        grid = torch.stack([inputs[:count], targets[:count], outputs[:count]], dim=1)
        grid = grid.view(-1, *grid.shape[2:])
        grid = VU.make_grid(grid.cpu(), nrow=3)  # .cpu() because of pytorch/vision#6533
        grid = VF.to_pil_image(grid)

        grid.save(f"{self.output_file}.png")
        if wandb and wandb.run:
            wandb.log({"images": wandb.Image(grid)}, step=self.step_id)

    def log_loss(self, losses):
        loss_dict = {f"loss/{k}": v for k, v in losses.items()}
        self.progress.set_postfix(**loss_dict)
        if wandb and wandb.run:
            lr = self.optim.param_groups[0]["lr"]
            wandb.log(
                {
                    "lr": lr,
                    **loss_dict,
                },
                step=self.step_id,
            )

    def stop_test(self, losses):
        ema_losses = self.ema.update(losses)
        if ema_losses["l2"] < self.config.stop_l2_loss:
            self.stop_counter += 1
        else:
            self.stop_counter = 0
        if self.stop_counter >= self.config.stop_consecutives:
            return True

    def train_step(self, batch):
        self.step_id += 1

        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        losses = {
            "l1": F.l1_loss(outputs, targets),
            "l2": F.mse_loss(outputs, targets),
            "lc": log_cosh_loss(outputs, targets),
        }
        losses["all"] = sum(losses.values())
        loss = losses[self.config.loss_type]
        losses = {k: v.item() for k, v in losses.items()}

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        self.log_images(inputs, targets, outputs)
        self.log_loss(losses)
        if self.stop_test(losses):
            raise StopIteration

        return losses

    def train(self):
        self.step_id = 0

        self.progress = tqdm(self.loader)
        self.model.train()
        iterator = zip(range(self.config.max_iterations), self.progress)
        for _, batch in iterator:
            try:
                losses = self.train_step(batch)
            except StopIteration:
                break

        return self.step_id, losses
