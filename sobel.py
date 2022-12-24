# Approximate Sobel Filter using a Pytorch network
import time
import random
from threading import Thread

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from tqdm import tqdm
from datasets import load_dataset


def transform_function(image: Image.Image):
    # this can be any arbitrary image transformation function
    # that does not create new information
    img_np = np.array(image.convert("L"))
    img_blur = cv2.GaussianBlur(img_np, (3, 3), 0)
    grad_x = cv2.Sobel(img_blur, cv2.CV_16S, 1, 0, ksize=3).astype(np.float32)
    grad_y = cv2.Sobel(img_blur, cv2.CV_16S, 0, 1, ksize=3).astype(np.float32)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255) / grad.max()
    return grad_norm.astype(np.uint8)


class ImageNetSobel(IterableDataset):
    def __init__(self):
        super().__init__()
        self.dataset = load_dataset(
            "imagenet-1k",
            split="train",
            streaming=True,
            use_auth_token=True,
        )
        self._buffer = []
        self._min_buffer = 100
        self._max_buffer = 1000

    def downloader(self):
        transform = VT.Compose(
            [
                VT.Resize(256),
                VT.CenterCrop(224),
            ]
        )
        while True:
            for x in self.dataset:
                image = transform(x["image"].convert("RGB"))
                sobel = transform_function(image)
                image, sobel = map(VT.ToTensor(), (image, sobel))
                self._buffer.append((image, sobel))
                self._buffer = self._buffer[-self._max_buffer :]

    def __iter__(self):
        Thread(
            target=self.downloader,
            daemon=True,
        ).start()
        while len(self._buffer) < self._min_buffer:
            time.sleep(0.01)
        worker_info = get_worker_info()
        seed = worker_info.seed if worker_info else None
        if seed:
            torch.manual_seed(seed)
        rnd = random.Random(seed)
        while True:
            yield rnd.choice(self._buffer)


class FilterNet(nn.Module):
    def __init__(self, f_inp=3, f_out=1, capacity=4, n_layers=2):
        super().__init__()
        self.base = nn.Conv2d(f_inp, capacity, 3, padding=1)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(capacity, capacity, 3, padding=1))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Conv2d(capacity, f_out, 3, padding=1)

    def forward(self, x):
        x = self.base(x)
        for layer in self.layers:
            skip = x
            # ---
            x = x / x.norm(dim=1, keepdim=True)
            x = layer(x)
            x = F.relu(x) ** 2
            # ---
            x = x + skip
        x = self.out(x)
        return x


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "mps"
        self.model = FilterNet().to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.dataset = ImageNetSobel()
        self.loader = DataLoader(
            self.dataset,
            batch_size=16,
            num_workers=0,
            pin_memory=True,
        )
        # ---
        print(
            "Number of parameters:",
            sum(p.numel() for p in self.model.parameters()),
        )

    def train(self):
        progress = tqdm(self.loader)
        self.model.train()
        for img, sbl in progress:
            img, sbl = img.to(self.device), sbl.to(self.device)
            self.optim.zero_grad()
            pred = self.model(img)
            loss = self.loss_fn(pred, sbl)
            loss.backward()
            self.optim.step()
            progress.set_postfix(loss=loss.item())
            if progress.n % 10 == 0:
                grid = torch.cat(
                    [
                        img[0],
                        sbl[0].expand(3, -1, -1),
                        pred[0].expand(3, -1, -1),
                    ],
                    dim=2,
                )
                VT.ToPILImage()(grid).save("grid.png")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
