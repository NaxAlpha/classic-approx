import time
import random
from threading import Thread
from functools import partial
from dataclasses import dataclass

import cv2
import datasets
import numpy as np
from PIL import Image

import torch
import functorch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


def sobel_filter(image: Image.Image, blur=(3, 3), kernel=3):
    img_np = np.array(image.convert("L"))
    img_blur = cv2.GaussianBlur(img_np, blur, 0)
    grad_x = cv2.Sobel(img_blur, cv2.CV_16S, 1, 0, ksize=kernel).astype(np.float32)
    grad_y = cv2.Sobel(img_blur, cv2.CV_16S, 0, 1, ksize=kernel).astype(np.float32)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255) / (grad.max() + 1e-8)
    return grad_norm.astype(np.uint8)


def sobel_canny_filter(image: Image.Image, blur=(3, 3), kernel=3, canny=(50, 150)):
    sobel_grad = sobel_filter(image, blur, kernel)
    return cv2.Canny(sobel_grad, *canny)


def get_random_kernel(seed=None):
    if not hasattr(get_random_kernel, f"kernel_{seed}"):
        rnd = random.Random(seed)
        ker = [rnd.random() for _ in range(13)]
        ker = [*ker] + [0] + [-k for k in ker]  # sobel-like kernel
        rnd.shuffle(ker)
        ker = torch.tensor(ker).view(1, 3, 3, 3)
        ker = ker / ker.abs().sum()
        mod = nn.Conv2d(3, 1, 3, padding=1, bias=False).requires_grad_(False)
        mod.weight.data = ker
        if seed:
            setattr(get_random_kernel, f"kernel_{seed}", mod)
        return mod
    return getattr(get_random_kernel, f"kernel_{seed}")


@torch.no_grad()
def random_filter(image: Image.Image, seed=None):
    images = VF.to_tensor(image).unsqueeze(0)
    kernel = get_random_kernel(seed)
    output = kernel(images).clamp(0, 1)
    return VF.to_pil_image(output[0])


@torch.no_grad()
def random_p2_filter(image: Image.Image, seeds=(51, 15)):
    a, b = seeds
    k1, k2 = get_random_kernel(a), get_random_kernel(b)
    images = VF.to_tensor(image).unsqueeze(0)
    output = torch.stack([k1(images), k2(images)])
    output = output.pow(2).sum(dim=0).sqrt()
    output = output / (output.max() + 1e-8)
    return VF.to_pil_image(output[0])


target_filters = {
    "sobel_3": partial(sobel_filter, kernel=3),
    "sobel_canny_5": partial(sobel_canny_filter, kernel=5),
    "random_41": partial(random_filter, seed=41),
    "random2_51_15": partial(random_p2_filter, seeds=(51, 15)),
}


@dataclass
class DataConfig:
    name: str = "imagenet-1k"
    split: str = "train"
    min_buffer: int = 1000
    max_buffer: int = 5000
    resize_base: int = 96
    buffer_delay: float = 0.1
    crop_size: int = 64
    image_key: str = "image"
    target_filter: str = "sobel_3"


class ImageFilterStream(data.IterableDataset):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.dataset = datasets.load_dataset(
            config.name,
            split=config.split,
            streaming=True,
            use_auth_token=True,
        )
        self._buffer = []
        self._min_buffer = config.min_buffer
        self._max_buffer = config.max_buffer

    def update_resolution(self, resize_base, crop_size):
        self.config.resize_base = resize_base
        self.config.crop_size = crop_size
        self._buffer.clear()

    def downloader(self):
        while True:
            for x in self.dataset:
                transform = VT.Compose(
                    [
                        VT.Resize(self.config.resize_base),
                        VT.RandomCrop(self.config.crop_size),
                    ]
                )
                image = transform(x[self.config.image_key].convert("RGB"))
                sobel = target_filters[self.config.target_filter](image)
                image, sobel = map(VF.to_tensor, (image, sobel))
                self._buffer.append((image, sobel))
                if len(self._buffer) > self._max_buffer:
                    del self._buffer[0]
                    time.sleep(self.config.buffer_delay)

    def __iter__(self):
        Thread(
            target=self.downloader,
            daemon=True,
        ).start()
        worker_info = data.get_worker_info()
        seed = worker_info.seed if worker_info else None
        if seed:
            torch.manual_seed(seed)
        rnd = random.Random(seed)
        # ---
        while True:
            while len(self._buffer) < self._min_buffer:
                time.sleep(0.01)
            try:
                yield rnd.choice(self._buffer)
            except:
                break
