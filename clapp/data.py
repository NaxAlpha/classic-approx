import time
import random
from threading import Thread
from dataclasses import dataclass

import cv2
import datasets
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


@dataclass
class DataConfig:
    name: str = "imagenet-1k"
    split: str = "train"
    min_buffer: int = 100
    max_buffer: int = 1000
    resize_base: int = 256
    crop_size: int = 224
    image_key: str = "image"


def sobel_filter(image: Image.Image, blur=(3, 3), kernel=3):
    img_np = np.array(image.convert("L"))
    img_blur = cv2.GaussianBlur(img_np, blur, 0)
    grad_x = cv2.Sobel(img_blur, cv2.CV_16S, 1, 0, ksize=kernel).astype(np.float32)
    grad_y = cv2.Sobel(img_blur, cv2.CV_16S, 0, 1, ksize=kernel).astype(np.float32)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255) / grad.max()
    return grad_norm.astype(np.uint8)


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

    def downloader(self):
        transform = VT.Compose(
            [
                VT.Resize(self.config.resize_base),
                VT.CenterCrop(self.config.crop_size),
            ]
        )
        while True:
            for x in self.dataset:
                image = transform(x[self.config.image_key].convert("RGB"))
                sobel = sobel_filter(image)
                image, sobel = map(VF.to_tensor, (image, sobel))
                self._buffer.append((image, sobel))
                if len(self._buffer) > self._max_buffer:
                    del self._buffer[0]

    def __iter__(self):
        Thread(
            target=self.downloader,
            daemon=True,
        ).start()
        while len(self._buffer) < self._min_buffer:
            time.sleep(0.01)
        worker_info = data.get_worker_info()
        seed = worker_info.seed if worker_info else None
        if seed:
            torch.manual_seed(seed)
        rnd = random.Random(seed)
        while True:
            yield rnd.choice(self._buffer)
