from PIL import Image
from omegaconf import OmegaConf

import torch
import torchvision.transforms.functional as VF

from clapp.model import FilterNet
from clapp.train import get_device

from run_exp import train_model, load_config


def train(
    model_save_path: str,
    config_file: str = None,
    verbose: bool = False,
    **overrides,
):
    config = load_config(config_file, overrides)
    config = OmegaConf.merge(config, overrides)
    model = train_model(config, verbose=verbose)
    torch.save(model.state_dict(), model_save_path)


def infer(
    model_path: str,
    input_path: str,
    output_path: str,
    config_file: str = None,
    **overrides,
):
    config = load_config(config_file, overrides)
    device = get_device(config.train.device)

    model = FilterNet(config.model)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval().to(device)

    image = Image.open(input_path)
    image = VF.to_tensor(image).to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    output = output.squeeze(0)
    output = VF.to_pil_image(output)
    output.save(output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            train=train,
            infer=infer,
        )
    )
