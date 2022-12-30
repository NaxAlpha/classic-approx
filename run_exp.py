from uuid import uuid4
from typing import Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf

from clapp.model import ModelConfig, FilterNet
from clapp.train import TrainConfig, SimpleTrainer
from clapp.data import DataConfig, ImageFilterStream

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class Config(OmegaConf):
    num_runs: int = 10
    num_resolutions: int = 2
    num_parallel_runs: int = 1
    experiment_name: Union[str, None] = None
    train_data: DataConfig = DataConfig(
        split="train",
        resize_base=96,
        crop_size=64,
    )
    valid_data: DataConfig = DataConfig(
        split="validation",
        resize_base=256,
        crop_size=224,
        buffer_delay=0.2,
    )
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(config_file: str, overrides: dict) -> Config:
    config = OmegaConf.structured(Config)
    if config_file is not None:
        config = OmegaConf.load(config_file)
    dot_list = [f'{k}={v}' for k, v in overrides.items()]
    override = OmegaConf.from_dotlist(dot_list)
    config = OmegaConf.merge(config, override)
    config = OmegaConf.create(config)
    return config


def train_model(config: Config, verbose: bool = True):
    model = FilterNet(config.model)
    param = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"==> Number of model parameters: {param}")
    train_dataset = ImageFilterStream(config.train_data)
    valid_dataset = ImageFilterStream(config.valid_data)
    last_step_id = 0
    resize_init = config.train_data.resize_base
    crop_init = config.train_data.crop_size
    trainer = SimpleTrainer(
        config=config.train,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        verbose=verbose,
    )
    for i in range(config.num_resolutions):
        trainer.step_id = last_step_id
        trainer.train(
            desc=f"Training {i + 1}/{config.num_resolutions}",
        )
        last_step_id = trainer.step_id
        config.train.min_iterations += last_step_id
        # --- Workaround for doubling the resolution
        resize_init *= 2
        crop_init *= 2
        train_dataset.update_resolution(resize_init, crop_init)
        next(trainer.train_loader)
        config.train.batch_size //= 2
        # ---
    return model


def flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def worker_func(run_group, config: Config):
    if wandb is not None:
        flat_config = flatten_dict(OmegaConf.to_container(config))
        wandb.init(
            project="clapp",
            group=run_group,
            config=flat_config,
        )
    train_model(config)
    if wandb is not None:
        wandb.finish()


def main(config_file: str = None, **overrides):
    config: Config = load_config(config_file, overrides)
    print("=== Config ===")
    print(OmegaConf.to_yaml(config))
    print("=============")

    run_group = config.experiment_name or str(uuid4())
    if config.num_parallel_runs == 0:
        for _ in range(config.num_runs):
            worker_func(run_group, config)
    else:
        with ProcessPoolExecutor(max_workers=config.num_parallel_runs) as executor:
            list(
                executor.map(
                    worker_func,
                    [run_group] * config.num_runs,
                    [config] * config.num_runs,
                )
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
