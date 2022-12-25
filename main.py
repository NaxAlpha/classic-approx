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
    num_runs: int = 5
    num_parallel_runs: int = 0
    run_group: Union[str, None] = None
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


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
    model = FilterNet(config.model)
    dataset = ImageFilterStream(config.data)
    trainer = SimpleTrainer(config.train, model, dataset)
    trainer.train()
    if wandb is not None:
        wandb.finish()


def main(config_file: str = None, **overrides):
    config: Config = OmegaConf.structured(Config)
    if config_file is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    config = OmegaConf.merge(config, overrides)
    config = OmegaConf.create(config)

    run_group = config.run_group or str(uuid4())
    if config.num_parallel_runs == 0:
        for _ in range(config.num_runs):
            worker_func(run_group, config)
    else:
        with ProcessPoolExecutor(max_workers=config.num_parallel_runs) as executor:
            futures = []
            for _ in range(config.num_runs):
                futures.append(executor.submit(worker_func, run_group, config))
            for future in futures:
                future.result()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
