from uuid import uuid4
from dataclasses import dataclass

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
    num_runs: int = 1
    run_group: str | None = None
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


def main(config_file: str = None, **overrides):
    config: Config = OmegaConf.structured(Config)
    if config_file is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    config = OmegaConf.merge(config, overrides)
    config = OmegaConf.create(config)

    run_group = config.run_group or str(uuid4())
    for _ in range(config.num_runs):
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


if __name__ == "__main__":
    import fire

    fire.Fire(main)
