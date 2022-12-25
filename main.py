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


def main(config_file: str = None, **overrides):
    config: Config = OmegaConf.structured(Config)
    if config_file is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    config = OmegaConf.merge(config, overrides)
    config = OmegaConf.create(config)

    run_group = config.run_group or str(uuid4())
    for _ in range(config.num_runs):
        if wandb is not None:
            wandb.init(
                project="clapp",
                group=run_group,
                config=config,
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
