from dataclasses import dataclass

from omegaconf import OmegaConf

from clapp.model import ModelConfig, FilterNet
from clapp.train import TrainConfig, SimpleTrainer
from clapp.data import DataConfig, ImageFilterStream


@dataclass
class Config(OmegaConf):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def main(config_file: str = None, **overrides):
    config: Config = OmegaConf.structured(Config)
    if config_file is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    config = OmegaConf.merge(config, overrides)
    config = OmegaConf.create(config)
    model = FilterNet(config.model)
    dataset = ImageFilterStream(config.data)
    trainer = SimpleTrainer(config.train, model, dataset)
    trainer.train()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
