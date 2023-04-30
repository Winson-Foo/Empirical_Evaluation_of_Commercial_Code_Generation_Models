import hydra
from hydra.core.config_store import ConfigStore
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import (
    DeepSpeechConfig,
    AdamConfig,
    SGDConfig,
    BiDirectionalConfig,
    UniDirectionalConfig,
)
from deepspeech_pytorch.training import train


def register_configs():
    # Register all the configuration nodes with Hydra's ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="config", node=DeepSpeechConfig)
    cs.store(group="optim", name="sgd", node=SGDConfig)
    cs.store(group="optim", name="adam", node=AdamConfig)
    cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
    cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
    cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_path=".", config_name="config")
def run_training(cfg: DeepSpeechConfig):
    # Run the training script with the provided configuration
    train(cfg=cfg)


if __name__ == "__main__":
    register_configs()
    run_training()