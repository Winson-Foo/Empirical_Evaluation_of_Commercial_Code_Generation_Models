import hydra.core.config_store.ConfigStore
import hydra.main
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import (
    AdamConfig,
    BiDirectionalConfig,
    DeepSpeechConfig,
    SGDConfig,
    UniDirectionalConfig,
)
from deepspeech_pytorch.training import train

# Define the configuration store with custom configs
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="deepspeech_config", node=DeepSpeechConfig)
config_store.store(group="optim", name="sgd", node=SGDConfig)
config_store.store(group="optim", name="adam", node=AdamConfig)
config_store.store(group="checkpoint", name="file", node=ModelCheckpointConf)
config_store.store(group="model", name="bidirectional", node=BiDirectionalConfig)
config_store.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_path=".", config_name="deepspeech_config")
def main(config: DeepSpeechConfig):
    # Call the training function with the provided configuration
    train(cfg=config)


if __name__ == "__main__":
    main()