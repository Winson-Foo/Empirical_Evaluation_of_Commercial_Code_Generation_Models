import hydra
from hydra.core.config_store import ConfigStore
from deepspeech_pytorch.configs import (
    DeepSpeechConfig, 
    AdamConfig, 
    SGDConfig, 
    ModelCheckpointConf, 
    BiDirectionalConfig, 
    UniDirectionalConfig
)
from deepspeech_pytorch.training import train

cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)

@hydra.main(config_path='.', config_name="config")
def train_model(cfg: DeepSpeechConfig) -> None:
    train(cfg)

if __name__ == '__main__':
    train_model()