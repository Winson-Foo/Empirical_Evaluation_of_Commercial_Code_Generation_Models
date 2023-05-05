import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
    UniDirectionalConfig
from deepspeech_pytorch.training import train


def handle_error(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            OmegaConf.to_yaml(kwargs)
    return wrapper


cs = ConfigStore.instance()

try:
    cs.store(name="config", node=DeepSpeechConfig)
    cs.store(group="optim", name="sgd", node=SGDConfig)
    cs.store(group="optim", name="adam", node=AdamConfig)
    cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
    cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
    cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)
except Exception as e:
    print(f"An error occurred: {e}")
    OmegaConf.to_yaml(kwargs)
    
    
@handle_error
@hydra.main(config_path='.', config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()