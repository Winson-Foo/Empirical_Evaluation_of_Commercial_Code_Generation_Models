from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import List

cs = ConfigStore.instance()


@dataclass
class LMConfig:
    alpha: float = 0.8
    beta: float = 1.5
    lm_path: str = ""
    trie_path: str = ""


@dataclass
class ModelConfig:
    model_path: str = ""
    spect_cfg: dict = field(default_factory=dict)
    precision: int = 16
    cuda: bool = False
    labels: str = ""


@dataclass
class ServerConfig:
    model: ModelConfig = ModelConfig()
    lm: LMConfig = LMConfig()
    host: str = ""
    port: int = 5000
    allowed_extensions: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.ogg', '.webm'])
    debug: bool = False


cs.store(name="config", node=ServerConfig)