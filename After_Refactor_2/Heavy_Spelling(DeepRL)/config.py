class Config:
    """
    Configuration parameters for the DQN algorithm.
    """
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    MEMORY_CAPACITY = 100000
    LEARNING_RATE = 0.0005
    NOISY_LAYER_STD = 0.5