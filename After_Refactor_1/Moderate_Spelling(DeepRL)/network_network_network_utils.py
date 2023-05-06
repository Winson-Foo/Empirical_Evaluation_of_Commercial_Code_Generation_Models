# In base_model.py module
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def reset_noise(self):
        pass


def layer_init(layer):
    nn.init.xavier_uniform_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0)
    return layer

    