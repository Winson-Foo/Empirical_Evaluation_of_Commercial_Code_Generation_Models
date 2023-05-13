import torch

class Config:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ...