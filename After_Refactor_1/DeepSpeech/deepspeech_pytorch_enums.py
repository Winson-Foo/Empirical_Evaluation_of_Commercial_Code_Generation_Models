from enum import Enum
from torch import nn

class DecoderType(Enum):
    """
    Enum for different types of decoders
    """
    GREEDY = 'greedy'
    BEAM = 'beam'

class SpectrogramWindow(Enum):
    """
    Enum for different types of spectrogram windows
    """
    HAMMING = 'hamming'
    HANN = 'hann'
    BLACKMAN = 'blackman'
    BARTLETT = 'bartlett'

class RNNType(Enum):
    """
    Enum for different types of RNNs
    """
    LSTM = nn.LSTM
    RNN = nn.RNN
    GRU = nn.GRU