from enum import Enum
import torch

class DecoderType(Enum):
    greedy = 'greedy'
    beam = 'beam'


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class RNNType(Enum):
    lstm = torch.nn.LSTM
    rnn = torch.nn.RNN
    gru = torch.nn.GRU

# Define a function to handle errors
def get_enum_value(enum_name, value):
    try:
        return enum_name[value].value
    except KeyError:
        return None

# Test the function using an invalid value
decoder = get_enum_value(DecoderType, 'invalid')
print(decoder) # Output: None