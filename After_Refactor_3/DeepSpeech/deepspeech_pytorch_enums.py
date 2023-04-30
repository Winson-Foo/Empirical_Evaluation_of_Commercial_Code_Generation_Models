from enum import Enum
from typing import Type

import torch.nn as nn


class DecoderType(Enum):
    GREEDY = 'greedy'
    BEAM = 'beam'


class SpectrogramWindow(Enum):
    HAMMING = 'hamming'
    HANN = 'hann'
    BLACKMAN = 'blackman'
    BARTLETT = 'bartlett'


class RNNType(Enum):
    LSTM = nn.LSTM
    RNN = nn.RNN
    GRU = nn.GRU


class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 rnn_type: Type[nn.Module], num_layers: int, dropout: float):
        super().__init__()
        self.rnn = rnn_type(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = self.initialize_hidden_state(x)
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :])
        return out

    def initialize_hidden_state(self, x):
        return torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)


class SpectrogramProcessor(nn.Module):
    def __init__(self, window_size: int, window_stride: int, window_fn: str):
        super().__init__()
        self.window = self.get_window(window_fn, window_size)
        self.stride = window_stride

    def forward(self, x):
        x = self.apply_window(x)
        return x

    def apply_window(self, x):
        window = self.window.to(x.device)
        x = x.unfold(2, window.shape[0], self.stride)
        x = x * window
        x = x.transpose(1, 2).contiguous()
        return x

    @staticmethod
    def get_window(window_fn: str, window_size: int):
        if window_fn == SpectrogramWindow.HAMMING:
            return nn.hamming_window(window_size)
        elif window_fn == SpectrogramWindow.HANN:
            return nn.hann_window(window_size)
        elif window_fn == SpectrogramWindow.BLACKMAN:
            return nn.blackman_window(window_size)
        elif window_fn == SpectrogramWindow.BARTLETT:
            return nn.bartlett_window(window_size)
        else:
            raise ValueError(f"Invalid window function {window_fn}")


class SpeechRecognizer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 rnn_type: Type[nn.Module], num_layers: int, dropout: float,
                 window_size: int, window_stride: int, window_fn: str, decoder_type: str,
                 beam_size: int = 10):
        super().__init__()
        self.preprocess = SpectrogramProcessor(window_size, window_stride, window_fn)
        self.rnn = RNNModel(input_size, hidden_size, output_size, rnn_type, num_layers, dropout)
        self.decoder_type = decoder_type
        self.beam_size = beam_size

    def forward(self, x):
        x = self.preprocess(x)
        x = self.rnn(x)
        if self.decoder_type == DecoderType.GREEDY:
            out = x.argmax(dim=-1)
        elif self.decoder_type == DecoderType.BEAM:
            out = self.beam_decode(x)
        else:
            raise ValueError(f"Invalid decoder type {self.decoder_type}")
        return out

    def beam_decode(self, x):
        pass # implementation of beam search algorithm here