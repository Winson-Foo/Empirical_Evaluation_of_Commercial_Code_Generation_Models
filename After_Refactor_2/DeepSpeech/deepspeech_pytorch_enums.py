import torch.nn as nn

from enum import Enum


class DecoderType(Enum):
    greedy = 'greedy'
    beam = 'beam'


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class RNNType(Enum):
    lstm = nn.LSTM
    rnn = nn.RNN
    gru = nn.GRU


class Config:
    # Set constants or configurations that may change over time in a single place.
    SPECTRO_WINDOW_TYPE = SpectrogramWindow.hamming
    DECODER_TYPE = DecoderType.greedy
    RNN_TYPE = RNNType.lstm


def get_rnn_type() -> type(nn.Module):
    # Return selected RNN type based on configuration.
    return Config.RNN_TYPE.value


def get_spectrogram_window_type() -> str:
    # Return selected spectrogram window type based on configuration.
    return Config.SPECTRO_WINDOW_TYPE.value


def get_decoder_type() -> str:
    # Return selected decoder type based on configuration.
    return Config.DECODER_TYPE.value