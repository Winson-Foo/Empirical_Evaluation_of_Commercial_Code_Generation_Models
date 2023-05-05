from enum import Enum
import torch

class DecoderType(Enum):
    GREEDY = 'greedy'
    BEAM = 'beam'


class SpectrogramWindow(Enum):
    HAMMING = 'hamming'
    HANN = 'hann'
    BLACKMAN = 'blackman'
    BARTLETT = 'bartlett'


class RNNType(Enum):
    LSTM = 'lstm'
    RNN = 'rnn'
    GRU = 'gru'

    @staticmethod
    def get_rnn_type(rnn_type_str):
        if rnn_type_str.lower() == RNNType.LSTM.value:
            return RNNType.LSTM 

        elif rnn_type_str.lower() == RNNType.GRU.value:
            return RNNType.GRU

        elif rnn_type_str.lower() == RNNType.RNN.value:
            return RNNType.RNN

        else:
            raise ValueError("Invalid RNN Type Provided")

def main():
    try:
        # Example usage
        rnn_type = RNNType.get_rnn_type('lstm')
        print(f"RNN Type {rnn_type}")
    except ValueError as ve:
        print(ve.args)

if __name__ == "__main__":
    main()