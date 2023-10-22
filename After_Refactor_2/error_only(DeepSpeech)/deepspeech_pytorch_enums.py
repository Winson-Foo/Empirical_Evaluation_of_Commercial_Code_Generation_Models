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


def decode_audio(audio_input, model, decoder_type=DecoderType.greedy, beam_width=5, 
                  spectrogram_window=SpectrogramWindow.hamming):
    try:
        audio_input = torch.tensor(audio_input)
        audio_input = audio_input.unsqueeze(0)
        audio_input = audio_input.transpose(2, 1)
        audio_input = audio_input.float()

        if decoder_type == DecoderType.greedy:
            decoded_output = model.greedy_decoder(audio_input, spectrogram_window)
        elif decoder_type == DecoderType.beam:
            decoded_output = model.beam_search_decoder(audio_input, beam_width, spectrogram_window)
        else:
            raise ValueError("Invalid DecoderType selected. Choose from DecoderType Enum.")

        return decoded_output

    except Exception as e:
        print("Error occurred: ", e)
        return None