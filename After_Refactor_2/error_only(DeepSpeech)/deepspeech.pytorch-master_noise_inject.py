import argparse
import os

import torch
from scipy.io.wavfile import write

from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
    parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
    parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate to save output as')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='The Signal to Noise ratio (higher means more noise)')
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        print('Error: Input file does not exist')
        return
    if not os.path.isfile(args.noise_path):
        print('Error: Noise file does not exist')
        return

    noise_injector = NoiseInjection()
    data = load_audio(args.input_path)
    try:
        mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level)
    except Exception as e:
        print('Error injecting noise into audio:', e)
        return

    mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dimension
    write(filename=args.output_path,
          rate=args.sample_rate,
          data=mixed_data.numpy())
    print('Saved mixed file to %s' % args.output_path)


if __name__ == '__main__':
    main()