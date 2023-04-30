import argparse
import os
from typing import Tuple

import torch
from scipy.io.wavfile import write

from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection


def mix_audio(input_path: str, noise_path: str, output_path: str,
              sample_rate: int = 16000, noise_level: float = 1.0) -> None:
    """
    Mixes noise from a file into an audio file and saves it to a new file.

    Args:
        input_path (str): Path to the input audio file.
        noise_path (str): Path to the noise file.
        output_path (str): Path to save the mixed audio file.
        sample_rate (int): Sample rate to save the output as.
        noise_level (float): The Signal to Noise ratio (higher means more noise).

    Raises:
        FileNotFoundError: If either input_path or noise_path is invalid.
    """
    noise_injector = NoiseInjection()
    try:
        data = load_audio(input_path)
        mixed_data = noise_injector.inject_noise_sample(data, noise_path, noise_level)
        mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
        with open(output_path, 'wb') as f:
            write(f, sample_rate, mixed_data.numpy())
        print(f'Saved mixed file to {output_path}')
    except FileNotFoundError:
        print('Invalid input_path or noise_path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='input.wav', help='Path to the input audio file.')
    parser.add_argument('--noise-path', default='noise.wav', help='Path to the noise file.')
    parser.add_argument('--output-path', default='output.wav', help='Path to save the mixed audio file.')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate to save the output as.')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='The Signal to Noise ratio (higher means more noise).')
    args = parser.parse_args()

    mix_audio(args.input_path, args.noise_path, args.output_path,
              args.sample_rate, args.noise_level)