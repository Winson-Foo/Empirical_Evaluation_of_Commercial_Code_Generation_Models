import argparse
import os

import torch
from scipy.io.wavfile import write

from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0,
                    help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

if not os.path.exists(args.input_path):
    raise FileNotFoundError(f"{args.input_path} does not exist.")
if not os.path.exists(args.noise_path):
    raise FileNotFoundError(f"{args.noise_path} does not exist.")
    
noise_injector = NoiseInjection()

try:
    data = load_audio(args.input_path)
except Exception as e:
    print(f"Error occured while loading data from {args.input_path}. {e}")
    exit(1)

mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level)

mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim

try:
    write(filename=args.output_path,
          rate=args.sample_rate,
          data=mixed_data.numpy())
except Exception as e:
    print(f"Error occured while writing to {args.output_path}. {e}")
    exit(1)

print(f'Saved mixed audio to {args.output_path}')