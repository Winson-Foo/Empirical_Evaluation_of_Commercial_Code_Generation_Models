import argparse
import os

import torch
from scipy.io.wavfile import write

from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0, help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

if not os.path.exists(args.input_path):
    print('Input file path does not exist.')
    exit()
    
if not os.path.exists(args.noise_path):
    print('Noise file path does not exist.')
    exit()

noise_injector = NoiseInjection()
data, sampling_rate = load_audio(args.input_path, args.sample_rate)
mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level, sampling_rate)
mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
write(filename=args.output_path, rate=args.sample_rate, data=mixed_data.numpy())
print('Saved mixed file to %s' % args.output_path)