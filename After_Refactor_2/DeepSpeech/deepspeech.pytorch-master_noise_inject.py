import argparse
import torch
from scipy.io.wavfile import write
from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection

DEFAULT_INPUT_PATH = 'input.wav'
DEFAULT_NOISE_PATH = 'noise.wav'
DEFAULT_OUTPUT_PATH = 'output.wav'
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NOISE_LEVEL = 1.0

INPUT_PATH_HELP = 'The input audio to inject noise into'
NOISE_PATH_HELP = 'The noise file to mix in'
OUTPUT_PATH_HELP = 'The noise file to mix in'
SAMPLE_RATE_HELP = 'Sample rate to save output as'
NOISE_LEVEL_HELP = 'The Signal to Noise ratio (higher means more noise)'

def inject_noise(args):
    """Inject noise into an audio file."""
    noise_injector = NoiseInjection()
    data = load_audio(args.input_path)
    mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level)
    mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
    write(filename=args.output_path, data=mixed_data.numpy(), rate=args.sample_rate)
    print('Saved mixed file to %s' % args.output_path)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default=DEFAULT_INPUT_PATH, help=INPUT_PATH_HELP)
    parser.add_argument('--noise-path', default=DEFAULT_NOISE_PATH, help=NOISE_PATH_HELP)
    parser.add_argument('--output-path', default=DEFAULT_OUTPUT_PATH, help=OUTPUT_PATH_HELP)
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE, help=SAMPLE_RATE_HELP)
    parser.add_argument('--noise-level', type=float, default=DEFAULT_NOISE_LEVEL, help=NOISE_LEVEL_HELP)
    return parser.parse_args()

def main():
    """Run the program."""
    args = parse_args()
    inject_noise(args)

if __name__ == '__main__':
    main()