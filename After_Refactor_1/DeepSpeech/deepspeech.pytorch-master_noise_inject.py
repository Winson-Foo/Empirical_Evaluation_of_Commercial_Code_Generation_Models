import argparse
import torch
from scipy.io.wavfile import write
from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection

def main(args):
    # Load audio data
    audio_data = load_audio(args.input_path)

    # Inject noise into audio
    noise_injector = NoiseInjection()
    mixed_data = noise_injector.inject_noise_sample(audio_data, args.noise_path, args.noise_level)

    # Convert data to tensor and add channels dimension
    mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)

    # Write mixed audio data to file
    write(filename=args.output_path, data=mixed_data.numpy(), rate=args.sample_rate)
    print('Saved mixed file to %s' % args.output_path)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='input.wav', help='Path to input audio file')
    parser.add_argument('--noise-path', default='noise.wav', help='Path to noise audio file')
    parser.add_argument('--output-path', default='output.wav', help='Path to output mixed audio file')
    parser.add_argument('--sample-rate', default=16000, help='Sample rate to save output as')
    parser.add_argument('--noise-level', type=float, default=1.0, help='Signal to Noise ratio (higher means more noise)')
    args = parser.parse_args()

    # Run main function
    main(args)