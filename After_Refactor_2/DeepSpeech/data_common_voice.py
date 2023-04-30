import argparse
import csv
import os
import tarfile
from multiprocessing.pool import ThreadPool
import tqdm
import wget
from sox import Transformer

# Constants
VERSION = 'cv-corpus-5.1-2020-06-22'
COMMON_VOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/en.tar.gz".format(VERSION)
FILES_TO_PROCESS = ['test.tsv', 'dev.tsv', 'train.tsv']

def download_common_voice_dataset(target_dir):
    """
    Downloads the Common Voice dataset and extracts it to a target directory.
    """
    target_unpacked_dir = os.path.join(target_dir, "CV_unpacked")

    if os.path.exists(target_unpacked_dir):
        print('Find existing folder {}'.format(target_unpacked_dir))
    else:
        print("Could not find Common Voice, Downloading corpus...")

        filename = wget.download(COMMON_VOICE_URL, target_dir)
        target_file = os.path.join(target_dir, os.path.basename(filename))

        os.makedirs(target_unpacked_dir, exist_ok=True)
        print("Unpacking corpus to {} ...".format(target_unpacked_dir))
        tar = tarfile.open(target_file)
        tar.extractall(target_unpacked_dir)
        tar.close()

    return os.path.join(target_unpacked_dir, VERSION + '/en/')


def convert_to_wav(csv_file, target_dir, num_workers, sample_rate):
    """
    Reads a CSV file with data descriptions, converts mp3 files to wav, processes text,
    and saves results to a target directory.
    """
    wav_dir = os.path.join(target_dir, 'wav/')
    txt_dir = os.path.join(target_dir, 'txt/')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    audio_clips_path = os.path.join(os.path.dirname(csv_file), 'clips')

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.strip().upper()
        with open(os.path.join(txt_dir, file_name + '.txt'), 'w') as f:
            f.write(text)
        audio_path = os.path.join(audio_clips_path, file_path)
        output_wav_path = os.path.join(wav_dir, file_name + '.wav')

        tfm = Transformer()
        tfm.rate(samplerate=sample_rate)
        tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

    print('Converting mp3 to wav for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        data = [(row['path'], row['sentence']) for row in reader]
        with ThreadPool(num_workers) as pool:
            list(tqdm.tqdm(pool.imap(process, data), total=len(data)))


def create_manifests(target_dir, manifest_dir, min_duration, max_duration, num_workers):
    """
    Creates manifests for multiple CSV files with data descriptions.
    """
    for csv_file in FILES_TO_PROCESS:
        create_manifest(
            data_path=os.path.join(target_dir, os.path.splitext(csv_file)[0]),
            output_name='commonvoice_' + os.path.splitext(csv_file)[0] + '_manifest.json',
            manifest_path=manifest_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            num_workers=num_workers
        )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Downloads and processes Mozilla Common Voice dataset.')
    parser.add_argument("--target-dir", default='CommonVoice_dataset/', type=str, help="Directory to store the dataset.")
    parser.add_argument("--language-dir", default='en', type=str, help="Language directory to process.")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of worker threads to use.")
    parser.add_argument("--sample-rate", default=16000, type=int, help="Sample rate of audio files.")
    parser.add_argument("--min-duration", default=1, type=int, help="Minimum duration of audio files in seconds.")
    parser.add_argument("--max-duration", default=15, type=int, help="Maximum duration of audio files in seconds.")
    parser.add_argument("--manifest-dir", default='manifests/', type=str, help="Directory to store manifests.")
    args = parser.parse_args()
    
    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Download and extract Common Voice dataset
    folder_path = download_common_voice_dataset(args.target_dir)

    # Convert mp3 files to wav and process text
    for csv_file in FILES_TO_PROCESS:
        convert_to_wav(
            csv_file=os.path.join(folder_path, csv_file),
            target_dir=os.path.join(args.target_dir, os.path.splitext(csv_file)[0]),
            num_workers=args.num_workers,
            sample_rate=args.sample_rate
        )

    # Create manifests
    create_manifests(
        target_dir=args.target_dir,
        manifest_dir=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()