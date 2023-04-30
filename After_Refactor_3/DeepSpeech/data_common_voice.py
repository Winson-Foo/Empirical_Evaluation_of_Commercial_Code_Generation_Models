import argparse
import csv
import os
import tarfile
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

from sox import Transformer
import tqdm
import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


parser = argparse.ArgumentParser(description='Downloads and processes Mozilla Common Voice dataset.')
parser = add_data_opts(parser)
parser.add_argument("--target-dir", default='CommonVoice_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument("--tar-path", type=str, help="Path to the Common Voice *.tar file if downloaded (Optional).")
parser.add_argument("--language-dir", default='en', type=str, help="Language dir to process.")
parser.add_argument('--files-to-process', nargs='+', default=['test.tsv', 'dev.tsv', 'train.tsv'],
                    type=str, help='list of *.csv file names to process')
args = parser.parse_args()
VERSION = 'cv-corpus-5.1-2020-06-22'
COMMON_VOICE_URL = ("https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" 
                    "{}/en.tar.gz".format(VERSION))


def create_directories(target_dir: str, *dirs: str):
    for directory in dirs:
        path = os.path.join(target_dir, directory)
        os.makedirs(path, exist_ok=True)


def process_row(row: dict, audio_clips_path: str, wav_dir: str, txt_dir: str) -> None:
    file_path, text = row['path'], row['sentence']
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    text = text.strip().upper()
    with open(os.path.join(txt_dir, file_name + '.txt'), 'w') as f:
        f.write(text)
    audio_path = os.path.join(audio_clips_path, file_path)
    output_wav_path = os.path.join(wav_dir, file_name + '.wav')

    tfm = Transformer()
    tfm.rate(samplerate=args.sample_rate)
    tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)


def convert_csv_to_wav(csv_file: str, target_dir: str, num_workers: int) -> None:
    """Read CSV file description, convert mp3 to wav, process text. Save results to target_dir."""
    wav_dir, txt_dir = os.path.join(target_dir, 'wav/'), os.path.join(target_dir, 'txt/')
    create_directories(target_dir, wav_dir, txt_dir)
    audio_clips_path = os.path.join(os.path.dirname(csv_file), 'clips')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        next(reader, None)  # skip the headers
        data = [(row['path'], row['sentence']) for row in reader]
        with ThreadPool(num_workers) as pool:
            list(tqdm.tqdm(pool.imap(
                func=lambda r: process_row(r, audio_clips_path, wav_dir, txt_dir), 
                iterable=data
            ), total=len(data)))


def download_and_unpack_corpus(target_dir: str) -> None:
    target_unpacked_dir = os.path.join(target_dir, "CV_unpacked")
    
    if os.path.exists(target_unpacked_dir):
        print(f"Found existing folder {target_unpacked_dir}")
    else:
        print("Could not find Common Voice. Downloading corpus...")
        filename = wget.download(COMMON_VOICE_URL, target_dir)
        target_file = os.path.join(target_dir, os.path.basename(filename))
        create_directories(target_dir, target_unpacked_dir)
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(target_unpacked_dir)
    return target_unpacked_dir


def process_files(target_dir: str, language_dir: str, files: List[str], num_workers: int, manifest_dir: str,
                  min_duration: float, max_duration: float) -> None:
    folder_path = os.path.join(target_dir, "CV_unpacked", VERSION, language_dir)
    
    for csv_file in files:
        csv_file_path = os.path.join(folder_path, csv_file)
        convert_csv_to_wav(csv_file_path, os.path.join(target_dir, os.path.splitext(csv_file)[0]), num_workers)
    
    print("Creating manifests...")
    for csv_file in files:
        create_manifest(
            data_path=os.path.join(target_dir, os.path.splitext(csv_file)[0]),
            output_name=f"commonvoice_{os.path.splitext(csv_file)[0]}_manifest.json",
            manifest_path=manifest_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            num_workers=num_workers
        )


def main(target_dir: str, language_dir: str, files: List[str], num_workers: int, manifest_dir: str,
         min_duration: float, max_duration: float) -> None:
    create_directories(target_dir)
    target_unpacked_dir = download_and_unpack_corpus(target_dir)
    process_files(target_dir, language_dir, files, num_workers, manifest_dir, min_duration, max_duration)


if __name__ == "__main__":
    main(args.target_dir, args.language_dir, args.files_to_process, args.num_workers, args.manifest_dir,
         args.min_duration, args.max_duration)