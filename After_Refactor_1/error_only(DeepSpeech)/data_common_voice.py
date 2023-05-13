import argparse
import csv
import os
import tarfile
import urllib.request
from multiprocessing.pool import ThreadPool

from sox import Transformer
import tqdm

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
COMMON_VOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/en.tar.gz".format(VERSION)

def download_corpus(url, directory):
    try:
        urllib.request.urlretrieve(url, os.path.join(directory, os.path.basename(url)))
    except urllib.error.URLError:
        print("Error - Could not retrieve data from {}. Please check your internet connection or URL".format(url))
        exit()

def convert_to_wav(csv_file, target_dir, num_workers):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to target_dir.

    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        target_dir: str, path to dir to save results; wav/ and txt/ dirs will be created
    """
    wav_dir = os.path.join(target_dir, 'wav/')
    txt_dir = os.path.join(target_dir, 'txt/')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    audio_clips_path = os.path.dirname(csv_file) + '/clips/'

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.strip().upper()
        try:
            with open(os.path.join(txt_dir, file_name + '.txt'), 'w') as f:
                f.write(text)
            audio_path = os.path.join(audio_clips_path, file_path)
            output_wav_path = os.path.join(wav_dir, file_name + '.wav')

            tfm = Transformer()
            tfm.rate(samplerate=args.sample_rate)
            tfm.build(
                input_filepath=audio_path,
                output_filepath=output_wav_path
            )
        except Exception as e:
                print(f"Error while processing {audio_path}: {str(e)}")
                pass

    print('Converting mp3 to wav for {}.'.format(csv_file))
    try:
        with open(csv_file) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            data = [(row['path'], row['sentence']) for row in reader]
            with ThreadPool(num_workers) as pool:
                list(tqdm.tqdm(pool.imap(process, data), total=len(data)))
    except FileNotFoundError:
        print(f"Error - File {csv_file} not found. Please check if the path is correct.")
        exit()

def main():
    target_dir = args.target_dir
    language_dir = args.language_dir
    
    os.makedirs(target_dir, exist_ok=True)

    target_unpacked_dir = os.path.join(target_dir, "CV_unpacked")

    if os.path.exists(target_unpacked_dir):
        print('Find existing folder {}'.format(target_unpacked_dir))
    else:
        print("Could not find Common Voice, Downloading corpus...")

        download_corpus(COMMON_VOICE_URL, target_dir)

        target_file = os.path.join(target_dir, os.path.basename(COMMON_VOICE_URL))

        os.makedirs(target_unpacked_dir, exist_ok=True)
        print("Unpacking corpus to {} ...".format(target_unpacked_dir))
        try:
            tar = tarfile.open(target_file)
            tar.extractall(target_unpacked_dir)
            tar.close()
        except tarfile.ReadError:
                print(f"Error - Could not extract {target_file}. Make sure it is a valid *.tar.gz file")
                exit()

    folder_path = os.path.join(target_unpacked_dir, VERSION + '/{}/'.format(language_dir))

    for csv_file in args.files_to_process:
        try:
            convert_to_wav(
                csv_file=os.path.join(folder_path, csv_file),
                target_dir=os.path.join(target_dir, os.path.splitext(csv_file)[0]),
                num_workers=args.num_workers
            )
        except FileNotFoundError:
            print(f"Error - File {csv_file} not found. Please check if the path is correct.")
            exit()

    print('Creating manifests...')
    for csv_file in args.files_to_process:
        create_manifest(
            data_path=os.path.join(target_dir, os.path.splitext(csv_file)[0]),
            output_name='commonvoice_' + os.path.splitext(csv_file)[0] + '_manifest.json',
            manifest_path=args.manifest_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()