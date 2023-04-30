from argparse import ArgumentParser
from csv import DictReader
from multiprocessing.pool import ThreadPool
from os import makedirs, path
from tarfile import open as open_tarfile
from tqdm import tqdm
from wget import download
from sox import Transformer
from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest

TARGET_DIR = 'CommonVoice_dataset/'
LANGUAGE_DIR = 'en'
FILES_TO_PROCESS = ['test.tsv', 'dev.tsv', 'train.tsv']
VERSION = 'cv-corpus-5.1-2020-06-22'
COMMON_VOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/en.tar.gz".format(VERSION)

def convert_to_wav(csv_file, target_dir, num_workers, sample_rate):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to target_dir.

    Args:
        csv_file: str, path to *.csv file with data description
        target_dir: str, path to dir to save results
        num_workers: int, number of threads to use
        sample_rate: int, the sample rate of the wav file
    """
    wav_dir = path.join(target_dir, 'wav/')
    txt_dir = path.join(target_dir, 'txt/')
    makedirs(wav_dir, exist_ok=True)
    makedirs(txt_dir, exist_ok=True)
    audio_clips_path = path.dirname(csv_file) + '/clips/'

    def process(x):
        file_path, text = x
        file_name = path.splitext(path.basename(file_path))[0]
        text = text.strip().upper()
        with open(path.join(txt_dir, file_name + '.txt'), 'w') as f:
            f.write(text)
        audio_path = path.join(audio_clips_path, file_path)
        output_wav_path = path.join(wav_dir, file_name + '.wav')

        tfm = Transformer()
        tfm.rate(samplerate=sample_rate)
        tfm.build(
            input_filepath=audio_path,
            output_filepath=output_wav_path
        )

    print('Converting mp3 to wav for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = DictReader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        data = [(row['path'], row['sentence']) for row in reader]
        with ThreadPool(num_workers) as pool:
            list(tqdm(pool.imap(process, data), total=len(data)))

def download_and_extract_common_voice(target_dir, language_dir):
    """ Download and extract Common Voice dataset if not already exist

        Args:
            target_dir: str, path to directory to store the dataset
            language_dir: str, language directory to process
    """
    target_unpacked_dir = path.join(target_dir, "CV_unpacked")

    if path.exists(target_unpacked_dir):
        print('Find existing folder {}'.format(target_unpacked_dir))
    else:
        print("Could not find Common Voice, Downloading corpus...")

        filename = download(COMMON_VOICE_URL, target_dir)
        target_file = path.join(target_dir, path.basename(filename))

        makedirs(target_unpacked_dir, exist_ok=True)
        print("Unpacking corpus to {} ...".format(target_unpacked_dir))
        tar = open_tarfile(target_file)
        tar.extractall(target_unpacked_dir)
        tar.close()

    folder_path = path.join(target_unpacked_dir, VERSION + '/{}/'.format(language_dir))
    return folder_path

def process_common_voice_dataset(args):
    """ Process Common Voice dataset by converting mp3 to wav and generating manifests

        Args:
            args: args, command-line arguments containing information to process Common Voice dataset
    """
    target_dir = args.target_dir
    language_dir = args.language_dir
    manifest_dir = args.manifest_dir
    min_duration = args.min_duration
    max_duration = args.max_duration
    num_workers = args.num_workers
    sample_rate = args.sample_rate

    makedirs(target_dir, exist_ok=True)
    folder_path = download_and_extract_common_voice(target_dir, language_dir)

    for csv_file in FILES_TO_PROCESS:
        convert_to_wav(
            csv_file=path.join(folder_path, csv_file),
            target_dir=path.join(target_dir, path.splitext(csv_file)[0]),
            num_workers=num_workers,
            sample_rate=sample_rate
        )

    print('Creating manifests...')
    for csv_file in FILES_TO_PROCESS:
        create_manifest(
            data_path=path.join(target_dir, path.splitext(csv_file)[0]),
            output_name='commonvoice_' + path.splitext(csv_file)[0] + '_manifest.json',
            manifest_path=manifest_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            num_workers=num_workers
        )

if __name__ == "__main__":
    parser = ArgumentParser(description='Downloads and processes Mozilla Common Voice dataset.')
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default=TARGET_DIR, type=str, help="Directory to store the dataset.")
    parser.add_argument("--tar-path", type=str, help="Path to the Common Voice *.tar file if downloaded (Optional).")
    parser.add_argument("--language-dir", default=LANGUAGE_DIR, type=str, help="Language dir to process.")
    parser.add_argument('--files-to-process', nargs='+', default=FILES_TO_PROCESS, type=str, help='list of *.csv file names to process')
    args = parser.parse_args()

    process_common_voice_dataset(args)