import argparse
import os
import tarfile
import logging
import wget

from deepspeech_pytorch.data.utils import create_manifest

LOGGER = logging.getLogger(__name__)

AN4_URL = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4.tar.gz'
AN4_TAR_PATH = 'an4.tar.gz'
TRAIN_PATH = 'train/'
VAL_PATH = 'val/'
TEST_PATH = 'test/'
MANIFEST_PREFIX = 'an4_'

def download_file(url, target_dir):
    """Downloads file from given URL to target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    local_path = os.path.join(target_dir, os.path.basename(url))
    if not os.path.exists(local_path):
        LOGGER.info('Downloading %s to %s...', url, target_dir)
        wget.download(url, out=target_dir)
        LOGGER.info('Done.')
    return local_path

def extract_tar(tar_path, target_dir):
    """Extracts given tar file to target directory."""
    tar = tarfile.open(tar_path)
    LOGGER.info('Extracting %s to %s...', tar_path, target_dir)
    tar.extractall(target_dir)
    LOGGER.info('Done.')

def create_an4_manifest(data_path, manifest_path, output_name, min_duration=None, max_duration=None, num_workers=1):
    """Creates manifest file for AN4 dataset."""
    LOGGER.info('Creating manifest for %s...', output_name)
    create_manifest(data_path=data_path,
                    output_name=output_name,
                    manifest_path=manifest_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_workers=num_workers)
    LOGGER.info('Done.')

def download_an4(target_dir='an4_dataset', manifest_dir='', min_duration=None, max_duration=None, num_workers=1):
    """Downloads AN4 dataset and creates manifests."""
    tar_path = download_file(AN4_URL, target_dir)
    extract_tar(tar_path, target_dir)
    train_path = os.path.join(target_dir, TRAIN_PATH)
    val_path = os.path.join(target_dir, VAL_PATH)
    test_path = os.path.join(target_dir, TEST_PATH)

    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
        create_an4_manifest(train_path, manifest_dir, MANIFEST_PREFIX + 'train_manifest.json', min_duration, max_duration, num_workers)
        create_an4_manifest(val_path, manifest_dir, MANIFEST_PREFIX + 'val_manifest.json', min_duration, max_duration, num_workers)
        create_an4_manifest(test_path, manifest_dir, MANIFEST_PREFIX + 'test_manifest.json', num_workers=num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes and downloads AN4 dataset.')
    parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
    parser.add_argument('--manifest-dir', default='', help='Path to save manifests')
    parser.add_argument('--min-duration', type=float, help='Minimum duration for audio files')
    parser.add_argument('--max-duration', type=float, help='Maximum duration for audio files')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker processes')
    args = parser.parse_args()

    download_an4(
        target_dir=args.target_dir,
        manifest_dir=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )