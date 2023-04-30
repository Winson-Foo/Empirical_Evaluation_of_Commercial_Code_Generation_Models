import argparse
import os
import tarfile

import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


RAW_TAR_URL = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4.tar.gz'
TRAIN_PATH = 'train'
VAL_PATH = 'val'
TEST_PATH = 'test'
MANIFEST_SUFFIX = '_manifest.json'


def download_file(url: str, file_path: str):
    """
    Downloads file from URL to specified file path.
    """
    if not os.path.exists(file_path):
        wget.download(url)
    tar = tarfile.open(file_path)
    return tar


def extract_files(tar: tarfile.TarFile, target_dir: str):
    """
    Extracts files from tar to the target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    tar.extractall(target_dir)


def create_an4_manifests(target_dir: str,
                         manifest_dir: str,
                         min_duration: float,
                         max_duration: float,
                         num_workers: int):
    """
    Creates manifests for AN4 dataset.
    """
    train_dir = os.path.join(target_dir, TRAIN_PATH)
    val_dir = os.path.join(target_dir, VAL_PATH)
    test_dir = os.path.join(target_dir, TEST_PATH)

    print('Creating manifests...')
    create_manifest(data_path=train_dir,
                    output_name=f'an4_{TRAIN_PATH}{MANIFEST_SUFFIX}',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_workers=num_workers)
    create_manifest(data_path=val_dir,
                    output_name=f'an4_{VAL_PATH}{MANIFEST_SUFFIX}',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_workers=num_workers)
    create_manifest(data_path=test_dir,
                    output_name=f'an4_{TEST_PATH}{MANIFEST_SUFFIX}',
                    manifest_path=manifest_dir,
                    num_workers=num_workers)


def download_an4(target_dir: str,
                 manifest_dir: str,
                 min_duration: float,
                 max_duration: float,
                 num_workers: int):
    """
    Downloads and extracts AN4 dataset, and creates manifests.
    """
    raw_tar_path = os.path.join(target_dir, os.path.basename(RAW_TAR_URL))
    tar = download_file(RAW_TAR_URL, raw_tar_path)
    extract_files(tar, target_dir)
    create_an4_manifests(target_dir, manifest_dir, min_duration, max_duration, num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes and downloads AN4 dataset.')
    parser = add_data_opts(parser)
    parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
    args = parser.parse_args()
    assert args.sample_rate == 16000, "AN4 only supports sample rate of 16000 currently."
    download_an4(
        target_dir=args.target_dir,
        manifest_dir=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )