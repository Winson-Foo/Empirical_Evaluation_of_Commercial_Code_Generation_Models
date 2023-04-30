import argparse
import os
import shutil
from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest
from utils import download_dataset, VOXFORGE_URL_16kHz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes and downloads VoxForge dataset.')
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default='voxforge_dataset/', type=str, help="Directory to store the dataset.")
    args = parser.parse_args()

    target_dir = args.target_dir
    sample_rate = args.sample_rate

    download_dataset(target_dir, sample_rate)

    print('Creating manifests...')
    create_manifest(
        data_path=target_dir,
        output_name='voxforge_train_manifest.json',
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )