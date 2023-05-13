import argparse
import os
import tarfile
import sys
import traceback

import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


def download_an4(target_dir: str,
                 manifest_dir: str,
                 min_duration: float,
                 max_duration: float,
                 num_workers: int):
    try:
        raw_tar_path = os.path.join(target_dir,'an4.tar.gz')
        if not os.path.exists(raw_tar_path):
            wget.download('https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4.tar.gz', out=target_dir)
        tar = tarfile.open(raw_tar_path)
        os.makedirs(target_dir, exist_ok=True)
        tar.extractall(target_dir)
        train_path = os.path.join(target_dir, 'an4', 'wav', 'train')
        val_path = os.path.join(target_dir, 'an4', 'wav', 'dev')
        test_path = os.path.join(target_dir, 'an4', 'wav', 'test')
    except Exception:
        print(f'Error while downloading and extracting data: {traceback.format_exc()}')
        sys.exit(1)

    try:
        print('Creating manifests...')
        create_manifest(data_path=train_path,
                        output_name=os.path.join(manifest_dir, 'an4_train_manifest.json'),
                        manifest_path=manifest_dir,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        num_workers=num_workers)
        create_manifest(data_path=val_path,
                        output_name=os.path.join(manifest_dir, 'an4_val_manifest.json'),
                        manifest_path=manifest_dir,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        num_workers=num_workers)
        create_manifest(data_path=test_path,
                        output_name=os.path.join(manifest_dir, 'an4_test_manifest.json'),
                        manifest_path=manifest_dir,
                        num_workers=num_workers)
    except Exception:
        print(f'Error while creating manifests: {traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Processes and downloads an4.')
        parser = add_data_opts(parser)
        parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
        args = parser.parse_args()
        if args.sample_rate != 16000:
            print("AN4 only supports sample rate of 16000 currently.")
            sys.exit(1)
        download_an4(
            target_dir=args.target_dir,
            manifest_dir=args.manifest_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            num_workers=args.num_workers
        )
    except Exception:
        print(f'Error: {traceback.format_exc()}')
        sys.exit(1)