import argparse
import io
import json
import os
from pathlib import Path

from tqdm import tqdm


# Constants for magic strings and numbers
DEFAULT_EXTENSION = "wav"
DEFAULT_NAME = "merged_manifest"
DEFAULT_OUT_DIR = "./"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Merges all manifest files in specified folder.")
    parser.add_argument("manifests", metavar="m", nargs="+", help="Path to all manifest files you want to merge.")
    parser.add_argument("-e", "--extension", default=DEFAULT_EXTENSION, type=str, help="Audio file extension")
    parser.add_argument("--name", default=DEFAULT_NAME, type=str, help="Merged dataset name")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, type=str, help="Output directory")
    return parser.parse_args()


def create_manifest_directory(out_path, extension):
    """Create the directory structure for the merged manifest"""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / extension).mkdir(parents=True, exist_ok=True)
    (out_path / 'txt').mkdir(parents=True, exist_ok=True)
    return out_path.absolute().as_posix()


def merge_manifests(manifest_paths):
    """Merge multiple manifest files into a single dict"""
    merged_manifest = {'samples': []}
    for manifest_path in tqdm(manifest_paths, desc="Manifests"):
        with open(manifest_path, "r") as manifest_file:
            manifest = json.load(manifest_file)

        root_path = Path(manifest['root_path'])
        for sample in tqdm(manifest['samples'], desc="Samples"):
            try:
                old_audio_path = root_path / Path(sample['wav_path'])
                new_audio_path = out_path / Path(sample['wav_path'])
                os.symlink(old_audio_path, new_audio_path)
                old_txt_path = root_path / Path(sample['transcript_path'])
                new_txt_path = out_path / Path(sample['transcript_path'])
                os.symlink(old_txt_path, new_txt_path)
            except FileExistsError:
                continue

        merged_manifest['samples'] += manifest['samples']

    return merged_manifest


def save_manifest(manifest, name):
    """Save the merged manifest to a JSON file"""
    with open(f"{name}_manifest.json", "w") as json_file:
        json.dump(manifest, json_file)


def main():
    args = parse_args()
    out_path = create_manifest_directory(args.out, args.extension)
    merged_manifest = merge_manifests(args.manifests)
    merged_manifest['root_path'] = out_path
    save_manifest(merged_manifest, args.name)


if __name__ == "__main__":
    main()