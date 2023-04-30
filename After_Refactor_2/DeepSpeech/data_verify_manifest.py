import argparse
import json
from pathlib import Path
from tqdm import tqdm

def parse_manifest(manifest_path):
    with open(manifest_path, "r") as manifest_file:
        manifest_json = json.load(manifest_file)
    return manifest_json

def verify_sample(root_path, sample):
    wav_path = Path(sample['wav_path'])
    transcript_path = Path(sample['transcript_path'])
    assert (root_path / wav_path).exists(), f"Error: {wav_path} does not exist"
    assert (root_path / transcript_path).exists(), f"Error: {transcript_path} does not exist"

def verify_manifest(manifest_path):
    manifest_json = parse_manifest(manifest_path)
    root_path = Path(manifest_json['root_path'])
    for sample in tqdm(manifest_json['samples'], desc=f"Verifying {manifest_path}"):
        verify_sample(root_path, sample)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
    args = parser.parse_args()

    for manifest_path in tqdm(args.manifests, desc="Parsing manifests"):
        verify_manifest(manifest_path)

if __name__ == "__main__":
    main()