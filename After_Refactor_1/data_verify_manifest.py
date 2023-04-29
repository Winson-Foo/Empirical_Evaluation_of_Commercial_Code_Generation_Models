import argparse
import json
from pathlib import Path

from tqdm import tqdm

def load_manifest(manifest_path):
    """Load a manifest file and return its contents as a dictionary."""
    with open(manifest_path, "r") as manifest_file:
        return json.load(manifest_file)

def check_sample_paths(manifest_json):
    """Check that all sample paths in the manifest exist on disk."""
    root_path = Path(manifest_json['root_path'])
    for sample in tqdm(manifest_json['samples']):
        wav_path = root_path / Path(sample['wav_path'])
        transcript_path = root_path / Path(sample['transcript_path'])
        assert wav_path.exists(), f"{wav_path} does not exist"
        assert transcript_path.exists(), f"{transcript_path} does not exist"

def main(manifests):
    """Load each manifest and check all sample paths."""
    for manifest_path in tqdm(manifests):
        manifest_json = load_manifest(manifest_path)
        check_sample_paths(manifest_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
    args = parser.parse_args()
    main(args.manifests)