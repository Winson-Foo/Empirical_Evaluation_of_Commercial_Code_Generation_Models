import argparse
import json
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm


def file_exists(path: Path) -> bool:
    """Check if file exists."""
    return path.exists() and path.is_file()


def validate_manifest_files(manifest_json: Dict) -> None:
    """Validate the existence of files in manifest."""
    root_path = Path(manifest_json['root_path'])
    for sample in tqdm(manifest_json['samples']):
        assert file_exists(root_path / Path(sample['wav_path'])), f"{sample['wav_path']} does not exist"
        assert file_exists(root_path / Path(sample['transcript_path'])), f"{sample['transcript_path']} does not exist"


def process_manifest(manifest_path: Path) -> None:
    """Process a single manifest file."""
    with open(manifest_path, "r") as manifest_file:
        manifest_json = json.load(manifest_file)

    validate_manifest_files(manifest_json)


def process_manifests(manifest_paths: List[Path]) -> None:
    """Process multiple manifest files."""
    for manifest_path in tqdm(manifest_paths):
        process_manifest(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
    args = parser.parse_args()

    process_manifests(args.manifests)


if __name__ == "__main__":
    main()