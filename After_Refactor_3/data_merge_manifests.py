import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

DEFAULT_EXTENSION = "wav"
DEFAULT_NAME = "merged_manifest"
DEFAULT_OUTPUT_DIR = "./"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merges all manifest files in specified folder.")
    parser.add_argument("manifests", metavar="m", nargs="+", help="Path to all manifest files you want to merge.")
    parser.add_argument("-e", "--extension", default=DEFAULT_EXTENSION, type=str, help="Audio file extension")
    parser.add_argument("--name", default=DEFAULT_NAME, type=str, help="Merged dataset name")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, type=str, help="Output directory")
    return parser.parse_args()


def create_directory(directory: Path) -> None:
    """Create a directory if it does not already exist."""
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory {directory.absolute().as_posix()}")


def create_manifest_directory(base_directory: Path, extension: str) -> Path:
    """Create a directory for the merged manifest files."""
    manifest_directory = base_directory / extension
    create_directory(manifest_directory)
    return manifest_directory


def create_transcript_directory(base_directory: Path) -> Path:
    """Create a directory for the merged transcript files."""
    transcript_directory = base_directory / "txt"
    create_directory(transcript_directory)
    return transcript_directory


def merge_manifests(manifests: List[str]) -> Dict[str, str]:
    """Merge all manifest files into a single dictionary."""
    merged_manifest = {"root_path": "", "samples": []}
    for manifest_path in tqdm(manifests, desc="Manifests"):
        with open(manifest_path, "r") as manifest_file:
            manifest = json.load(manifest_file)
        merged_manifest["root_path"] = manifest["root_path"]
        for sample in tqdm(manifest["samples"], desc="Samples"):
            merged_manifest["samples"].append(sample)
    return merged_manifest


def create_file_symlink(source_file: Path, target_directory: Path) -> None:
    """Create a symlink for a file in the target directory."""
    target_file = target_directory / source_file.name
    try:
        os.symlink(source_file, target_file)
        logging.info(f"Created symlink {target_file.absolute().as_posix()}")
    except FileExistsError:
        pass


def create_symlinks_for_sample(sample: Dict[str, str], root_path: Path, new_manifest_path: Path) -> None:
    """Create symlinks for audio and transcript files in the merged manifest directory."""
    audio_path = root_path / Path(sample["wav_path"])
    create_file_symlink(audio_path, new_manifest_path / audio_path.name)
    transcript_path = root_path / Path(sample["transcript_path"])
    create_file_symlink(transcript_path, new_manifest_path / transcript_path.name)


def create_audio_symlinks_for_manifest(manifest: Dict[str, str], root_path: Path, new_manifest_path: Path) -> None:
    """Create audio symlinks for all samples in a manifest."""
    for sample in tqdm(manifest["samples"], desc="Samples"):
        create_symlinks_for_sample(sample, root_path, new_manifest_path)


def create_merged_manifest(manifests: List[str], output_directory: Path, extension: str, name: str) -> None:
    """Merge all manifests and create symlinks for audio and transcript files in the merged manifest directory."""
    new_manifest_path = output_directory / name
    create_directory(new_manifest_path)
    manifest_directory = create_manifest_directory(new_manifest_path, extension)
    transcript_directory = create_transcript_directory(new_manifest_path)
    merged_manifest = merge_manifests(manifests)
    create_audio_symlinks_for_manifest(merged_manifest, Path(merged_manifest["root_path"]), manifest_directory)
    with open(f"{name}_manifest.json", "w") as json_file:
        json.dump(merged_manifest, json_file)
    logging.info(f"Created merged manifest {json_file.name}")
    logging.info(f"Manifest directory: {manifest_directory.absolute().as_posix()}")
    logging.info(f"Transcript directory: {transcript_directory.absolute().as_posix()}")


def main() -> None:
    args = parse_args()
    output_directory = Path(args.out)
    create_directory(output_directory)
    create_merged_manifest(args.manifests, output_directory, args.extension, args.name)


if __name__ == "__main__":
    main()