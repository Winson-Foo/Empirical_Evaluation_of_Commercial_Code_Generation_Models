import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List


from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def create_manifest_directories(output_dir: Path, extension: str) -> None:
    """Create the necessary directories for the merged manifest files."""
    root_dir = output_dir / "merged_manifest"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / extension).mkdir(parents=True, exist_ok=True)
    (root_dir / "txt").mkdir(parents=True, exist_ok=True)


def merge_manifest_files(manifest_paths: List[Path]) -> Dict:
    """Merge multiple manifest files into a single dictionary."""
    merged_manifest = {
        "root_path": "",
        "samples": []
    }

    for manifest_path in manifest_paths:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        root_path = Path(manifest["root_path"])

        for sample in manifest["samples"]:
            try:
                old_audio_path = root_path / Path(sample["wav_path"])
                new_audio_path = merged_manifest["root_path"] / Path(sample["wav_path"])
                os.symlink(old_audio_path, new_audio_path)
                old_txt_path = root_path / Path(sample["transcript_path"])
                new_txt_path = merged_manifest["root_path"] / Path(sample["transcript_path"])
                os.symlink(old_txt_path, new_txt_path)
            except FileExistsError:
                LOGGER.warning(f"File already exists: {old_audio_path}")

        merged_manifest["samples"] += manifest["samples"]

    return merged_manifest


def write_manifest_file(manifest: Dict, name: str) -> None:
    """Write the merged manifest file to disk."""
    with open(f"{name}.json", "w") as f:
        json.dump(manifest, f)


def main(args):
    output_dir = Path(args.out)
    create_manifest_directories(output_dir, args.extension)

    merged_manifest = merge_manifest_files(args.manifests)
    merged_manifest["root_path"] = output_dir / "merged_manifest"
    write_manifest_file(merged_manifest, args.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Merges all manifest files in specified folder.")
    parser.add_argument("manifests", metavar="m", nargs="+", help="Path to all manifest files you want to merge.")
    parser.add_argument("-e", "--extension", default="wav", type=str, help="Audio file extension")
    parser.add_argument("--name", default="merged_manifest", type=str, help="Merged dataset name")
    parser.add_argument("--out", default="./", type=str, help="Output directory")
    args = parser.parse_args()

    main(args)