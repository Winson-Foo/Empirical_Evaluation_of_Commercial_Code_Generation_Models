import argparse
import io
import json
import os
from pathlib import Path

from tqdm import tqdm


def create_directories(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / args.extension).mkdir(parents=True, exist_ok=True)
    (path / 'txt').mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser(description="Merges all manifest files in specified folder.")
parser.add_argument("manifests", metavar="m", nargs="+", help="Path to all manifest files you want to merge.")
parser.add_argument("-e", "--extension", default="wav", type=str, help="Audio file extension")
parser.add_argument("--name", default="merged_manifest", type=str, help="Merged dataset name")
parser.add_argument("--out", default="./", type=str, help="Output directory")
args = parser.parse_args()

try:
    main_path = Path(args.out)
    if not main_path.is_dir():
        raise ValueError(f"Output directory '{args.out}' does not exist.")
except ValueError as e:
    print(e)
    exit(1)
except Exception as e:
    print("An error occurred while validating output directory:", e)
    exit(1)

new_manifest_path = main_path / Path(args.name)
create_directories(new_manifest_path)

new_manifest = {
    'root_path': new_manifest_path.absolute().as_posix(),
    'samples': []
}

for manifest in tqdm(args.manifests, desc="Manifests"):
    try:
        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            raise ValueError(f"Manifest file '{manifest}' does not exist.")
    except ValueError as e:
        tqdm.write(str(e))
        continue
    except Exception as e:
        tqdm.write("An error occurred while validating manifest file:", str(e))
        continue

    with open(manifest_path, "r") as manifest_file:
        try:
            manifest_json = json.load(manifest_file)
        except json.JSONDecodeError:
            tqdm.write(f"Error reading {manifest_path}")
            continue

    root_path = Path(manifest_json.get('root_path', ''))
    if not root_path.is_dir():
        tqdm.write(f"Root path '{root_path}' not found in manifest {manifest_path}")
        continue

    samples = manifest_json.get('samples')
    if not isinstance(samples, list):
        tqdm.write(f"No samples found in manifest {manifest_path}")
        continue

    for sample in tqdm(samples, desc="Samples"):
        try:
            old_audio_path = root_path / Path(sample.get('wav_path', ''))
            if not old_audio_path.is_file():
                raise ValueError(f"WAV file '{old_audio_path}' not found in manifest {manifest_path}")
            new_audio_path = new_manifest_path.absolute() / Path(sample.get('wav_path', ''))
            os.symlink(old_audio_path, new_audio_path)
            old_txt_path = root_path / Path(sample['transcript_path'])
            if not old_txt_path.is_file():
                raise ValueError(f"Transcript file '{old_txt_path}' not found in manifest {manifest_path}")
            new_txt_path = new_manifest_path.absolute() / Path(sample.get('transcript_path', ''))
            os.symlink(old_txt_path, new_txt_path)
        except ValueError as e:
            tqdm.write(str(e))
            continue
        except FileExistsError:
            continue
        except Exception as e:
            tqdm.write(f"An error occurred while creating sym links for {sample} in manifest {manifest_path}: {e}")
            continue

    new_manifest['samples'] += samples

with open(new_manifest_path / f"{args.name}_manifest.json", "w") as json_file:
    json.dump(new_manifest, json_file, indent=4)