import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
        args = parser.parse_args()

        for manifest_path in tqdm(args.manifests):
            try:
                with open(manifest_path, "r") as manifest_file:
                    manifest_json = json.load(manifest_file)
            except FileNotFoundError:
                print(f"{manifest_path} does not exist", file=sys.stderr)
                continue

            root_path = Path(manifest_json.get('root_path'))
            
            if not root_path.exists():
                print(f"{root_path} does not exist", file=sys.stderr)
                continue

            for sample in tqdm(manifest_json.get('samples', [])):
                wav_path = Path(sample.get('wav_path'))
                if not (root_path / wav_path).exists():
                    print(f"{root_path / wav_path} does not exist", file=sys.stderr)

                transcript_path = Path(sample.get('transcript_path'))
                if not (root_path / transcript_path).exists():
                    print(f"{root_path / transcript_path} does not exist", file=sys.stderr)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()