import argparse
import json
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
args = parser.parse_args()

def main():
    try:
        for manifest_path in tqdm(args.manifests):
            with open(manifest_path, "r") as manifest_file:
                manifest_json = json.load(manifest_file)

            root_path = Path(manifest_json['root_path'])
            for sample in tqdm(manifest_json['samples']):
                wav_path = root_path / Path(sample.get('wav_path', ''))
                transcript_path = root_path / Path(sample.get('transcript_path', ''))
                if not wav_path.exists():
                    raise FileNotFoundError(f"{sample.get('wav_path', '')} does not exist")
                if not transcript_path.exists():
                    raise FileNotFoundError(f"{sample.get('transcript_path', '')} does not exist")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()