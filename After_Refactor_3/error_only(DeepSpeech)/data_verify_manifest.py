import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
args = parser.parse_args()

def main():
    try:
        for manifest_path in tqdm(args.manifests):
            with open(manifest_path, "r") as manifest_file:
                manifest_json = json.load(manifest_file)

            root_path = Path(manifest_json.get('root_path',''))
            samples = manifest_json.get('samples',[])
            for sample in tqdm(samples):
                wav_path = sample.get('wav_path','')
                transcript_path = sample.get('transcript_path','')
                if not wav_path or not transcript_path:
                    print(f"File paths not given for sample {sample} in manifest {manifest_path}")
                    continue
                wav_file = root_path / Path(wav_path)
                transcript_file = root_path / Path(transcript_path)
                if not wav_file.exists():
                    print(f"File {wav_path} does not exist in manifest {manifest_path}")
                    continue
                if not transcript_file.exists():
                    print(f"File {transcript_path} does not exist in manifest {manifest_path}")
                    continue
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()