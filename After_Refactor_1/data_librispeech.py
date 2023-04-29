import os
import shutil
import subprocess
import tarfile
import wget
from argparse import ArgumentParser
from tqdm import tqdm
from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest

LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
              "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
              "http://www.openslr.org/resources/12/train-other-500.tar.gz"],
    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"],
    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


def _preprocess_transcript(phrase):
    """Preprocess transcript by stripping and converting to uppercase."""
    return phrase.strip().upper()


def _convert_flac_to_wav(wav_dir, txt_dir, base_filename, root_dir, sample_rate):
    """Convert flac files to WAV and extract transcripts."""
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)

    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call([f"sox {full_recording_path} -r {sample_rate} -b 16 -c 1 {wav_recording_path}"], shell=True)

    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), f"Transcript file {transcript_file} does not exist."
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, f"{key} is not in the transcriptions"
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


def download_and_convert_files(args):
    """Download LibriSpeech dataset files, unpack them, and convert to WAV."""
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(target_dl_dir, split_type)

        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        split_wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(split_wav_dir):
            os.makedirs(split_wav_dir)

        split_txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(split_txt_dir):
            os.makedirs(split_txt_dir)

        extracted_dir = os.path.join(split_dir, "LibriSpeech")
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)

        for url in lst_libri_urls:
            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)

            if args.files_to_use and filename not in args.files_to_use:
                print(f"Skipping url: {url}")
                continue

            if not os.path.exists(target_filename):
                wget.download(url, split_dir)

            print(f"Unpacking {filename}...")
            tar = tarfile.open(target_filename)
            tar.extractall(split_dir)
            tar.close()
            os.remove(target_filename)

            print(f"Converting flac files to wav and extracting transcripts...")
            assert os.path.exists(extracted_dir), f"Archive {filename} was not properly uncompressed."
            for root, subdirs, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.find(".flac") != -1:
                        _convert_flac_to_wav(wav_dir=split_wav_dir, txt_dir=split_txt_dir, base_filename=f, root_dir=root,
                                             sample_rate=args.sample_rate)

            print(f"Finished downloading, unpacking and converting {url}")
            shutil.rmtree(extracted_dir)

        manifest_args = {
            "data_path": split_dir,
            "manifest_path": args.manifest_dir,
            "num_workers": args.num_workers,
            "output_name": f"libri_{split_type}_manifest.json"
        }

        if split_type == "train":
            manifest_args["min_duration"] = args.min_duration
            manifest_args["max_duration"] = args.max_duration

        create_manifest(**manifest_args)


def main(args):
    download_and_convert_files(args)


if __name__ == "__main__":
    parser = ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default="LibriSpeech_dataset/", type=str, help="Directory to store the dataset.")
    parser.add_argument('--files-to-use', default=None, type=str, help='List of file names to download.')
    args = parser.parse_args()

    main(args)