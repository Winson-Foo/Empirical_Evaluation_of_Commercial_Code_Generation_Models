import os
import argparse
import shutil
import urllib.request
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
    return phrase.strip().upper()


def _process_file(wav_dir, txt_dir, base_filename, root_dir, sample_rate):
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    os.system(f"sox {full_recording_path} -r {sample_rate} -b 16 -c 1 {wav_recording_path}")
    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), f"Transcript file {transcript_file} does not exist."
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, f"{key} is not in the transcriptions."
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


def download_and_extract(url, target_dir):
    filename = url.split("/")[-1]
    target_filename = os.path.join(target_dir, filename)
    if not os.path.exists(target_filename):
        urllib.request.urlretrieve(url, target_filename)
    print(f"Unpacking {filename}...")
    with tarfile.open(target_filename, "r:*") as tar:
        tar.extractall(target_dir)
    os.remove(target_filename)


def process_files(split_type, lst_libri_urls, target_dl_dir, files_to_use, sample_rate):
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
    with ThreadPoolExecutor() as executor:
        futures = []
        for url in lst_libri_urls:
            # check if we want to download this file
            dl_flag = False
            for f in files_to_use:
                if f in url:
                    dl_flag = True
                    break
            if not dl_flag:
                print(f"Skipping url: {url}")
                continue
            futures.append(executor.submit(partial(download_and_extract, url, split_dir)))
        for future in tqdm(futures, desc=f"Downloading and extracting {split_type}"):
            future.result()
        print(f"Converting flac files to wav and extracting transcripts for {split_type}...")
        assert os.path.exists(extracted_dir), f"Archive {filename} was not properly uncompressed."
        for root, subdirs, files in tqdm(os.walk(extracted_dir), desc=f"Processing {split_type} files"):
            for f in files:
                if f.endswith(".flac"):
                    _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir,
                                  base_filename=f, root_dir=root, sample_rate=sample_rate)
        print(f"Finished processing {split_type}.")
        shutil.rmtree(extracted_dir)
        if split_type == "train":
            create_manifest(split_dir, f"libri_{split_type}_manifest.json",
                            args.manifest_dir, args.min_duration, args.max_duration, args.num_workers)
        else:
            create_manifest(split_dir, f"libri_{split_type}_manifest.json",
                            args.manifest_dir, num_workers=args.num_workers)


def main():
    parser = argparse.ArgumentParser(description="Processes and downloads LibriSpeech dataset.")
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default="LibriSpeech_dataset/", type=str,
                        help="Directory to store the dataset.")
    parser.add_argument("--files-to-use", default="train-clean-100,train-clean-360,train-other-500,"
                                                      "dev-clean,dev-other,test-clean,test-other", type=str,
                        help="Comma-separated list of file names to download, without .tar.gz extension.")
    args = parser.parse_args()
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)
    files_to_use = [name + ".tar.gz" for name in args.files_to_use.split(",")]
    sample_rate = args.sample_rate
    with ThreadPoolExecutor() as executor:
        futures = []
        for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
            futures.append(executor.submit(partial(process_files, split_type, lst_libri_urls,
                                                    target_dl_dir, files_to_use, sample_rate)))
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()