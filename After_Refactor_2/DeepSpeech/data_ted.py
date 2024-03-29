import os
import io
import subprocess
import tarfile
import argparse
import unicodedata
import wget

from tqdm import tqdm
from deepspeech_pytorch.data.utils import create_manifest
from deepspeech_pytorch.data.data_opts import add_data_opts


TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"


def preprocess_transcript(phrase):
    """
    Preprocesses the transcript by stripping whitespace and converting to uppercase
    :param phrase: A transcript string
    :return: A string
    """
    return phrase.strip().upper()


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    """
    Filters out utterances that are shorter than the minimum length specified
    :param utterance_info: A dictionary containing the timings and transcript of an utterance
    :param min_len_sec: Minimum length (in seconds) of an utterance to be included
    :return: Boolean indicating whether or not the utterance is long enough
    """
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec


def get_utterances_from_stm(stm_file):
    """
    Parses the stm file and returns a list of entries containing phrase and its start/end timings
    :param stm_file: Path to the stm file
    :return: A list of dictionaries
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res


def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
    """
    Cuts the utterance from the source sph file to create a target wav file
    :param src_sph_file: Path to the source sph file
    :param target_wav_file: Path to the target wav file to be created
    :param start_time: Start time of the utterance (in seconds)
    :param end_time: End time of the utterance (in seconds)
    :param sample_rate: Sample rate of the target wav file
    """
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def prepare_dir(ted_dir, args):
    """
    Prepares the directory containing the TEDLIUM files by converting sph files to wav and creating transcript files
    :param ted_dir: Path to the directory containing TEDLIUM files
    :param args: Command-line arguments specified
    """
    converted_dir = os.path.join(ted_dir, "converted")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(converted_dir, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(converted_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
        stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
        all_utterances = get_utterances_from_stm(stm_file_full)

        all_utterances = filter(filter_short_utterances, all_utterances)
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
            target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
            cut_utterance(sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                          sample_rate=args.sample_rate)
            with io.FileIO(target_txt_file, "w") as f:
                f.write(preprocess_transcript(utterance["transcript"]).encode('utf-8'))
        counter += 1


def download_dataset(args):
    """
    Downloads the TEDLIUM dataset tar file and unpacks it
    :param args: Command-line arguments specified
    :return: Path to the directory containing the TEDLIUM files
    """
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    target_unpacked_dir = os.path.join(target_dl_dir, "TEDLIUM_release2")
    if args.tar_path and os.path.exists(args.tar_path):
        target_file = args.tar_path
    else:
        print("Could not find downloaded TEDLIUM archive, Downloading corpus...")
        wget.download(TED_LIUM_V2_DL_URL, target_dl_dir)
        target_file = os.path.join(target_dl_dir, "TEDLIUM_release2.tar.gz")

    if not os.path.exists(target_unpacked_dir):
        print("Unpacking corpus...")
        tar = tarfile.open(target_file)
        tar.extractall(target_dl_dir)
        tar.close()
    else:
        print("Found TEDLIUM directory, skipping unpacking of tar files")

    return target_unpacked_dir


def create_manifests(data_path, output_name, manifest_path, min_duration=None, max_duration=None, num_workers=None):
    """
    Creates a manifest file for a given dataset
    :param data_path: Path to the directory containing the dataset
    :param output_name: Name for the manifest file to be created
    :param manifest_path: Path to the directory where the manifest file should be created
    :param min_duration: Minimum duration (in seconds) of an utterance to be included
    :param max_duration: Maximum duration (in seconds) of an utterance to be included
    :param num_workers: Number of workers to use for creating the manifest
    :return:
    """
    create_manifest(
        data_path=data_path,
        output_name=output_name,
        manifest_path=manifest_path,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )


def main():
    parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
    parser.add_argument("--tar-path", type=str, help="Path to the TEDLIUM_release tar if downloaded (Optional).")
    args = parser.parse_args()

    target_unpacked_dir = download_dataset(args)

    train_ted_dir = os.path.join(target_unpacked_dir, "train")
    val_ted_dir = os.path.join(target_unpacked_dir, "dev")
    test_ted_dir = os.path.join(target_unpacked_dir, "test")

    prepare_dir(train_ted_dir, args)
    prepare_dir(val_ted_dir, args)
    prepare_dir(test_ted_dir, args)
    print('Creating manifests...')

    create_manifests(
        data_path=train_ted_dir,
        output_name='ted_train_manifest.json',
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )
    create_manifests(
        data_path=val_ted_dir,
        output_name='ted_val_manifest.json',
        manifest_path=args.manifest_dir,
        num_workers=args.num_workers

    )
    create_manifests(
        data_path=test_ted_dir,
        output_name='ted_test_manifest.json',
        manifest_path=args.manifest_dir,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()