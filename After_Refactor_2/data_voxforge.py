import argparse
import re
from tqdm import tqdm
from deepspeech_pytorch.data.utils import create_manifest
from deepspeech_pytorch.data_opts import add_data_opts
import config
from download_manager import DownloadManager
import logging


def get_recordings_dir(sample_dir, recording_name):
    wav_dir = os.path.join(sample_dir, recording_name, "wav")
    if os.path.exists(wav_dir):
        return "wav", wav_dir
    flac_dir = os.path.join(sample_dir, recording_name, "flac")
    if os.path.exists(flac_dir):
        return "flac", flac_dir
    raise Exception("wav or flac directory was not found for recording name: {}".format(recording_name))


def prepare_samples(manager, file_list):
    for f in tqdm(file_list, total=len(file_list)):
        recording_name = f.replace(".tgz", "")
        manager.download_and_extract(f, recording_name)


if __name__ == '__main__':
    args = add_data_opts(argparse.ArgumentParser(description='Processes and downloads VoxForge dataset.')).parse_args()
    
    target_dir = config.TARGET_DIR
    sample_rate = args.sample_rate

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

    logging.info("Starting downloading of VoxForge samples...")
    manager = DownloadManager(config.VOXFORGE_URL_16kHz, target_dir)
    
    request = urllib.request.Request(config.VOXFORGE_URL_16kHz)
    response = urllib.request.urlopen(request)
    content = response.read()
    all_files = re.findall("href\=\"(.*\.tgz)\"", content.decode("utf-8"))
    
    prepare_samples(manager, all_files)

    logging.info("Creating manifests...")
    create_manifest(
        data_path=target_dir,
        output_name='voxforge_train_manifest.json',
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )