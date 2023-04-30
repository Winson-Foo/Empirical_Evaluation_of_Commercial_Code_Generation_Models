from __future__ import print_function

import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import sox
from tqdm import tqdm


MIN_DURATION: Optional[float] = None
MAX_DURATION: Optional[float] = None
FILE_EXTENSION: str = "wav"


def create_manifest(
        data_path: str,
        output_name: str,
        manifest_path: str,
        num_workers: int,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        file_extension: str = FILE_EXTENSION) -> None:
    """
    Create a manifest file for audio data.

    Args:
    - data_path (str): Path to directory of audio data.
    - output_name (str): Name of output manifest file.
    - manifest_path (str): Path to output directory of manifest file.
    - num_workers (int): Number of worker processes to use for multiprocessing.
    - min_duration (float): Minimum duration of audio files to include in manifest (optional).
    - max_duration (float): Maximum duration of audio files to include in manifest (optional).
    - file_extension (str): File extension of audio data (default "wav").

    Returns:
    - None
    """
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))
    file_paths = order_and_prune_files(
        file_paths=file_paths,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    manifest = {
        'root_path': data_path,
        'samples': []
    }

    for audio_file_path in tqdm(file_paths, total=len(file_paths)):
        audio_file_path = audio_file_path.relative_to(data_path)
        transcript_path = audio_file_path.parent.with_name("txt") / audio_file_path.with_suffix(".txt").name
        manifest['samples'].append({
            'wav_path': audio_file_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })

    output_path.write_text(json.dumps(manifest), encoding='utf8')


def _get_duration_file_path(path: str) -> Tuple[str, float]:
    """
    Helper function to get the duration of an audio file.

    Args:
    - path (str): Path to audio file.

    Returns:
    - Tuple[str,float]: Tuple containing audio file path and duration.
    """
    return path, sox.file_info.duration(path)


def order_and_prune_files(
        file_paths: List[str],
        min_duration: Optional[float],
        max_duration: Optional[float],
        num_workers: int) -> List[str]:
    """
    Order and prune audio files by duration.

    Args:
    - file_paths (List[str]): List of file paths to audio files.
    - min_duration (float): Minimum duration of audio files to include in manifest (optional).
    - max_duration (float): Maximum duration of audio files to include in manifest (optional).
    - num_workers (int): Number of worker processes to use for multiprocessing.

    Returns:
    - List[str]: Ordered and pruned list of file paths to audio files.
    """
    print("Gathering durations...")

    with Pool(processes=num_workers) as pool:
        duration_file_paths = list(tqdm(pool.imap(_get_duration_file_path, file_paths), total=len(file_paths)))

    print("Sorting manifests...")

    if min_duration and max_duration:
        print(f"Pruning manifests between {min_duration} and {max_duration} seconds")
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    total_duration = sum([x[1] for x in duration_file_paths])
    print(f"Total duration of split: {total_duration:.4f}s")

    return [x[0] for x in duration_file_paths]