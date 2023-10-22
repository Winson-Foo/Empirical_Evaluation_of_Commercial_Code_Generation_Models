from typing import Optional
import json
import os
from multiprocessing import Pool
from pathlib import Path

import sox
from tqdm import tqdm

def create_manifest(
    data_path: str,
    output_name: str,
    manifest_path: str,
    num_workers: int,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    file_extension: str = "wav"
) -> None:
    """
    Create a manifest file containing paths to wav files and their transcripts.

    Args:
        data_path: Path to the directory containing audio and transcript files.
        output_name: Name of the output manifest file.
        manifest_path: Path to the directory where the output manifest file will be saved.
        num_workers: Number of worker processes to use for file processing.
        min_duration: Minimum duration of audio files to include in the manifest.
        max_duration: Maximum duration of audio files to include in the manifest.
        file_extension: File extension of the audio files.

    Returns:
        None
    """

    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))
    # Order and prune files based on their duration
    file_paths = order_and_prune_files(
        file_paths=file_paths,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )

    # Create output directory if it doesn't exist
    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    manifest = {'root_path': data_path, 'samples': []}
    for wav_path in tqdm(file_paths, total=len(file_paths)):
        wav_path = wav_path.relative_to(data_path)
        transcript_path = wav_path.parent.with_name("txt") / wav_path.with_suffix(".txt").name
        manifest['samples'].append({
            'wav_path': wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })

    # Write manifest to a file
    with open(output_path, 'w') as f:
        json.dump(manifest, f)

def order_and_prune_files(
    file_paths,
    min_duration: Optional[float],
    max_duration: Optional[float],
    num_workers: int
) -> list:
    """
    Process audio files and sort them based on their duration.

    Args:
        file_paths: List of paths to audio files to be processed.
        min_duration: Minimum duration of audio files to include in the output list.
        max_duration: Maximum duration of audio files to include in the output list.
        num_workers: Number of worker processes to use for file processing.

    Returns:
        A list of file paths sorted in increasing order based on file duration.
    """

    # Gather durations of all files
    with Pool(processes=num_workers) as p:
        duration_file_paths = list(
            tqdm(
                p.imap(_duration_file_path, file_paths),
                total=len(file_paths),
                desc="Gathering durations"
            )
        )

    # Order file paths based on duration
    duration_file_paths.sort(key=lambda x: x[1])
    
    # Prune files based on their duration
    if min_duration and max_duration:
        pruned_duration_file_paths = [
            (path, duration) for path, duration in duration_file_paths 
            if duration >= min_duration and duration <= max_duration
        ]
        # If no files are left after pruning, raise an error
        if not pruned_duration_file_paths:
            raise ValueError("No files found within the specified duration range")
        duration_file_paths = pruned_duration_file_paths
    
    # Remove duration from file paths
    return [x[0] for x in duration_file_paths]

def _duration_file_path(path):
    """
    Utility function to return the duration of an audio file.

    Args:
        path: Path to the input audio file.

    Returns:
        A tuple containing the file path and its duration.
    """
    return path, sox.file_info.duration(path)