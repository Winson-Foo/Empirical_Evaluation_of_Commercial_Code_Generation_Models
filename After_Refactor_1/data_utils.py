from __future__ import print_function

import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import sox
from tqdm import tqdm


WAV_EXTENSION = "wav"
TXT_EXTENSION = "txt"
MANIFEST_ROOT_PATH_KEY = "root_path"
MANIFEST_SAMPLES_KEY = "samples"


def _duration_file_path(path: Path) -> Tuple[Path, float]:
    return path, sox.file_info.duration(str(path))


def gather_durations(file_paths: List[Path], num_workers: int) -> List[Tuple[Path, float]]:
    with Pool(processes=num_workers) as p:
        duration_file_paths = list(tqdm(p.imap(_duration_file_path, file_paths), total=len(file_paths)))
    return duration_file_paths


def sort_manifests(duration_file_paths: List[Tuple[Path, float]], min_duration: Optional[float], max_duration: Optional[float]) -> List[Path]:
    if min_duration is not None and max_duration is not None:
        filtered_duration_file_paths = [(path, duration) for path, duration in duration_file_paths if min_duration <= duration <= max_duration]
    else:
        filtered_duration_file_paths = duration_file_paths

    sorted_paths = [path for path, _ in sorted(filtered_duration_file_paths, key=lambda x: x[1])]
    return sorted_paths


def create_sample_manifest(data_path: Path, output_name: str, manifest_path: Path, num_workers: int, min_duration: Optional[float] = None, max_duration: Optional[float] = None, file_extension: str = WAV_EXTENSION):
    data_path = data_path.resolve()
    file_paths = list(data_path.rglob(f"*.{file_extension}"))
    duration_file_paths = gather_durations(file_paths, num_workers)
    sorted_paths = sort_manifests(duration_file_paths, min_duration, max_duration)

    output_path = manifest_path / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    manifest = {
        MANIFEST_ROOT_PATH_KEY: str(data_path),
        MANIFEST_SAMPLES_KEY: [],
    }
    for wav_path in tqdm(sorted_paths, total=len(sorted_paths)):
        wav_rel_path = wav_path.relative_to(data_path)
        transcript_path = data_path / wav_rel_path.parent / "txt" / (wav_rel_path.stem + "." + TXT_EXTENSION)

        manifest[MANIFEST_SAMPLES_KEY].append({
            'wav_path': str(wav_rel_path),
            'transcript_path': str(transcript_path),
        })

    output_path.write_text(json.dumps(manifest), encoding='utf8')