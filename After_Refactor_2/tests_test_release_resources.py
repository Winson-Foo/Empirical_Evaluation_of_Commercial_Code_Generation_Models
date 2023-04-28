import pytest
from contextlib import contextmanager
from videoflow.utils.downloader import get_file
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.processors.vision.detectors import BASE_URL_DETECTION


@contextmanager
def download_files():
    try:
        for dataset_id in BoundingBoxAnnotator.supported_datasets:
            filename = f"labels_{dataset_id}.pbtxt"
            url_path = BASE_URL_DETECTION + filename
            get_file(filename, url_path)
            yield
    finally:
        for dataset_id in BoundingBoxAnnotator.supported_datasets:
            filename = f"labels_{dataset_id}.pbtxt"
            get_file(filename, delete=True)


def test_bboxannotator_resources():
    with download_files():
        pass


if __name__ == "__main__":
    pytest.main([__file__])