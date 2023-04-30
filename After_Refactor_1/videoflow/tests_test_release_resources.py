import pytest

from videoflow.utils.downloader import get_file
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.processors.vision.detectors import BASE_URL_DETECTION

LABELS_FILENAME_TEMPLATE = 'labels_{}.pbtxt'

def download_labels_file(dataset_id):
    filename = LABELS_FILENAME_TEMPLATE.format(dataset_id)
    url_path = BASE_URL_DETECTION + filename
    get_file(filename, url_path)

def test_bboxannotator_resources():
    for dataset_id in BoundingBoxAnnotator.supported_datasets:
        download_labels_file(dataset_id)

if __name__ == "__main__":
    pytest.main([__file__])
