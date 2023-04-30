# implementation code

from videoflow.utils.downloader import get_file
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.processors.vision.detectors import BASE_URL_DETECTION

def download_bboxannotator_resources():
    for datasetid in BoundingBoxAnnotator.supported_datasets:
        filename = f'labels_{datasetid}.pbtxt'
        url_path = BASE_URL_DETECTION + filename
        get_file(filename, url_path)

# test code

import pytest

def test_bboxannotator_resources_are_present():
    download_bboxannotator_resources()
    # TODO: assert that resources are present in repo releases and can be downloaded

if __name__ == "__main__":
    pytest.main([__file__])