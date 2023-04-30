import numpy as np
import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker
from videoflow.processors.vision.annotators import TrackerAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = "intersection.mp4"
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

def filter_bounding_boxes(dets, class_indexes_to_keep):
    f = np.array([dets[:, 4] == a for a in class_indexes_to_keep])
    f = np.any(f, axis=0)
    filtered = dets[f]
    return filtered

def split_frame_index(data):
    index, frame = data
    return frame

def main():
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"
    
    reader = VideofileReader(input_file)
    frame = split_frame_index(reader)
    detector = TensorflowObjectDetector(num_classes=2, architecture='fasterrcnn-resnet101', dataset='kitti', nb_tasks=1)(frame)
    detector_filtered = BoundingBoxesFilter([2, 3, 4, 6])(detector)
    tracker = KalmanFilterBoundingBoxTracker()(detector_filtered)
    annotator = TrackerAnnotator()(frame, tracker)
    writer = VideofileWriter(output_file, fps=30)(annotator)
    fl = flow.Flow([reader], [writer], flow_type=BATCH)
    fl.run()
    fl.join()

if __name__ == "__main__":
    main()
