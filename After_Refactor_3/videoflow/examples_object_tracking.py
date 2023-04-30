import numpy as np
from videoflow.consumers import VideofileWriter
from videoflow.core.constants import BATCH
import videoflow.core.flow as flow
from videoflow.producers import VideofileReader
from videoflow.utils.downloader import get_file
import videoflow

from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker
from videoflow.processors.vision.annotators import TrackerAnnotator

# Constants
BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = "intersection.mp4"
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME
CLASS_INDEXES_TO_KEEP = [1]  # Will keep only automobile classes

def read_video(input_file):
    """
    Reads the input video file.
    """
    reader = VideofileReader(input_file)
    return reader

def filter_bounding_boxes(dets, class_indexes_to_keep):
    """
    Filters the bounding boxes to keep only specific classes.

    Arguments:
    - dets: np.array of shape (nb_boxes, 6).
            Specifically (nb_boxes, [xmin, ymin, xmax, ymax, class_index, score])
    - class_indexes_to_keep: List of integer indexes corresponding to the classes to keep.
    """
    f = np.array([dets[:, 4] == a for a in class_indexes_to_keep])
    f = np.any(f, axis=0)
    filtered = dets[f]
    return filtered

def detect_objects(frame):
    """
    Detects objects in the input frame of a video.

    Arguments:
    - frame: input video frame.
    """
    detector = TensorflowObjectDetector(num_classes=2, architecture='fasterrcnn-resnet101', dataset='kitti', nb_tasks=1)(frame)
    return detector

def track_bounding_boxes(detector):
    """
    Tracks the bounding boxes over time.

    Arguments:
    - detector: output of object detection step.
    """
    tracker = KalmanFilterBoundingBoxTracker()(detector)
    return tracker

def annotate_frames(frame, tracker):
    """
    Adds annotations to the input video frame.

    Arguments:
    - frame: input video frame.
    - tracker: output of object tracking step.
    """
    annotator = TrackerAnnotator()(frame, tracker)
    return annotator

def write_video(output_file, annotator):
    """
    Writes the output video to file.

    Arguments:
    - output_file: output video filename.
    - annotator: output of annotation step.
    """
    writer = VideofileWriter(output_file, fps=30)(annotator)
    return writer

def main():
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"

    reader = read_video(input_file)
    frame = videoflow.wrap(reader, name="Video Frames")
    filtered_boxes = filter_bounding_boxes(frame, CLASS_INDEXES_TO_KEEP)
    detected_boxes = detect_objects(filtered_boxes)
    tracker = track_bounding_boxes(detected_boxes)
    annotated_frames = annotate_frames(frame, tracker)
    writer = write_video(output_file, annotated_frames)

    fl = flow.Flow([reader], [writer], flow_type=BATCH)
    fl.run()
    fl.join()

if __name__ == "__main__":
    main()